import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.amp import autocast, GradScaler
import math
import time
import random
from typing import List
from dataclasses import dataclass

from ps4_data.ps4_dataset_processor import PS4DatasetProcessor
from cb513_data.cb513_dataset_processor import CB513DatasetProcessor
from esm_embeddings import ESMEmbeddings


DSSP8_CHARS = list("BCEGHIST")
DSSP8_TO_IDX = {ss: i for i, ss in enumerate(DSSP8_CHARS)}
IDX_TO_DSSP8 = {i: ss for ss, i in DSSP8_TO_IDX.items()}
PAD_LABEL = -100


@dataclass
class ModelConfig:
    esm_dim: int = 320
    num_classes: int = 8
    max_seq_len: int = 1723  # Hardcoded max length from PS4
    num_workers: int = 4

    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-6

    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    batch_size: int = 64
    learning_rate: float = 3e-3
    epochs: int = 10
    warmup_steps: int = 600
    label_smoothing: float = 0.05
    lstm_hidden: int = 128
    lstm_layers: int = 1

    @classmethod
    def quick(cls):
        return cls(
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=256,
            dropout=0.1,
            batch_size=64,
            learning_rate=3e-3,
            epochs=10,
            warmup_steps=600,
            lstm_hidden=128,
            lstm_layers=1,
            label_smoothing=0.05,
        )

    @classmethod
    def medium(cls):
        return cls(
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ff=512,
            dropout=0.1,
            batch_size=32,
            learning_rate=1e-3,
            epochs=15,
            warmup_steps=1000,
            lstm_hidden=256,
            lstm_layers=2,
            label_smoothing=0.05,
        )

    @classmethod
    def full(cls):
        return cls(
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            dropout=0.1,
            batch_size=16,
            learning_rate=5e-4,
            epochs=50,
            warmup_steps=1000,
            lstm_hidden=512,
            lstm_layers=2,
            label_smoothing=0.05,
        )


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class ProteinDatasetESM(Dataset):
    def __init__(self, embeddings: List[torch.Tensor], target_seqs: List[str]):
        self.embeddings = embeddings
        self.target_seqs = target_seqs
        self.lengths = [emb.size(0) for emb in embeddings]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        tgt = self.target_seqs[idx]
        target_ids = torch.tensor([DSSP8_TO_IDX[c] for c in tgt])
        return emb, target_ids, emb.size(0)


def collate_fn(batch):
    embeddings, targets, lengths = zip(*batch)
    max_len = max(lengths)
    batch_size = len(embeddings)
    esm_dim = embeddings[0].size(1)

    padded_emb = torch.zeros((batch_size, max_len, esm_dim), dtype=torch.float)
    padded_targets = torch.full((batch_size, max_len), PAD_LABEL, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, (emb, tgt, length) in enumerate(zip(embeddings, targets, lengths)):
        padded_emb[i, :length, :] = emb
        padded_targets[i, :length] = tgt
        attention_mask[i, :length] = True

    return padded_emb, padded_targets, attention_mask


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        d_model,
        num_classes,
        lstm_hidden,
        lstm_layers,
        dropout,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.bilstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            x, _ = self.bilstm(x)
        x = self.lstm_norm(x)
        return self.classifier(x)


class ProteinStructureTransformerESM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Sequential(
            nn.Linear(config.esm_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
        )

        self.pos_encoding = LearnablePositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
            mask_check=False,
        )

        self.classifier = BiLSTMClassifier(
            d_model=config.d_model,
            num_classes=config.num_classes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dropout=config.dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "bilstm" not in name and "pe" not in name:
                nn.init.xavier_uniform_(p)

    def forward(self, esm_embeddings, attention_mask):
        x = self.input_proj(esm_embeddings)
        x = self.pos_encoding(x)
        src_key_padding_mask = ~attention_mask
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.classifier(x, attention_mask)


def compute_metrics(logits, padded_targets, attention_mask):
    preds = logits.argmax(dim=-1)
    correct = ((preds == padded_targets) & attention_mask).sum().item()
    total = attention_mask.sum().item()
    return correct / total if total > 0 else 0.0


def train_epoch(model, loader, optimizer, scheduler, scaler, device, label_smoothing):
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for padded_emb, padded_targets, attention_mask in loader:
        padded_emb = padded_emb.to(device, non_blocking=True)
        padded_targets = padded_targets.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", dtype=torch.float16):
            logits = model(padded_emb, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                padded_targets.view(-1),
                ignore_index=PAD_LABEL,
                label_smoothing=label_smoothing,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_acc += compute_metrics(logits, padded_targets, attention_mask)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for padded_emb, padded_targets, attention_mask in loader:
        padded_emb = padded_emb.to(device, non_blocking=True)
        padded_targets = padded_targets.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.float16):
            logits = model(padded_emb, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                padded_targets.view(-1),
                ignore_index=PAD_LABEL,
            )

        total_loss += loss.item()
        total_acc += compute_metrics(logits, padded_targets, attention_mask)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config, train_dataset, val_dataset, device):
    print("\nTraining Configuration")
    print(f"ESM dim: {config.esm_dim} -> d_model: {config.d_model}")
    print(
        f"n_heads: {config.n_heads}, n_layers: {config.n_layers}, d_ff: {config.d_ff}"
    )
    print(
        f"batch_size: {config.batch_size}, lr: {config.learning_rate}, epochs: {config.epochs}"
    )
    print(f"BiLSTM: hidden={config.lstm_hidden}, layers={config.lstm_layers}")

    train_sampler = SortedBatchSampler(
        train_dataset.lengths, config.batch_size, shuffle=True
    )
    val_sampler = SortedBatchSampler(
        val_dataset.lengths, config.batch_size, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    model = ProteinStructureTransformerESM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=0.01,
        fused=True,
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, total_steps
    )
    scaler = GradScaler("cuda")

    best_val_acc = 0.0
    best_model_state = None
    start_time = time.time()

    print(f"\nStarting training for {config.epochs} epochs")

    for epoch in range(config.epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            config.label_smoothing,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1:3d}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Total training time: {(time.time() - start_time) / 60:.1f} minutes")

    model.load_state_dict(best_model_state)
    return model


def main():
    device = torch.device("cuda")
    torch.manual_seed(1729)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision("high")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nLoading datasets and ESM embeddings")
    ps4 = PS4DatasetProcessor.load("ps4_data/ps4_dataset.pt")
    ps4_emb = ESMEmbeddings.load("ps4_data/ps4_esm_embeddings.pt")
    cb513 = CB513DatasetProcessor.load("cb513_data/cb513_dataset.pt")
    cb513_emb = ESMEmbeddings.load("cb513_data/cb513_esm_embeddings.pt")

    n_total = len(ps4.input_seqs)
    n_val = int(0.2 * n_total)
    indices = torch.randperm(n_total).tolist()
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    train_dataset = ProteinDatasetESM(
        [ps4_emb.embeddings[i] for i in train_idx],
        [ps4.dssp8_seqs[i] for i in train_idx],
    )
    val_dataset = ProteinDatasetESM(
        [ps4_emb.embeddings[i] for i in val_idx],
        [ps4.dssp8_seqs[i] for i in val_idx],
    )
    test_dataset = ProteinDatasetESM(cb513_emb.embeddings, cb513.dssp8_seqs)

    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    choice = input("Enter choice of training mode (1/2/3): ").strip() or "1"

    if choice == "1":
        config = ModelConfig.quick()
    elif choice == "2":
        config = ModelConfig.medium()
    else:
        config = ModelConfig.full()

    model = train(config, train_dataset, val_dataset, device)

    test_sampler = SortedBatchSampler(
        test_dataset.lengths, config.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"CB513 Test Loss: {test_loss:.4f}")
    print(f"CB513 Test Accuracy (Q8): {test_acc:.4f} ({test_acc * 100:.2f}%)")

    save_path = f"psp_esm_model_{choice}.pt"
    torch.save(
        {
            "config": config,
            "model_state": model.state_dict(),
            "test_accuracy": test_acc,
        },
        save_path,
    )
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
