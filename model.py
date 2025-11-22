import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import math
import time
from typing import List
from dataclasses import dataclass

from ps4_data.ps4_dataset_processor import PS4DatasetProcessor
from cb513_data.cb513_dataset_processor import CB513DatasetProcessor


AA_CHARS = list("ACDEFGHIKLMNPQRSTVWXY")
DSSP8_CHARS = list("BCEGHIST")

AA_TO_IDX = {aa: i for i, aa in enumerate(AA_CHARS)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}
DSSP8_TO_IDX = {ss: i for i, ss in enumerate(DSSP8_CHARS)}
IDX_TO_DSSP8 = {i: ss for ss, i in DSSP8_TO_IDX.items()}

PAD_IDX = len(AA_CHARS)
PAD_LABEL = -100


@dataclass
class ModelConfig:
    vocab_size: int = 22
    num_classes: int = 8
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 2048
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 10
    warmup_steps: int = 500

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
            epochs=3,
            warmup_steps=100,
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
            warmup_steps=300,
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
        )


class ProteinDataset(Dataset):
    def __init__(self, input_seqs: List[str], target_seqs: List[str]):
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.lengths = [len(s) for s in input_seqs]

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        inp = self.input_seqs[idx]
        tgt = self.target_seqs[idx]
        input_ids = torch.tensor([AA_TO_IDX.get(c, AA_TO_IDX["X"]) for c in inp])
        target_ids = torch.tensor([DSSP8_TO_IDX[c] for c in tgt])
        return input_ids, target_ids, len(inp)


def collate_fn_nested(batch):
    inputs, targets, lengths = zip(*batch)
    max_len = max(lengths)
    padded_inputs = torch.full((len(inputs), max_len), PAD_IDX, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), PAD_LABEL, dtype=torch.long)
    attention_mask = torch.zeros((len(inputs), max_len), dtype=torch.bool)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, : len(inp)] = inp
        padded_targets[i, : len(tgt)] = tgt
        attention_mask[i, : len(inp)] = 1
    return padded_inputs, padded_targets, attention_mask, lengths


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, L, d_model)
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


class ProteinStructureTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=PAD_IDX
        )
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
            bias=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            norm=None,
            enable_nested_tensor=True,
            mask_check=False,
        )

        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask):
        # input_ids: (B, L), attention_mask: (B, L)
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        # Transformer expects mask: True for positions to mask (i.e., padding)
        src_key_padding_mask = ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.classifier(x)
        seq_lens = attention_mask.sum(dim=1).tolist()
        return logits, seq_lens


def compute_metrics(logits, padded_targets, seq_lens):
    preds = logits.argmax(dim=-1)

    correct = 0
    total = 0
    for i, length in enumerate(seq_lens):
        correct += (preds[i, :length] == padded_targets[i, :length]).sum().item()
        total += length

    return correct / total if total > 0 else 0.0


def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    total_acc = 0
    n_batches = 0

    for padded_inputs, padded_targets, attention_mask, lengths in loader:
        padded_inputs = padded_inputs.to(device)
        padded_targets = padded_targets.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        with autocast("cuda", dtype=torch.float16):
            logits, seq_lens = model(padded_inputs, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                padded_targets.view(-1),
                ignore_index=PAD_LABEL,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_acc += compute_metrics(logits, padded_targets, seq_lens)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    n_batches = 0

    for padded_inputs, padded_targets, attention_mask, lengths in loader:
        padded_inputs = padded_inputs.to(device)
        padded_targets = padded_targets.to(device)
        attention_mask = attention_mask.to(device)

        with autocast("cuda", dtype=torch.float16):
            logits, seq_lens = model(padded_inputs, attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                padded_targets.view(-1),
                ignore_index=PAD_LABEL,
            )

        total_loss += loss.item()
        total_acc += compute_metrics(logits, padded_targets, seq_lens)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config: ModelConfig, train_dataset, val_dataset, device):
    print("\nTraining Configuration:")
    print(
        f"d_model: {config.d_model}, n_heads: {config.n_heads}, n_layers: {config.n_layers}"
    )
    print(f"d_ff: {config.d_ff}, dropout: {config.dropout}")
    print(
        f"batch_size: {config.batch_size}, lr: {config.learning_rate}, epochs: {config.epochs}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn_nested,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn_nested,
        num_workers=0,
        pin_memory=True,
    )

    model = ProteinStructureTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.01
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, total_steps
    )
    scaler = GradScaler("cuda")

    best_val_acc = 0
    best_model_state = None
    start_time = time.time()

    print(f"\nStarting training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1:3d}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s (Total: {total_time / 60:.1f}m)"
        )

    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Total training time: {(time.time() - start_time) / 60:.1f} minutes")

    model.load_state_dict(best_model_state)
    return model


def predict_sequence(model, sequence: str, device) -> str:
    model.eval()
    input_ids = torch.tensor(
        [AA_TO_IDX.get(c, AA_TO_IDX["X"]) for c in sequence], dtype=torch.long
    )
    padded_input = input_ids.unsqueeze(0)
    attention_mask = torch.ones_like(padded_input, dtype=torch.bool)
    padded_input = padded_input.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits, _ = model(padded_input, attention_mask)

    preds = logits[0, : len(sequence)].argmax(dim=-1)
    return "".join(IDX_TO_DSSP8[p.item()] for p in preds)


def main():
    device = torch.device("cuda")
    torch.manual_seed(1729)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nLoading datasets...")
    ps4 = PS4DatasetProcessor.load("ps4_data/ps4_dataset.pt")
    cb513 = CB513DatasetProcessor.load("cb513_data/cb513_dataset.pt")

    n_total = len(ps4.input_seqs)
    n_val = int(0.2 * n_total)
    indices = torch.randperm(n_total).tolist()
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    train_dataset = ProteinDataset(
        [ps4.input_seqs[i] for i in train_idx],
        [ps4.dssp8_seqs[i] for i in train_idx],
    )

    val_dataset = ProteinDataset(
        [ps4.input_seqs[i] for i in val_idx],
        [ps4.dssp8_seqs[i] for i in val_idx],
    )

    test_dataset = ProteinDataset(cb513.input_seqs, cb513.dssp8_seqs)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn_nested,
        num_workers=0,
    )

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"CB513 Test Loss: {test_loss:.4f}")
    print(f"CB513 Test Accuracy (Q8): {test_acc:.4f} ({test_acc * 100:.2f}%)")

    save_path = f"psp_model_{choice}.pt"
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
