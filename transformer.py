import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import time
from protein_dataset_processor import ProteinDatasetProcessor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 0, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class ProteinTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 21,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 2048,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_seq_length, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.q8_head = nn.Linear(d_model, 8)
        self.q3_head = nn.Linear(d_model, 3)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_key_padding_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)

        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )

        q8_logits = self.q8_head(output)
        q3_logits = self.q3_head(output)

        return q8_logits, q3_logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ProteinDataset(Dataset):
    def __init__(self, processor, indices=None):
        self.processor = processor
        self.indices = (
            indices if indices is not None else list(range(len(processor.chain_ids)))
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return {
            "input": self.processor.input_encoded[real_idx],
            "dssp8": self.processor.dssp8_encoded[real_idx],
            "dssp3": self.processor.dssp3_encoded[real_idx],
            "length": self.processor.seq_lengths[real_idx],
            "chain_id": self.processor.chain_ids[real_idx],
        }


def collate_fn(batch):
    inputs = [item["input"] for item in batch]
    dssp8 = [item["dssp8"] for item in batch]
    dssp3 = [item["dssp3"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch])
    chain_ids = [item["chain_id"] for item in batch]

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    dssp8_padded = pad_sequence(dssp8, batch_first=True, padding_value=-1)
    dssp3_padded = pad_sequence(dssp3, batch_first=True, padding_value=-1)

    padding_mask = torch.arange(inputs_padded.size(1))[None, :] >= lengths[:, None]

    return {
        "input": inputs_padded,
        "dssp8": dssp8_padded,
        "dssp3": dssp3_padded,
        "padding_mask": padding_mask,
        "lengths": lengths,
        "chain_ids": chain_ids,
    }


class ProteinStructurePredictor:
    def __init__(self, model_config: Dict = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        default_config = {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 3,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "max_seq_length": 2048,
        }

        if model_config is not None:
            default_config.update(model_config)

        self.model_config = default_config

        self.model = ProteinTransformer(**self.model_config).to(self.device)
        self.history = {
            "train_loss": [],
            "train_q8_acc": [],
            "train_q3_acc": [],
            "val_loss": [],
            "val_q8_acc": [],
            "val_q3_acc": [],
        }

    def train_model(
        self,
        processor,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 0.001,
        train_split: float = 0.8,
        verbose: bool = True,
    ):
        n_samples = len(processor.chain_ids)
        n_train = int(n_samples * train_split)

        indices = torch.randperm(n_samples).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dataset = ProteinDataset(processor, train_indices)
        val_dataset = ProteinDataset(processor, val_indices)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        if verbose:
            print(f"Model Parameters: {self.model.count_parameters():,}")
            print(
                f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)}"
            )
            print(f"Device: {self.device}\n")

        start_time = time.time()

        for epoch in range(epochs):
            self.model.train()
            train_loss, train_q8_correct, train_q3_correct, train_total = 0, 0, 0, 0

            for batch in train_loader:
                inputs = batch["input"].to(self.device)
                dssp8_targets = batch["dssp8"].to(self.device)
                dssp3_targets = batch["dssp3"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)

                optimizer.zero_grad()

                q8_logits, q3_logits = self.model(inputs, padding_mask)

                loss_q8 = criterion(q8_logits.reshape(-1, 8), dssp8_targets.reshape(-1))
                loss_q3 = criterion(q3_logits.reshape(-1, 3), dssp3_targets.reshape(-1))
                loss = loss_q8 + loss_q3

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

                mask = dssp8_targets != -1
                train_q8_correct += (
                    (q8_logits.argmax(-1)[mask] == dssp8_targets[mask]).sum().item()
                )
                train_q3_correct += (
                    (q3_logits.argmax(-1)[mask] == dssp3_targets[mask]).sum().item()
                )
                train_total += mask.sum().item()

            train_loss /= len(train_loader)
            train_q8_acc = 100 * train_q8_correct / train_total
            train_q3_acc = 100 * train_q3_correct / train_total

            val_loss, val_q8_acc, val_q3_acc = self.evaluate(val_loader, criterion)

            self.history["train_loss"].append(train_loss)
            self.history["train_q8_acc"].append(train_q8_acc)
            self.history["train_q3_acc"].append(train_q3_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_q8_acc"].append(val_q8_acc)
            self.history["val_q3_acc"].append(val_q3_acc)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {train_loss:.4f} - Q8: {train_q8_acc:.2f}% - Q3: {train_q3_acc:.2f}% - "
                    f"Val Loss: {val_loss:.4f} - Val Q8: {val_q8_acc:.2f}% - Val Q3: {val_q3_acc:.2f}%"
                )

        elapsed = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {elapsed:.2f} seconds")

    def evaluate(self, data_loader, criterion):
        self.model.eval()
        total_loss, q8_correct, q3_correct, total = 0, 0, 0, 0

        with torch.no_grad():
            for batch in data_loader:
                inputs = batch["input"].to(self.device)
                dssp8_targets = batch["dssp8"].to(self.device)
                dssp3_targets = batch["dssp3"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)

                q8_logits, q3_logits = self.model(inputs, padding_mask)

                loss_q8 = criterion(q8_logits.reshape(-1, 8), dssp8_targets.reshape(-1))
                loss_q3 = criterion(q3_logits.reshape(-1, 3), dssp3_targets.reshape(-1))
                total_loss += (loss_q8 + loss_q3).item()

                mask = dssp8_targets != -1
                q8_correct += (
                    (q8_logits.argmax(-1)[mask] == dssp8_targets[mask]).sum().item()
                )
                q3_correct += (
                    (q3_logits.argmax(-1)[mask] == dssp3_targets[mask]).sum().item()
                )
                total += mask.sum().item()

        return (
            total_loss / len(data_loader),
            100 * q8_correct / total,
            100 * q3_correct / total,
        )

    def predict_sequence(self, processor, chain_id: str):
        self.model.eval()
        sample = processor.get_by_chain_id(chain_id)

        input_tensor = sample["input_encoded"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            q8_logits, q3_logits = self.model(input_tensor)

        q8_pred = q8_logits[0].argmax(-1).cpu()
        q3_pred = q3_logits[0].argmax(-1).cpu()

        q8_pred_str = "".join([processor.IDX_TO_DSSP8[idx.item()] for idx in q8_pred])
        q3_pred_str = "".join([processor.IDX_TO_DSSP3[idx.item()] for idx in q3_pred])

        q8_matches = sum(
            1 for i in range(len(q8_pred_str)) if q8_pred_str[i] == sample["dssp8"][i]
        )
        q3_matches = sum(
            1 for i in range(len(q3_pred_str)) if q3_pred_str[i] == sample["dssp3"][i]
        )

        return {
            "chain_id": chain_id,
            "sequence": sample["input"],
            "true_q8": sample["dssp8"],
            "pred_q8": q8_pred_str,
            "true_q3": sample["dssp3"],
            "pred_q3": q3_pred_str,
            "q8_accuracy": 100 * q8_matches / len(q8_pred_str),
            "q3_accuracy": 100 * q3_matches / len(q3_pred_str),
            "length": len(sample["input"]),
        }

    def plot_training_history(self, save_path: str = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.history["train_loss"]) + 1)

        axes[0].plot(
            epochs, self.history["train_loss"], "b-", label="Train Loss", linewidth=2
        )
        axes[0].plot(
            epochs, self.history["val_loss"], "r-", label="Val Loss", linewidth=2
        )
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title(
            "Training and Validation Loss", fontsize=14, fontweight="bold"
        )
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            epochs, self.history["train_q8_acc"], "b-", label="Train Q8", linewidth=2
        )
        axes[1].plot(
            epochs, self.history["val_q8_acc"], "r-", label="Val Q8", linewidth=2
        )
        axes[1].plot(
            epochs, self.history["train_q3_acc"], "b--", label="Train Q3", linewidth=2
        )
        axes[1].plot(
            epochs, self.history["val_q3_acc"], "r--", label="Val Q3", linewidth=2
        )
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy (%)", fontsize=12)
        axes[1].set_title("Q8 and Q3 Accuracy", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def analyze_predictions(self, processor, num_samples: int = 100):
        """
        Analyze model predictions across the dataset to find patterns.
        Returns best/worst performing samples and structural insights.
        """
        print("Qualitative Analysis")

        self.model.eval()
        results = []

        indices = torch.randperm(len(processor.chain_ids))[:num_samples].tolist()

        print(f"\nAnalyzing {num_samples} random samples...")
        for idx in indices:
            chain_id = processor.chain_ids[idx]
            pred = self.predict_sequence(processor, chain_id)

            sample = processor.get_by_index(idx)
            true_q8 = sample["dssp8"]
            true_q3 = sample["dssp3"]

            structure_counts_q8 = {k: true_q8.count(k) for k in processor.DSSP8_CHARS}
            structure_counts_q3 = {k: true_q3.count(k) for k in processor.DSSP3_CHARS}

            results.append(
                {
                    "chain_id": chain_id,
                    "q8_accuracy": pred["q8_accuracy"],
                    "q3_accuracy": pred["q3_accuracy"],
                    "length": pred["length"],
                    "helix_frac": structure_counts_q3["H"] / len(true_q3),
                    "sheet_frac": structure_counts_q3["E"] / len(true_q3),
                    "coil_frac": structure_counts_q3["C"] / len(true_q3),
                    "structure_counts_q8": structure_counts_q8,
                    "sequence": sample["input"],
                    "true_q8": true_q8,
                    "pred_q8": pred["pred_q8"],
                    "true_q3": true_q3,
                    "pred_q3": pred["pred_q3"],
                }
            )

        results.sort(key=lambda x: x["q8_accuracy"])

        avg_q8 = np.mean([r["q8_accuracy"] for r in results])
        avg_q3 = np.mean([r["q3_accuracy"] for r in results])

        print(f"\nAverage Q8 Accuracy: {avg_q8:.2f}%")
        print(f"Average Q3 Accuracy: {avg_q3:.2f}%")

        print("\n5 Best Predictions")
        for i, result in enumerate(results[-5:][::-1], 1):
            print(f"\n{i}. Chain: {result['chain_id']}")
            print(f"Length: {result['length']} residues")
            print(
                f"Q8 Accuracy: {result['q8_accuracy']:.2f}% (Δ={result['q8_accuracy'] - avg_q8:+.2f}%)"
            )
            print(f"Q3 Accuracy: {result['q3_accuracy']:.2f}%")
            print(
                f"Structure: H={result['helix_frac'] * 100:.1f}%, E={result['sheet_frac'] * 100:.1f}%, C={result['coil_frac'] * 100:.1f}%"
            )
            print(f"Sequence: {result['sequence'][:60]}...")

        print("\n5 Worst Predictions")
        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. Chain: {result['chain_id']}")
            print(f"Length: {result['length']} residues")
            print(
                f"Q8 Accuracy: {result['q8_accuracy']:.2f}% (Δ={result['q8_accuracy'] - avg_q8:+.2f}%)"
            )
            print(f"Q3 Accuracy: {result['q3_accuracy']:.2f}%")
            print(
                f"Structure: H={result['helix_frac'] * 100:.1f}%, E={result['sheet_frac'] * 100:.1f}%, C={result['coil_frac'] * 100:.1f}%"
            )
            print(f"Sequence: {result['sequence'][:60]}...")

        short_seqs = [r for r in results if r["length"] < 100]
        medium_seqs = [r for r in results if 100 <= r["length"] < 300]
        long_seqs = [r for r in results if r["length"] >= 300]

        if short_seqs:
            print(f"\nShort Sequences (<100 residues, n={len(short_seqs)}):")
            print(f"Avg Q8: {np.mean([r['q8_accuracy'] for r in short_seqs]):.2f}%")
            print(f"Avg Q3: {np.mean([r['q3_accuracy'] for r in short_seqs]):.2f}%")

        if medium_seqs:
            print(f"\nMedium Sequences (100-300 residues, n={len(medium_seqs)}):")
            print(f"Avg Q8: {np.mean([r['q8_accuracy'] for r in medium_seqs]):.2f}%")
            print(f"Avg Q3: {np.mean([r['q3_accuracy'] for r in medium_seqs]):.2f}%")

        if long_seqs:
            print(f"\nLong Sequences (>300 residues, n={len(long_seqs)}):")
            print(f"Avg Q8: {np.mean([r['q8_accuracy'] for r in long_seqs]):.2f}%")
            print(f"Avg Q3: {np.mean([r['q3_accuracy'] for r in long_seqs]):.2f}%")

        helix_rich = [r for r in results if r["helix_frac"] > 0.5]
        sheet_rich = [r for r in results if r["sheet_frac"] > 0.4]
        coil_rich = [r for r in results if r["coil_frac"] > 0.5]

        if helix_rich:
            print(f"\nHelix-Rich Proteins (>50% helix, n={len(helix_rich)}):")
            print(f"Avg Q8: {np.mean([r['q8_accuracy'] for r in helix_rich]):.2f}%")
            print(f"Avg Q3: {np.mean([r['q3_accuracy'] for r in helix_rich]):.2f}%")

        if sheet_rich:
            print(f"\nSheet-Rich Proteins (>40% sheet, n={len(sheet_rich)}):")
            print(f"Avg Q8: {np.mean([r['q8_accuracy'] for r in sheet_rich]):.2f}%")
            print(f"Avg Q3: {np.mean([r['q3_accuracy'] for r in sheet_rich]):.2f}%")

        if coil_rich:
            print(f"\nCoil-Rich Proteins (>50% coil, n={len(coil_rich)}):")
            print(f"Avg Q8: {np.mean([r['q8_accuracy'] for r in coil_rich]):.2f}%")
            print(f"Avg Q3: {np.mean([r['q3_accuracy'] for r in coil_rich]):.2f}%")

        length_corr = np.corrcoef(
            [r["length"] for r in results], [r["q8_accuracy"] for r in results]
        )[0, 1]
        print(f"\nLength vs Accuracy Correlation: {length_corr:.3f}")

        helix_corr = np.corrcoef(
            [r["helix_frac"] for r in results], [r["q8_accuracy"] for r in results]
        )[0, 1]
        print(f"\nHelix Content vs Accuracy Correlation: {helix_corr:.3f}")

        sheet_corr = np.corrcoef(
            [r["sheet_frac"] for r in results], [r["q8_accuracy"] for r in results]
        )[0, 1]
        print(f"\nSheet Content vs Accuracy Correlation: {sheet_corr:.3f}")

        coil_corr = np.corrcoef(
            [r["coil_frac"] for r in results], [r["q8_accuracy"] for r in results]
        )[0, 1]
        print(f"\nCoil Content vs Accuracy Correlation: {coil_corr:.3f}")

        return results

    def save_model(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model_config,
                "history": self.history,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model_config = checkpoint["model_config"]
        self.model = ProteinTransformer(**self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint["history"]
        print(f"Model loaded from {path}")


def main():
    pt_path = "ps4_dataset.pt"
    processor = ProteinDatasetProcessor.load(pt_path)
    predictor_baseline = ProteinStructurePredictor()

    """
    print("Baseline Model, trains in ~6 minutes on my GTX 1650 with 4GB VRAM")
    predictor_baseline.train_model(
        processor,
        epochs=5,
        batch_size=8,
        learning_rate=0.001,
        train_split=0.8
    )
    predictor_baseline.plot_training_history(save_path="training_curves_baseline.png")
    predictor_baseline.save_model("protein_transformer_baseline.pt")
    """

    """
    predictor = ProteinStructurePredictor(model_config={
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 1024,
        'dropout': 0.1
    })
    predictor.train_model(
        processor,
        epochs=20,
        batch_size=16,
        learning_rate=0.0005,
        train_split=0.8
    )
    predictor.plot_training_history(save_path='training_curves_full.png')
    predictor.save_model('protein_transformer_full.pt')
    """

    predictor_baseline.load_model(path="protein_transformer_baseline.pt")
    predictor_baseline.plot_training_history(save_path="training_curves_baseline.png")

    analysis_results = predictor_baseline.analyze_predictions(
        processor, num_samples=100
    )


if __name__ == "__main__":
    main()
