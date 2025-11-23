import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Dict
import os

from model import (
    TCGNet,
    ProteinDatasetESM,
    SortedBatchSampler,
    collate_fn,
    DSSP8_TO_IDX,
)
from ps4_data.ps4_dataset_processor import PS4DatasetProcessor
from cb513_data.cb513_dataset_processor import CB513DatasetProcessor
from esm_embeddings import ESMEmbeddings

DSSP8_CHARS = ["B", "C", "E", "G", "H", "I", "S", "T"]
DSSP3_CHARS = ["C", "E", "H"]
DSSP8_TO_DSSP3 = {
    **{k: "C" for k in "CST"},
    **{k: "E" for k in "EB"},
    **{k: "H" for k in "GHI"},
}


def convert_q8_to_q3(dssp8_seq: str) -> str:
    return "".join(DSSP8_TO_DSSP3[c] for c in dssp8_seq)


def compute_q3_accuracy(logits, targets, attention_mask):
    preds = logits.argmax(dim=-1)

    batch_size, seq_len = preds.shape
    correct = 0
    total = 0

    for b in range(batch_size):
        for i in range(seq_len):
            if attention_mask[b, i]:
                pred_idx = preds[b, i].item()
                target_idx = targets[b, i].item()

                pred_char = DSSP8_CHARS[pred_idx]
                target_char = DSSP8_CHARS[target_idx]

                pred_q3 = DSSP8_TO_DSSP3[pred_char]
                target_q3 = DSSP8_TO_DSSP3[target_char]

                if pred_q3 == target_q3:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_model(model, loader, device, compute_q3=True):
    model.eval()

    total_q8_correct = 0
    total_q3_correct = 0
    total_positions = 0

    for padded_emb, padded_targets, attention_mask in loader:
        padded_emb = padded_emb.to(device, non_blocking=True)
        padded_targets = padded_targets.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        logits = model(padded_emb, attention_mask)
        preds = logits.argmax(dim=-1)

        q8_correct = ((preds == padded_targets) & attention_mask).sum().item()
        total_q8_correct += q8_correct

        if compute_q3:
            q3_correct = compute_q3_accuracy(logits, padded_targets, attention_mask)
            total_q3_correct += q3_correct * attention_mask.sum().item()

        total_positions += attention_mask.sum().item()

    q8_acc = total_q8_correct / total_positions if total_positions > 0 else 0.0
    q3_acc = (
        total_q3_correct / total_positions
        if total_positions > 0 and compute_q3
        else 0.0
    )

    return q8_acc, q3_acc


def analyze_protein_composition(dssp8_seq: str) -> Dict[str, float]:
    dssp3_seq = convert_q8_to_q3(dssp8_seq)
    total = len(dssp3_seq)

    helix_count = dssp3_seq.count("H")
    sheet_count = dssp3_seq.count("E")
    coil_count = dssp3_seq.count("C")

    return {
        "helix": helix_count / total,
        "sheet": sheet_count / total,
        "coil": coil_count / total,
        "length": total,
    }


@torch.no_grad()
def evaluate_by_composition_and_length(model, dataset, embeddings, device, sample_size):
    model.eval()

    sample_indices = torch.randperm(len(dataset))[:sample_size].tolist()

    results_by_helix_pct = defaultdict(
        lambda: {"q8_correct": 0, "q3_correct": 0, "total": 0, "count": 0}
    )
    results_by_sheet_pct = defaultdict(
        lambda: {"q8_correct": 0, "q3_correct": 0, "total": 0, "count": 0}
    )
    results_by_coil_pct = defaultdict(
        lambda: {"q8_correct": 0, "q3_correct": 0, "total": 0, "count": 0}
    )
    results_by_length = defaultdict(
        lambda: {"q8_correct": 0, "q3_correct": 0, "total": 0, "count": 0}
    )

    individual_results = []

    for idx in sample_indices:
        emb = embeddings[idx]
        tgt_seq = dataset[idx]

        target_ids = torch.tensor([DSSP8_TO_IDX[c] for c in tgt_seq])
        emb_batch = emb.unsqueeze(0).to(device)
        target_batch = target_ids.unsqueeze(0).to(device)
        mask_batch = torch.ones((1, len(tgt_seq)), dtype=torch.bool).to(device)

        logits = model(emb_batch, mask_batch)
        preds = logits.argmax(dim=-1)

        q8_correct = (preds == target_batch).sum().item()
        q8_acc = q8_correct / len(tgt_seq)

        q3_correct = 0
        for i in range(len(tgt_seq)):
            pred_char = DSSP8_CHARS[preds[0, i].item()]
            target_char = DSSP8_CHARS[target_batch[0, i].item()]
            if DSSP8_TO_DSSP3[pred_char] == DSSP8_TO_DSSP3[target_char]:
                q3_correct += 1
        q3_acc = q3_correct / len(tgt_seq)

        composition = analyze_protein_composition(tgt_seq)
        length = composition["length"]

        helix_pct = int(composition["helix"] * 100)
        helix_bin = f"{(helix_pct // 10) * 10}-{(helix_pct // 10) * 10 + 10}%"

        sheet_pct = int(composition["sheet"] * 100)
        sheet_bin = f"{(sheet_pct // 10) * 10}-{(sheet_pct // 10) * 10 + 10}%"

        coil_pct = int(composition["coil"] * 100)
        coil_bin = f"{(coil_pct // 10) * 10}-{(coil_pct // 10) * 10 + 10}%"

        if length < 100:
            length_bin = "0-100"
        elif length < 200:
            length_bin = "100-200"
        elif length < 300:
            length_bin = "200-300"
        elif length < 500:
            length_bin = "300-500"
        else:
            length_bin = "500+"

        results_by_helix_pct[helix_bin]["q8_correct"] += q8_correct
        results_by_helix_pct[helix_bin]["q3_correct"] += q3_correct
        results_by_helix_pct[helix_bin]["total"] += len(tgt_seq)
        results_by_helix_pct[helix_bin]["count"] += 1

        results_by_sheet_pct[sheet_bin]["q8_correct"] += q8_correct
        results_by_sheet_pct[sheet_bin]["q3_correct"] += q3_correct
        results_by_sheet_pct[sheet_bin]["total"] += len(tgt_seq)
        results_by_sheet_pct[sheet_bin]["count"] += 1

        results_by_coil_pct[coil_bin]["q8_correct"] += q8_correct
        results_by_coil_pct[coil_bin]["q3_correct"] += q3_correct
        results_by_coil_pct[coil_bin]["total"] += len(tgt_seq)
        results_by_coil_pct[coil_bin]["count"] += 1

        results_by_length[length_bin]["q8_correct"] += q8_correct
        results_by_length[length_bin]["q3_correct"] += q3_correct
        results_by_length[length_bin]["total"] += len(tgt_seq)
        results_by_length[length_bin]["count"] += 1

        individual_results.append(
            {
                "idx": idx,
                "q8_acc": q8_acc,
                "q3_acc": q3_acc,
                "length": length,
                "helix_pct": composition["helix"] * 100,
                "sheet_pct": composition["sheet"] * 100,
                "coil_pct": composition["coil"] * 100,
                "composition": composition,
            }
        )

    for bin_dict in [
        results_by_helix_pct,
        results_by_sheet_pct,
        results_by_coil_pct,
        results_by_length,
    ]:
        for bin_name in bin_dict:
            total = bin_dict[bin_name]["total"]
            if total > 0:
                bin_dict[bin_name]["q8_acc"] = bin_dict[bin_name]["q8_correct"] / total
                bin_dict[bin_name]["q3_acc"] = bin_dict[bin_name]["q3_correct"] / total

    individual_results.sort(key=lambda x: x["q8_acc"], reverse=True)
    best_performer = individual_results[0] if individual_results else None
    worst_performer = individual_results[-1] if individual_results else None

    return (
        dict(results_by_helix_pct),
        dict(results_by_sheet_pct),
        dict(results_by_coil_pct),
        dict(results_by_length),
        best_performer,
        worst_performer,
    )


def load_model_and_evaluate(model_path: str, device: torch.device):
    print("Loading model")
    checkpoint = torch.load(model_path, weights_only=False)
    config = checkpoint["config"]

    model = TCGNet(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Loading datasets")
    ps4 = PS4DatasetProcessor.load("ps4_data/ps4_dataset.pt")
    ps4_emb = ESMEmbeddings.load("ps4_data/ps4_esm_embeddings.pt")
    cb513 = CB513DatasetProcessor.load("cb513_data/cb513_dataset.pt")
    cb513_emb = ESMEmbeddings.load("cb513_data/cb513_esm_embeddings.pt")

    # Create datasets
    ps4_dataset = ProteinDatasetESM(ps4_emb.embeddings, ps4.dssp8_seqs)
    cb513_dataset = ProteinDatasetESM(cb513_emb.embeddings, cb513.dssp8_seqs)

    # Create dataloaders
    ps4_sampler = SortedBatchSampler(
        ps4_dataset.lengths, config.batch_size, shuffle=False
    )
    cb513_sampler = SortedBatchSampler(
        cb513_dataset.lengths, config.batch_size, shuffle=False
    )

    ps4_loader = DataLoader(
        ps4_dataset,
        batch_sampler=ps4_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    cb513_loader = DataLoader(
        cb513_dataset,
        batch_sampler=cb513_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    ps4_q8, ps4_q3 = evaluate_model(model, ps4_loader, device)
    print(f"\nPS/4 Q8 Accuracy: {ps4_q8:.4f} ({ps4_q8 * 100:.2f}%)")
    print(f"PS/4 Q3 Accuracy: {ps4_q3:.4f} ({ps4_q3 * 100:.2f}%)")

    cb513_q8, cb513_q3 = evaluate_model(model, cb513_loader, device)
    print(f"\nCB513 Q8 Accuracy: {cb513_q8:.4f} ({cb513_q8 * 100:.2f}%)")
    print(f"CB513 Q3 Accuracy: {cb513_q3:.4f} ({cb513_q3 * 100:.2f}%)")

    by_helix, by_sheet, by_coil, by_length, best, worst = (
        evaluate_by_composition_and_length(
            model, ps4.dssp8_seqs, ps4_emb.embeddings, device, sample_size=200
        )
    )

    print("\nPerformance by Helix Percentage:")
    total_proteins = sum(r["count"] for r in by_helix.values())
    for bin_name in sorted(by_helix.keys(), key=lambda x: int(x.split("-")[0])):
        results = by_helix[bin_name]
        print(
            f"{bin_name:10s}: Q8={results['q8_acc']:.4f}, Q3={results['q3_acc']:.4f}, "
            f"proteins={results['count']}, residues={results['total']}"
        )
    print(f"Total proteins: {total_proteins}")

    print("\nPerformance by Sheet Percentage:")
    total_proteins = sum(r["count"] for r in by_sheet.values())
    for bin_name in sorted(by_sheet.keys(), key=lambda x: int(x.split("-")[0])):
        results = by_sheet[bin_name]
        print(
            f"{bin_name:10s}: Q8={results['q8_acc']:.4f}, Q3={results['q3_acc']:.4f}, "
            f"proteins={results['count']}, residues={results['total']}"
        )
    print(f"Total proteins: {total_proteins}")

    print("\nPerformance by Coil Percentage:")
    total_proteins = sum(r["count"] for r in by_coil.values())
    for bin_name in sorted(by_coil.keys(), key=lambda x: int(x.split("-")[0])):
        results = by_coil[bin_name]
        print(
            f"{bin_name:10s}: Q8={results['q8_acc']:.4f}, Q3={results['q3_acc']:.4f}, "
            f"proteins={results['count']}, residues={results['total']}"
        )
    print(f"Total proteins: {total_proteins}")

    print("\nPerformance by Sequence Length:")
    total_proteins = sum(r["count"] for r in by_length.values())
    for lbin in ["0-100", "100-200", "200-300", "300-500", "500+"]:
        if lbin in by_length:
            results = by_length[lbin]
            print(
                f"{lbin:10s}: Q8={results['q8_acc']:.4f}, Q3={results['q3_acc']:.4f}, "
                f"proteins={results['count']}, residues={results['total']}"
            )
    print(f"Total proteins: {total_proteins}")

    if best:
        print("\nBest Performer:")
        print(
            f"Index {best['idx']}: Q8={best['q8_acc']:.4f}, Q3={best['q3_acc']:.4f}, "
            f"Length={best['length']}, Helix={best['helix_pct']:.1f}%, "
            f"Sheet={best['sheet_pct']:.1f}%, Coil={best['coil_pct']:.1f}%"
        )

    if worst:
        print("\nWorst Performer:")
        print(
            f"Index {worst['idx']}: Q8={worst['q8_acc']:.4f}, Q3={worst['q3_acc']:.4f}, "
            f"Length={worst['length']}, Helix={worst['helix_pct']:.1f}%, "
            f"Sheet={worst['sheet_pct']:.1f}%, Coil={worst['coil_pct']:.1f}%"
        )

    statistics = {
        "ps4_q8": ps4_q8,
        "ps4_q3": ps4_q3,
        "cb513_q8": cb513_q8,
        "cb513_q3": cb513_q3,
        "by_helix_pct": by_helix,
        "by_sheet_pct": by_sheet,
        "by_coil_pct": by_coil,
        "by_length": by_length,
        "best_performer": best,
        "worst_performer": worst,
    }

    return statistics


def main():
    device = torch.device("cuda")
    torch.manual_seed(1729)

    model_path = "model.pt"
    stats_path = "model_statistics.pt"

    if os.path.exists(stats_path):
        print(f"Loading pre-computed statistics from {stats_path}")
        statistics = torch.load(stats_path, weights_only=False)

        print("Statistics")
        print(f"PS/4 Q8: {statistics['ps4_q8']:.4f}")
        print(f"PS/4 Q3: {statistics['ps4_q3']:.4f}")
        print(f"CB513 Q8: {statistics['cb513_q8']:.4f}")
        print(f"CB513 Q3: {statistics['cb513_q3']:.4f}")
    else:
        print("Computing statistics")
        statistics = load_model_and_evaluate(model_path, device)
        print(f"\nSaving statistics to {stats_path}")
        torch.save(statistics, stats_path)
        print(f"\nModel saved to {stats_path}")


if __name__ == "__main__":
    main()
