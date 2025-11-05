import pandas as pd
import torch
from collections import Counter
from typing import Dict
import time


class ProteinDatasetProcessor:
    RES_DICT = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "X": "UNK",
        "Y": "TYR",
    }

    AA_CHARS = list(RES_DICT.keys())

    AA3_CHARS = sorted(RES_DICT.values())

    DSSP8_CHARS = ["B", "C", "E", "G", "H", "I", "S", "T"]

    DSSP3_CHARS = ["C", "E", "H"]

    DSSP8_TO_DSSP3 = {
        **{k: "C" for k in "CST"},
        **{k: "E" for k in "EB"},
        **{k: "H" for k in "GHI"},
    }

    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_CHARS)}
    IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}

    DSSP8_TO_IDX = {ss: i for i, ss in enumerate(DSSP8_CHARS)}
    IDX_TO_DSSP8 = {i: ss for ss, i in DSSP8_TO_IDX.items()}

    DSSP3_TO_IDX = {ss: i for i, ss in enumerate(DSSP3_CHARS)}
    IDX_TO_DSSP3 = {i: ss for ss, i in DSSP3_TO_IDX.items()}

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

        self.chain_ids = None
        self.first_res = None
        self.input_seqs = None
        self.dssp8_seqs = None
        self.dssp3_seqs = None

        self.input_encoded = None
        self.dssp8_encoded = None
        self.dssp3_encoded = None
        self.seq_lengths = None

        self.stats = {}
        self.chain_id_to_idx = {}
        self.idx_to_chain_id = {}

    def process(self):
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} proteins")

        self.chain_ids = df["chain_id"].tolist()
        self.first_res = (df["first_res"]).tolist()
        self.input_seqs = df["input"].tolist()
        self.dssp8_seqs = df["dssp8"].tolist()

        self.validate_and_clean()

        self.dssp3_seqs = [
            "".join(self.DSSP8_TO_DSSP3.get(c) for c in seq) for seq in self.dssp8_seqs
        ]

        self.encode_sequences()
        self.create_lookups()
        self.calculate_statistics()

        print("Processing complete!")

    def validate_and_clean(self):
        for i in range(len(self.input_seqs)):
            self.input_seqs[i] = self.input_seqs[i].upper()
            invalid_aa = set(self.input_seqs[i]) - set(self.AA_CHARS)
            if invalid_aa:
                print(
                    f"Warning: Protein {i} ({self.chain_ids[i]}) invalid AA: {invalid_aa}"
                )
                for char in invalid_aa:
                    self.input_seqs[i] = self.input_seqs[i].replace(char, "X")

            invalid_ss = set(self.dssp8_seqs[i]) - set(self.DSSP8_CHARS)
            if invalid_ss:
                print(
                    f"Warning: Protein {i} ({self.chain_ids[i]}) invalid SS: {invalid_ss}"
                )
                for char in invalid_ss:
                    self.dssp8_seqs[i] = self.dssp8_seqs[i].replace(char, "C")

    def encode_sequences(self):
        self.input_encoded, self.dssp8_encoded, self.dssp3_encoded, lengths = (
            [],
            [],
            [],
            [],
        )

        for seq, ss8, ss3 in zip(self.input_seqs, self.dssp8_seqs, self.dssp3_seqs):
            self.input_encoded.append(
                torch.tensor([self.AA_TO_IDX[a] for a in seq], dtype=torch.long)
            )
            self.dssp8_encoded.append(
                torch.tensor([self.DSSP8_TO_IDX[s] for s in ss8], dtype=torch.long)
            )
            self.dssp3_encoded.append(
                torch.tensor([self.DSSP3_TO_IDX[s] for s in ss3], dtype=torch.long)
            )
            lengths.append(len(seq))

        self.seq_lengths = torch.tensor(lengths, dtype=torch.int32)

    def create_lookups(self):
        self.chain_id_to_idx = {cid: i for i, cid in enumerate(self.chain_ids)}
        self.idx_to_chain_id = {i: cid for cid, i in self.chain_id_to_idx.items()}

    def calculate_statistics(self):
        self.stats = {
            "n_proteins": len(self.chain_ids),
            "amino_acid_counts": dict(Counter("".join(self.input_seqs))),
            "dssp8_counts": dict(Counter("".join(self.dssp8_seqs))),
            "dssp3_counts": dict(Counter("".join(self.dssp3_seqs))),
            "input_length_stats": {
                "min": self.seq_lengths.min().item(),
                "max": self.seq_lengths.max().item(),
                "median": self.seq_lengths.float().median().item(),
            },
        }

    def get_by_index(self, idx: int) -> Dict:
        return {
            "index": idx,
            "chain_id": self.chain_ids[idx],
            "first_res": self.first_res[idx],
            "input": self.input_seqs[idx],
            "dssp8": self.dssp8_seqs[idx],
            "dssp3": self.dssp3_seqs[idx],
            "input_encoded": self.input_encoded[idx],
            "dssp8_encoded": self.dssp8_encoded[idx],
            "dssp3_encoded": self.dssp3_encoded[idx],
            "length": self.seq_lengths[idx],
        }

    def get_by_chain_id(self, chain_id: str) -> Dict:
        return self.get_by_index(self.chain_id_to_idx[chain_id])

    def print_statistics(self):
        print(f"\nTotal proteins: {self.stats['n_proteins']}")

        def print_dist(title, counts, names):
            print(f"\n{title}")
            total = sum(counts.values())
            for key, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                name = names.get(key)
                pct = 100 * count / total
                print(f"{key} ({name}): {count:,} ({pct:.2f}%)")

        print_dist(
            "Amino Acid Distribution", self.stats["amino_acid_counts"], self.RES_DICT
        )

        print_dist(
            "DSSP8 Distribution",
            self.stats["dssp8_counts"],
            {
                "H": "Alpha-helix",
                "G": "3-10 helix",
                "I": "Pi-helix",
                "E": "Beta-sheet",
                "B": "Beta-bridge",
                "T": "Turn",
                "S": "Bend",
                "C": "Coil",
            },
        )
        print_dist(
            "DSSP3 Distribution",
            self.stats["dssp3_counts"],
            {"H": "Helix", "E": "Sheet", "C": "Coil"},
        )

        print("\nSequence Length Statistics")
        stats = self.stats["input_length_stats"]
        print(
            f"Min: {stats['min']}, Max: {stats['max']}, Median: {stats['median']:.2f}"
        )

    def save(self, output_path: str = "protein_dataset.pt"):
        data = {
            "chain_ids": self.chain_ids,
            "first_res": self.first_res,
            "input_seqs": self.input_seqs,
            "dssp8_seqs": self.dssp8_seqs,
            "dssp3_seqs": self.dssp3_seqs,
            "input_encoded": self.input_encoded,
            "dssp8_encoded": self.dssp8_encoded,
            "dssp3_encoded": self.dssp3_encoded,
            "seq_lengths": self.seq_lengths,
            "stats": self.stats,
            "chain_id_to_idx": self.chain_id_to_idx,
            "idx_to_chain_id": self.idx_to_chain_id,
        }
        torch.save(data, output_path)
        print(f"Saved dataset to {output_path}")

    @staticmethod
    def load(pt_path: str = "protein_dataset.pt"):
        data = torch.load(pt_path)
        processor = ProteinDatasetProcessor.__new__(ProteinDatasetProcessor)
        for key, value in data.items():
            setattr(processor, key, value)
        print(f"\nLoaded dataset from {pt_path}")
        print(f"Loaded {processor.stats['n_proteins']} proteins successfully!")
        return processor


def main():
    start = time.perf_counter()
    # Cleaning the csv
    csv_path = "ps4_dataset.csv"
    processor = ProteinDatasetProcessor(csv_path)
    processor.process()
    processor.save("ps4_dataset.pt")
    end = time.perf_counter()

    print(
        f"\nData processed for {len(processor.input_seqs)} sequences in {end - start:.6f} seconds"
    )

    # Stats
    processor.print_statistics()

    sample_index = 0
    print(f"\nSample protein at index {sample_index}:")
    sample = processor.get_by_index(sample_index)
    for key, value in sample.items():
        if key in ["input", "dssp8", "dssp3"]:
            print(f"{key}: {value[:50]}..." if len(value) > 50 else f"{key}: {value}")
        elif key.endswith("_encoded"):
            print(f"{key}: tensor of shape {value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {value}")

    sample_chain_id = "4trtA"
    print(f"\nSample protein by chain_id '{sample_chain_id}':")
    sample = processor.get_by_chain_id(sample_chain_id)
    for key, value in sample.items():
        if key in ["input", "dssp8", "dssp3"]:
            print(f"{key}: {value[:50]}..." if len(value) > 50 else f"{key}: {value}")
        elif key.endswith("_encoded"):
            print(f"{key}: tensor of shape {value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {value}")

    # Loading cleaned data in PyTorch object format
    start2 = time.perf_counter()
    loaded_processor = ProteinDatasetProcessor.load("ps4_dataset.pt")
    end2 = time.perf_counter()
    print(
        f"Data loaded for {len(loaded_processor.input_seqs)} sequences in {end2 - start2:.6f} seconds"
    )


if __name__ == "__main__":
    main()
