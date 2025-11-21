import pandas as pd
import torch
from collections import Counter
from typing import Dict
import time


class PS4DatasetProcessor:
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

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.chain_ids = None
        self.first_res = None
        self.input_seqs = None
        self.dssp8_seqs = None
        self.dssp3_seqs = None
        self.seq_lengths = None
        self.stats = None

    def process(self):
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} proteins")

        self.chain_ids = df["chain_id"].tolist()
        self.first_res = (df["first_res"]).tolist()
        self.input_seqs = df["input"].tolist()
        self.dssp8_seqs = df["dssp8"].tolist()

        for i in range(len(self.input_seqs)):
            self.input_seqs[i] = self.input_seqs[i].upper()
            invalid_aa = set(self.input_seqs[i]) - set(self.AA_CHARS)
            if invalid_aa:
                # print(
                #     f"Warning: Protein {i} ({self.chain_ids[i]}) invalid AA: {invalid_aa}"
                # )
                for char in invalid_aa:
                    self.input_seqs[i] = self.input_seqs[i].replace(char, "X")

            invalid_ss = set(self.dssp8_seqs[i]) - set(self.DSSP8_CHARS)
            if invalid_ss:
                # print(
                #     f"Warning: Protein {i} ({self.chain_ids[i]}) invalid SS: {invalid_ss}"
                # )
                for char in invalid_ss:
                    self.dssp8_seqs[i] = self.dssp8_seqs[i].replace(char, "C")

        self.dssp3_seqs = [
            "".join(self.DSSP8_TO_DSSP3.get(c) for c in seq) for seq in self.dssp8_seqs
        ]

        self.seq_lengths = torch.tensor([len(seq) for seq in self.input_seqs])

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

        print("Processing complete")

    def print_statistics(self):
        print("\nStatistics")
        print(f"Total proteins: {self.stats['n_proteins']}")

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

    def print_sample(self, index: int):
        print(f"\nSample protein at index {index}")
        sample = self.get_by_index(index)
        for key, value in sample.items():
            if key in ["input", "dssp8", "dssp3"]:
                print(
                    f"{key}: {value[:50]}..." if len(value) > 50 else f"{key}: {value}"
                )

    def get_by_index(self, index: int) -> Dict:
        return {
            "chain_id": self.chain_ids[index],
            "first_res": self.first_res[index],
            "input": self.input_seqs[index],
            "dssp8": self.dssp8_seqs[index],
            "dssp3": self.dssp3_seqs[index],
            "length": self.seq_lengths[index].item(),
        }

    def save(self, output_path: str):
        data = {
            "chain_ids": self.chain_ids,
            "first_res": self.first_res,
            "input_seqs": self.input_seqs,
            "dssp8_seqs": self.dssp8_seqs,
            "dssp3_seqs": self.dssp3_seqs,
            "seq_lengths": self.seq_lengths,
            "stats": self.stats,
        }
        torch.save(data, output_path)
        print(f"Saved dataset to {output_path}")

    def load(pt_path: str):
        data = torch.load(pt_path, weights_only=False)
        processor = PS4DatasetProcessor.__new__(PS4DatasetProcessor)
        for key, value in data.items():
            setattr(processor, key, value)
        print(f"\nLoaded dataset from {pt_path}")
        print(f"Loaded {processor.stats['n_proteins']} proteins successfully!")
        return processor


def main():
    csv_path = "ps4_data/ps4_dataset.csv"
    save_path = "ps4_data/ps4_dataset.pt"
    sample_index = 0

    # Cleaning the CSV into a PyTorch object
    start = time.perf_counter()
    processor = PS4DatasetProcessor(csv_path)
    processor.process()
    processor.save(save_path)
    end = time.perf_counter()
    print(
        f"Data processed for {len(processor.input_seqs)} sequences in {end - start:.6f} seconds"
    )

    # General stats
    processor.print_statistics()

    # Specific sample
    processor.print_sample(sample_index)

    # Loading cleaned data in PyTorch object format
    start2 = time.perf_counter()
    loaded_processor = PS4DatasetProcessor.load(save_path)
    end2 = time.perf_counter()
    print(
        f"Data loaded for {len(loaded_processor.input_seqs)} sequences in {end2 - start2:.6f} seconds"
    )


if __name__ == "__main__":
    main()
