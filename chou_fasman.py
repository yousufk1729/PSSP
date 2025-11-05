from protein_dataset_processor import ProteinDatasetProcessor
import numpy as np
import time


class ChouFasmanPredictor:
    CF_TABLE = {
        "A": {"Pa": 141, "Pb": 75, "Pt": 66, "f": [0.06, 0.076, 0.035, 0.058]},
        "R": {"Pa": 121, "Pb": 85, "Pt": 95, "f": [0.070, 0.106, 0.099, 0.085]},
        "D": {"Pa": 82, "Pb": 55, "Pt": 146, "f": [0.147, 0.110, 0.179, 0.081]},
        "N": {"Pa": 73, "Pb": 63, "Pt": 156, "f": [0.161, 0.083, 0.191, 0.091]},
        "C": {"Pa": 85, "Pb": 136, "Pt": 119, "f": [0.149, 0.050, 0.117, 0.128]},
        "E": {"Pa": 139, "Pb": 65, "Pt": 74, "f": [0.056, 0.060, 0.077, 0.064]},
        "Q": {"Pa": 126, "Pb": 72, "Pt": 98, "f": [0.074, 0.098, 0.037, 0.098]},
        "G": {"Pa": 44, "Pb": 67, "Pt": 156, "f": [0.102, 0.085, 0.190, 0.152]},
        "H": {"Pa": 87, "Pb": 99, "Pt": 95, "f": [0.140, 0.047, 0.093, 0.054]},
        "I": {"Pa": 104, "Pb": 179, "Pt": 47, "f": [0.043, 0.034, 0.013, 0.056]},
        "L": {"Pa": 128, "Pb": 115, "Pt": 59, "f": [0.061, 0.025, 0.036, 0.070]},
        "K": {"Pa": 117, "Pb": 76, "Pt": 101, "f": [0.055, 0.115, 0.072, 0.095]},
        "M": {"Pa": 126, "Pb": 101, "Pt": 60, "f": [0.068, 0.082, 0.014, 0.055]},
        "F": {"Pa": 100, "Pb": 140, "Pt": 60, "f": [0.059, 0.041, 0.065, 0.065]},
        "P": {"Pa": 44, "Pb": 40, "Pt": 152, "f": [0.102, 0.301, 0.034, 0.068]},
        "S": {"Pa": 76, "Pb": 81, "Pt": 143, "f": [0.120, 0.139, 0.125, 0.106]},
        "T": {"Pa": 78, "Pb": 121, "Pt": 96, "f": [0.086, 0.108, 0.065, 0.079]},
        "W": {"Pa": 107, "Pb": 123, "Pt": 96, "f": [0.077, 0.013, 0.064, 0.167]},
        "Y": {"Pa": 98, "Pb": 137, "Pt": 114, "f": [0.082, 0.065, 0.114, 0.125]},
        "V": {"Pa": 91, "Pb": 200, "Pt": 50, "f": [0.062, 0.048, 0.028, 0.053]},
        "X": {"Pa": 100, "Pb": 100, "Pt": 100, "f": [0.1, 0.1, 0.1, 0.1]},
    }

    DSSP3_MAP = {"H": "H", "E": "E", "C": "C", "T": "C"}

    def __init__(self, processor: ProteinDatasetProcessor):
        self.processor = processor
        self.helix_window = 6
        self.sheet_window = 5
        self.turn_window = 4
        self.helix_threshold = 103
        self.sheet_threshold = 105

        self.predictions_dssp3 = []
        self.predictions_dssp8 = []
        self.stats = {}

    def predict_sequence(self, sequence):
        seq_len = len(sequence)
        scores_helix = np.zeros(seq_len)
        scores_sheet = np.zeros(seq_len)
        scores_turn = np.zeros(seq_len)

        for i in range(seq_len):
            start_h = max(0, i - self.helix_window // 2)
            end_h = min(seq_len, start_h + self.helix_window)
            if end_h - start_h == self.helix_window:
                window = sequence[start_h:end_h]
                scores_helix[i] = np.mean([self.CF_TABLE[aa]["Pa"] for aa in window])

            start_s = max(0, i - self.sheet_window // 2)
            end_s = min(seq_len, start_s + self.sheet_window)
            if end_s - start_s == self.sheet_window:
                window = sequence[start_s:end_s]
                scores_sheet[i] = np.mean([self.CF_TABLE[aa]["Pb"] for aa in window])

            if i + self.turn_window <= seq_len:
                window = sequence[i : i + self.turn_window]
                turn_prob = 1.0
                for j, aa in enumerate(window):
                    turn_prob *= self.CF_TABLE[aa]["f"][j]
                scores_turn[i : i + self.turn_window] += turn_prob

        prediction_q3 = []
        prediction_q8 = []

        for i in range(seq_len):
            if scores_turn[i] > 0.0001 and scores_turn[i] > max(
                scores_helix[i] / 150, scores_sheet[i] / 150
            ):
                prediction_q3.append("C")
                prediction_q8.append("T")
            elif (
                scores_helix[i] > self.helix_threshold
                and scores_helix[i] > scores_sheet[i]
            ):
                prediction_q3.append("H")
                prediction_q8.append("H")
            elif scores_sheet[i] > self.sheet_threshold:
                prediction_q3.append("E")
                prediction_q8.append("E")
            else:
                prediction_q3.append("C")
                prediction_q8.append("C")

        return "".join(prediction_q3), "".join(prediction_q8)

    def evaluate(self):
        print("\nEvaluating Chou-Fasman predictions")

        correct_q3 = 0
        correct_q8 = 0
        total_residues = 0

        self.predictions_dssp3 = []
        self.predictions_dssp8 = []

        for i in range(len(self.processor.input_seqs)):
            sequence = self.processor.input_seqs[i]
            true_dssp3 = self.processor.dssp3_seqs[i]
            true_dssp8 = self.processor.dssp8_seqs[i]

            pred_q3, pred_q8 = self.predict_sequence(sequence)

            self.predictions_dssp3.append(pred_q3)
            self.predictions_dssp8.append(pred_q8)

            for j in range(len(sequence)):
                total_residues += 1
                if pred_q3[j] == true_dssp3[j]:
                    correct_q3 += 1

                pred_q8_mapped = pred_q8[j]
                if pred_q8_mapped == true_dssp8[j]:
                    correct_q8 += 1

        q3_accuracy = 100 * correct_q3 / total_residues
        q8_accuracy = 100 * correct_q8 / total_residues

        self.stats = {
            "q3_accuracy": q3_accuracy,
            "q8_accuracy": q8_accuracy,
            "total_proteins": len(self.processor.input_seqs),
            "total_residues": total_residues,
            "correct_q3": correct_q3,
            "correct_q8": correct_q8,
        }

        print("\nResults")
        print(f"Total proteins evaluated: {self.stats['total_proteins']}")
        print(f"Total residues: {self.stats['total_residues']:,}")
        print(f"Q3 Accuracy: {q3_accuracy:.2f}%")
        print(f"Q8 Accuracy: {q8_accuracy:.2f}%")

        return self.stats

    def get_prediction_by_index(self, idx):
        return {
            "index": idx,
            "chain_id": self.processor.chain_ids[idx],
            "sequence": self.processor.input_seqs[idx],
            "true_dssp3": self.processor.dssp3_seqs[idx],
            "true_dssp8": self.processor.dssp8_seqs[idx],
            "predicted_dssp3": self.predictions_dssp3[idx],
            "predicted_dssp8": self.predictions_dssp8[idx],
            "length": len(self.processor.input_seqs[idx]),
        }

    def get_prediction_by_chain_id(self, chain_id):
        idx = self.processor.chain_id_to_idx[chain_id]
        return self.get_prediction_by_index(idx)

    def print_sample_prediction(self, idx):
        sample = self.get_prediction_by_index(idx)

        print(f"\nSample protein at index {idx} (chain ID: {sample['chain_id']})")
        print(f"Sequence length: {sample['length']}")

        seq = sample["sequence"]
        true_q3 = sample["true_dssp3"]
        pred_q3 = sample["predicted_dssp3"]
        true_q8 = sample["true_dssp8"]
        pred_q8 = sample["predicted_dssp8"]

        matches_q3 = sum(1 for i in range(len(seq)) if true_q3[i] == pred_q3[i])
        matches_q8 = sum(1 for i in range(len(seq)) if true_q8[i] == pred_q8[i])

        print(f"Q3 Accuracy for this protein: {100 * matches_q3 / len(seq):.2f}%")
        print(f"Q8 Accuracy for this protein: {100 * matches_q8 / len(seq):.2f}%")

        display_len = min(60, len(seq))
        print(f"First {display_len} residues")
        print(f"Sequence: {seq[:display_len]}")
        print(f"True Q3:  {true_q3[:display_len]}")
        print(f"Pred Q3:  {pred_q3[:display_len]}")
        print(f"True Q8:  {true_q8[:display_len]}")
        print(f"Pred Q8:  {pred_q8[:display_len]}")


def main():
    start = time.perf_counter()
    pt_path = "ps4_dataset.pt"
    processor = ProteinDatasetProcessor.load(pt_path)
    predictor = ChouFasmanPredictor(processor)
    predictor.evaluate()
    end = time.perf_counter()

    predictor.print_sample_prediction(0)

    print(
        f"Loading and predictions made for {len(predictor.predictions_dssp3)} sequences in {end - start:.6f} seconds,\n"
        f"with {predictor.stats['q3_accuracy']:.2f}% Q3 accuracy and {predictor.stats['q8_accuracy']:.2f}% Q8 accuracy"
    )


if __name__ == "__main__":
    main()
