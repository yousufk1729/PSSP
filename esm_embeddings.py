import torch
from dataclasses import dataclass
from typing import List
import esm

from ps4_data.ps4_dataset_processor import PS4DatasetProcessor
from cb513_data.cb513_dataset_processor import CB513DatasetProcessor


@dataclass
class ESMEmbeddings:
    embeddings: List[torch.Tensor]
    esm_dim: int

    def save(self, path: str):
        torch.save(
            {
                "embeddings": self.embeddings,
                "esm_dim": self.esm_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ESMEmbeddings":
        data = torch.load(path, weights_only=False)
        return cls(
            embeddings=data["embeddings"],
            esm_dim=data["esm_dim"],
        )


def extract_esm_embeddings(
    sequences: List[str],
    device: torch.device,
    max_tokens_per_batch: int,
) -> ESMEmbeddings:
    print(f"Loading ESM-2 model on {device}")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    esm_dim = 320

    indexed_seqs = [(i, seq) for i, seq in enumerate(sequences)]
    indexed_seqs.sort(key=lambda x: len(x[1]))

    all_embeddings = [None] * len(sequences)
    n_seqs = len(sequences)

    print(f"Extracting embeddings for {n_seqs} sequences")

    i = 0
    processed = 0
    while i < n_seqs:
        batch_indices = []
        batch_seqs = []
        total_tokens = 0

        while i < n_seqs:
            idx, seq = indexed_seqs[i]
            seq_tokens = len(seq) + 2

            if len(batch_seqs) > 0 and total_tokens + seq_tokens > max_tokens_per_batch:
                break

            batch_indices.append(idx)
            batch_seqs.append(seq)
            total_tokens += seq_tokens
            i += 1

        batch_data = [
            (f"seq_{idx}", seq) for idx, seq in zip(batch_indices, batch_seqs)
        ]

        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)

        token_embeddings = results["representations"][6]

        for j, (idx, seq) in enumerate(zip(batch_indices, batch_seqs)):
            seq_len = len(seq)
            emb = token_embeddings[j, 1 : seq_len + 1, :].cpu()
            all_embeddings[idx] = emb

        del batch_tokens, results, token_embeddings
        torch.cuda.empty_cache()

        processed += len(batch_seqs)
        if processed % 1000 < len(batch_seqs) or processed == n_seqs:
            print(f"Processed {processed}/{n_seqs} sequences")

    print("Embedding extraction complete")
    return ESMEmbeddings(embeddings=all_embeddings, esm_dim=esm_dim)


def main():
    device = torch.device("cuda")
    torch.manual_seed(1729)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision("high")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    ps4_load_path = "ps4_data/ps4_dataset.pt"
    cb513_load_path = "cb513_data/cb513_dataset.pt"
    ps4_save_path = "ps4_data/ps4_esm_embeddings.pt"
    cb513_save_path = "cb513_data/cb513_esm_embeddings.pt"

    print("\nLoading PS4 dataset")
    ps4 = PS4DatasetProcessor.load(ps4_load_path)

    print("\nExtracting PS4 embeddings")
    ps4_emb = extract_esm_embeddings(
        ps4.input_seqs, device=device, max_tokens_per_batch=1024
    )
    ps4_emb.save(ps4_save_path)
    print(f"Saved to {ps4_save_path}")

    print("\nLoading CB513 dataset")
    cb513 = CB513DatasetProcessor.load(cb513_load_path)

    print("\nExtracting CB513 embeddings")
    cb513_emb = extract_esm_embeddings(
        cb513.input_seqs, device=device, max_tokens_per_batch=1024
    )
    cb513_emb.save(cb513_save_path)
    print(f"Saved to {cb513_save_path}")


if __name__ == "__main__":
    main()
