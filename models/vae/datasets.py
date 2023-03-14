"""
TODO.
"""

from os.path import exists
from pathlib import Path
from random import sample

import numpy as np
import torch
from prepare.text.text_preprocess import TextPreprocessor
from prepare.text.tokenizer import train_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.datasets import IMDB
from transformers import PreTrainedTokenizerFast


class ApiCallsDataset(Dataset):
    def __init__(self, paths: list[str]):
        base_data_path = Path("C:/Analysis/reports")

        tokenizer_file="C:/Analysis/code/classifier/lucidrains_perceiver_io/saved_tokenizers/bpe_tokenizer.json"

        self.max_seq_len = 100000
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            ) if exists(tokenizer_file) else train_tokenizer(
                paths,
                save_path=tokenizer_file
                )

        self.label_codes = {
            str(path).split("\\")[-1]: num_code
            for num_code, path in enumerate(base_data_path.glob("*"))
            }

        self.labelled_data = [
            (
                Path(path).read_text(encoding="utf-8").replace('"', '').replace("\n", " "),
                path.split("\\")[3]
                ) for path in paths
            ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        api_sequence, threat_group = self.labelled_data[index]

        label_code = self.label_codes[threat_group]
        api_tokens = self.tokenizer(text=api_sequence, padding=False)
        tokens = api_tokens.encodings[0].ids
        truncated = tokens[:self.max_seq_len] if len(tokens) > self.max_seq_len else tokens
        return torch.tensor(truncated), torch.tensor([label_code])

    def __len__(self):
        return len(list(self.labelled_data))

    def get_raw_examples(self, number: int = 1) -> str:
        return sample(self.labelled_data, k=number)

    def pad_batch(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor]]
        ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences, labels = zip(*batch)
        padded_sequences = pad_sequence(sequences=sequences, batch_first=True)
        # print(f"\nBatch dims: {padded_sequences.size()}, {torch.tensor(labels).size()}\n")
        return padded_sequences, torch.tensor(labels)

    def padding_mask(self, padded_batch: torch.Tensor) -> torch.Tensor:
        return (padded_batch != 0)


if __name__ == "__main__":
    base_data_path = Path("C:/Analysis/reports")
    api_data_paths = [str(path) for path in base_data_path.glob("**/cuckoo_extracted_api/*.txt")]
    ds = ApiCallsDataset(api_data_paths)

    seq1 = [1, 2, 3, 4]
    seq2 = [1, 2, 3, 4, 5, 6, 7]
    seq3 = [1, 2]
    seq4 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    seq5 = [1, 2, 3, 4, 5]

    test_batch = [
        (torch.Tensor(seq1), torch.Tensor([4])),
        (torch.Tensor(seq2), torch.Tensor([4])),
        (torch.Tensor(seq3), torch.Tensor([4])),
        (torch.Tensor(seq4), torch.Tensor([4])),
        (torch.Tensor(seq5), torch.Tensor([4])),
        ]

    padded_seqs, labels = ds.pad_batch(test_batch)

    print(f"Padded batch: \n{padded_seqs}\n\nPadding mask: \n{ds.padding_mask(padded_seqs)}")


    # pad_starts = (padded_seqs == 0).int().argmax(axis=1)