"""
TODO.
"""
import tokenizers
import torch


class TextPreprocessor():
    """
    TODO.
    """
    def __init__(
            self,
            tokenizer: tokenizers.Tokenizer,
            ):
        self.tokenizer = tokenizer

    def __call__(self, text_batch: list[str]):
        sorted_batch = sorted(text_batch, key=len, reverse=True)
        tokenized_batch = self.tokenizer.encode_batch(sorted_batch)
        tokenized_batch = torch.Tensor(
            [sequence.ids for sequence in tokenized_batch]
            ).int()
        return tokenized_batch.T
