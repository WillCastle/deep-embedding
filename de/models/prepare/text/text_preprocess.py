"""
TODO.
"""
import torch
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)


def train_tokenizer(
        data_paths: list[str],
        save_path: str = "bpe_tokenizer.json"
        ) -> Tokenizer:
    """TODO.
        Args:

        Returns:

    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer_trainer = trainers.BpeTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )  # TODO: some of these not used?
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.train(files=data_paths, trainer=tokenizer_trainer)
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"),
        pad_token="[PAD]"
        )
    tokenizer.save(save_path)

    return Tokenizer.from_file(path=save_path)


class TextPreprocessor():
    """
    TODO.
    """
    def __init__(
            self,
            tokenizer: Tokenizer,
            ):
        self.tokenizer = tokenizer

    def __call__(self, text_batch: list[str]):
        sorted_batch = sorted(text_batch, key=len, reverse=True)
        tokenized_batch = self.tokenizer.encode_batch(sorted_batch)
        tokenized_batch = torch.Tensor(
            [sequence.ids for sequence in tokenized_batch]
            ).int()
        return tokenized_batch.T



# def training_mask(sequence: str, mask_probability: float | int):


#     return sequence

# encoding.input_ids[0, 22:31] = tk.token_to_id("[MASK]")

if __name__ == "__main__":
    import os
    from pathlib import Path

    from transformers.tokenization_utils_base import BatchEncoding

    text_batch = [
        "There are some unseen words here banana zen 😁 but also some seen text",
        "Lots of unknowns in this one I expect",
        "text and words and tokens and stuff"
    ]

    test_tokenizer_path = "C:/Users/willf/DataScience/Repos/deep-embedding/de/vae/prepare/text/test_data/test_bpe_tokenizer.json"
    base_data_path = Path("C:/Users/willf/DataScience/Repos/deep-embedding/de/vae/prepare/text/test_data")
    data_paths = [str(path) for path in base_data_path.glob("**/*.txt")]

    tk = train_tokenizer(data_paths=data_paths, save_path=test_tokenizer_path)

    encoded_batch: BatchEncoding = tk.encode_batch(text_batch)

    detokenized = tk.decode_batch(sequences=[encoding.ids for encoding in encoded_batch], skip_special_tokens=False)

    text_preprocessor = TextPreprocessor(tokenizer=tk)

    print(text_preprocessor(text_batch))

    print(detokenized, "\n", text_batch)

    os.remove(test_tokenizer_path)

