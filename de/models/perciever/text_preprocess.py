"""
TODO.
"""
import torch
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)


# TODO: write 'load_tokenizer'
def train_tokeniser(
        data_paths: list[str],
        save_path: str = "bpe_tokeniser.json"
        ) -> Tokenizer:
    """TODO.
        Args:

        Returns:

    """
    tokeniser = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokeniser.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokeniser_trainer = trainers.BpeTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )  # TODO: some of these not used?
    tokeniser.post_processor = processors.ByteLevel(trim_offsets=True)
    tokeniser.decoder = decoders.ByteLevel()

    tokeniser.train(files=data_paths, trainer=tokeniser_trainer)
    tokeniser.enable_padding(
        pad_id=tokeniser.token_to_id("[PAD]"),
        pad_token="[PAD]"
        )
    tokeniser.save(save_path)

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

    tk = train_tokeniser(data_paths=data_paths, save_path=test_tokenizer_path)

    encoded_batch: BatchEncoding = tk.encode_batch(text_batch)

    detokenized = tk.decode_batch(sequences=[encoding.ids for encoding in encoded_batch], skip_special_tokens=False)

    text_preprocessor = TextPreprocessor(tokenizer=tk)

    print(text_preprocessor(text_batch))

    print(detokenized, "\n", text_batch)

    os.remove(test_tokenizer_path)

