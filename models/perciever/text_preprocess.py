"""
TODO.
"""
import torch
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)


# TODO: write 'load_tokeniser'
def load_tokeniser(tokeniser_path: str) -> Tokenizer:
    """
    TODO.
        Args:
            tokeniser_path: str -

        Returns:

    """
    return Tokenizer.from_file(path=tokeniser_path)


def train_tokeniser(
        data_paths: list[str],
        save_path: str = "bpe_tokeniser.json"
    ) -> Tokenizer:
    """
    TODO.
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
            tokeniser: Tokenizer,
            ):
        self.tokeniser = tokeniser

    def __call__(self, text_batch: list[str]):
        sorted_batch = sorted(text_batch, key=len, reverse=True)
        tokenised_batch = self.tokeniser.encode_batch(sorted_batch)
        tokenised_batch = torch.Tensor(
            [sequence.ids for sequence in tokenised_batch]
            ).int()
        return tokenised_batch.T



# def training_mask(sequence: str, mask_probability: float | int):


#     return sequence

# encoding.input_ids[0, 22:31] = tk.token_to_id("[MASK]")

if __name__ == "__main__":
    import os
    from pathlib import Path

    from transformers.tokenization_utils_base import BatchEncoding

    # Setup
    TEXT_BATCH = [
        "There are some unseen words here banana zen 😁 but also some seen text",
        "Lots of unknowns in this one I expect",
        "text and words and tokens and stuff"
    ]
    TEST_TOKENISER_PATH = "C:/Users/willf/DataScience/Repos/deep-embedding/de/models/prepare/text/test_data/test_bpe_tokeniser.json"
    BASE_DATA_PATH = Path("C:/Users/willf/DataScience/Repos/deep-embedding/de/models/prepare/text/test_data")
    data_paths = [str(path) for path in BASE_DATA_PATH.glob("**/*.txt")]

    # Train tokeniser test
    tk = train_tokeniser(data_paths=data_paths, save_path=TEST_TOKENISER_PATH)
    encoded_batch: BatchEncoding = tk.encode_batch(TEXT_BATCH)
    detokenised = tk.decode_batch(sequences=[encoding.ids for encoding in encoded_batch], skip_special_tokens=False)
    print(detokenised, "\n", TEXT_BATCH)

    # Load tokeniser test
    tk = load_tokeniser(tokeniser_path=TEST_TOKENISER_PATH)
    encoded_batch: BatchEncoding = tk.encode_batch(TEXT_BATCH)
    detokenised = tk.decode_batch(sequences=[encoding.ids for encoding in encoded_batch], skip_special_tokens=False)
    print(detokenised, "\n", TEXT_BATCH)



    # text_preprocessor = TextPreprocessor(tokeniser=tk)
    # print(text_preprocessor(TEXT_BATCH))

    # os.remove(test_tokeniser_path)

