"""TODO."""
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)


def train_tokenizer(data_paths: list[str], save_path: str="bpe_tokenizer.json") -> Tokenizer:
    """TODO.
        Args:

        Returns:

    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer_trainer = trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])  # TODO some of these not used?
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.train(files=data_paths, trainer=tokenizer_trainer)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    tokenizer.save(save_path)

    return Tokenizer.from_file(path=save_path)
