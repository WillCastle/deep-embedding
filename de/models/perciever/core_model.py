"""
TODO.
"""
import os
import sys

# sys.path.insert(0, '.de/models/prepare')
# print(sys.path)
import numpy as np
import torch
import torch.nn as nn
from text_preprocess import train_tokeniser
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer, PerceiverConfig, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverClassificationDecoder, PerceiverImagePreprocessor,
    PerceiverTextPreprocessor)


class TextPreprocessor():
    """
    TODO.
    """
    def __init__(
            self,
            tokeniser: Tokenizer,
            ):
        """
        TODO.
        """
        self.tokeniser = tokeniser

    def __call__(self, text_batch: list[str]) -> torch.Tensor:
        """
        TODO.
        """
        sorted_batch = sorted(text_batch, key=len, reverse=True)
        tokenised_batch = self.tokeniser.encode_batch(sorted_batch)
        tokenised_batch = torch.tensor(
            [sequence.ids for sequence in tokenised_batch]
            ).int()
        return tokenised_batch#.T


class IMDBDataset(Dataset):
    """
    TODO.
    """
    def __init__(self, data: IterableDataset, preprocessor: TextPreprocessor) -> None:
        """
        TODO.
        """
        super().__init__()
        self.data = np.array(list(data))
        self.preprocessor = preprocessor

    def __len__(self):
        """
        TODO.
        """
        return len(self.data)

    # TODO: get this working with Dataloader, dataloader currently seems to index one at a time
    def __getitem__(self, index: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        index = [index] if not isinstance(index, list) else index
        batch = self.data[[index]]
        labels = [int(label) for label in batch[:, 0]]
        texts = batch[:, 1]
        tokenised_batch = self.preprocessor(list(texts))
        return tokenised_batch, torch.tensor(labels)



if __name__ == "__main__":
    from pathlib import Path

    from torchtext.datasets import IMDB

    base_data_path = "C:/Users/willf/DataScience/Repos/deep-embedding/de/models/prepare/text/imdb_data"
    # make splits for data
    train_dataset, test_dataset = IMDB(root=(base_data_path), split=(str("train"), str("test")))
    # for sample_idx, train_sample in enumerate(train_dataset):
    #     with open(base_data_path + f"/train/{sample_idx}.txt", 'w', encoding="utf-8") as f:
    #         f.write(f"{str(train_sample[0])}, {train_sample[1]}")

    # for sample_idx, test_sample in enumerate(test_dataset):
    #     with open(base_data_path + f"/test/{sample_idx}.txt", 'w', encoding="utf-8") as f:
    #         f.write(f"{str(test_sample[0])}, {test_sample[1]}")

    # print("Finished writing text files.")
    # test_dataset, = IMDB(tokenizer=tokenizer, vocab=vocab, data_select='test')

    data_paths = [str(path) for path in Path(base_data_path).glob("train/*.txt")]
    tokeniser = train_tokeniser(data_paths=data_paths, save_path=base_data_path + "/imdb_bpe_tokenizer.json")
    vocab_size = tokeniser.get_vocab_size()
    # tokenise

    # Generate a list of tuples of text length, index, label, text
    data_len = [(len(txt), idx, label, txt) for idx, (label, txt) in enumerate(train_dataset)]
    data_len.sort()  # sort by length and pad sequences with similar lengths
    text_preprocessor = TextPreprocessor(tokeniser=tokeniser)
    # Generate the pad id
    pad_id = tokeniser.token_to_id("[PAD]")

    # Generate 8x8 batches
    # Pad sequences with similar lengths

    # def pad_data(data):
    #     # Find max length of the mini-batch
    #     max_len = max(list(zip(*data))[0])
    #     label_list = list(zip(*data))[2]
    #     txt_list = list(zip(*data))[3]
    #     padded_tensors = torch.stack(
    #         [
    #             torch.cat((txt, torch.tensor([pad_id] * (max_len - len(txt))).long()))
    #             for txt in txt_list
    #         ]
    #     )
    #     return padded_tensors, label_list

    train_dataset = IMDBDataset(data=train_dataset, preprocessor=text_preprocessor)

    dataloader = DataLoader(train_dataset, batch_size=8)
    for idx, (txt, label) in enumerate(dataloader):
        print(idx, txt.size(), label)

    PERCEIVER_CONFIG = PerceiverConfig(vocab_size=vocab_size, image_size=224)

    perceiver_text_preprocessor = PerceiverTextPreprocessor(config=PERCEIVER_CONFIG)


    image_preprocess = PerceiverImagePreprocessor(
        config=PERCEIVER_CONFIG,
        prep_type="conv1x1",
        spatial_downsample=1,
        out_channels=256,
        position_encoding_type="trainable",
        concat_or_add_pos="concat",
        project_pos_dim=256,
        trainable_position_encoding_kwargs=dict(
            num_channels=256,
            index_dims=PERCEIVER_CONFIG.image_size**2,
        ),
    )
    decoder = PerceiverClassificationDecoder(
        PERCEIVER_CONFIG,
        num_channels=PERCEIVER_CONFIG.d_latents,
        trainable_position_encoding_kwargs=dict(num_channels=PERCEIVER_CONFIG.d_latents, index_dims=1),
        use_query_residual=True,
    )
    perceiver = PerceiverModel(PERCEIVER_CONFIG, input_preprocessor=text_preprocess, decoder=decoder)

    # you can then do a forward pass as follows:
    text_batch = [
        "There are some unseen words here banana zen 😁 but also some seen text",
        "Lots of unknowns in this one I expect",
        "text and words and tokens and stuff"
    ]
    inputs = torch.tensor([encoding.ids for encoding in tokenizer.encode_batch(text_batch)])

    with torch.no_grad():
        outputs = perceiver(inputs=inputs)
    logits = outputs.logits
    print(list(logits.shape))

    # to train, one can train the model using standard cross-entropy:
    criterion = torch.nn.CrossEntropyLoss()

    labels = torch.tensor([1])
    loss = criterion(logits, labels)
    print(loss)
