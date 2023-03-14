"""
TODO.
"""
import numpy as np
import torch
import torch.nn as nn
from text_preprocess import load_tokeniser, train_tokeniser
from tokenizers import Tokenizer
from torch.nn.utils.rnn import (PackedSequence, pack_padded_sequence,
                                pad_packed_sequence, pad_sequence)
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import PerceiverConfig, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverClassificationDecoder, PerceiverDecoderOutput,
    PerceiverImagePreprocessor, PerceiverTextPreprocessor)


class TextPreprocessor():
    """
    TODO.
    """
    def __init__(
            self,
            tokeniser: Tokenizer,
            device: str = "cpu",
            ):
        """
        TODO.
        """
        self.tokeniser = tokeniser
        self.device = device

    def __call__(self, text_batch: list[str]) -> torch.Tensor:
        """
        TODO.
        """
        tokenised_batch = self.tokeniser.encode_batch(text_batch)
        tokenised_batch = torch.tensor(
            [encoding.ids for encoding in tokenised_batch]
            ).int()
        return tokenised_batch.T.to(self.device)


class IMDBDataset(Dataset):
    """
    TODO.
    """
    def __init__(self, data: IterableDataset, preprocessor: TextPreprocessor, device: str = "cpu") -> None:
        """
        TODO.
        """
        super().__init__()
        self.data = np.array(list(data))
        self.labels = self.data[:, 0].astype(int)
        self.texts = self.data[:, 1]
        self.preprocessor = preprocessor
        self.tokeniser = self.preprocessor.tokeniser
        self.pad_value = self.tokeniser.token_to_id("[PAD]")
        self.device = device

    def __len__(self):
        """
        TODO.
        """
        return len(self.data)

    # TODO: get this working with Dataloader, dataloader currently seems to index one at a time
    def __getitem__(self, index: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO.
        """
        # index = [index] if not isinstance(index, list) else index
        # batch = self.data[[index]]
        labels = self.labels[[index]]
        texts = self.texts[[index]]
        tokenised_batch = self.preprocessor(list(texts))
        return tokenised_batch, torch.tensor(labels)

    def pad_batch(
            self,
            batch: list[tuple[torch.Tensor, torch.Tensor]]
            ) -> tuple[PackedSequence, torch.Tensor, torch.Tensor]:
        """
        TODO.
        """
        # Called by DataLoader().__next__
        texts = [sample[0] for sample in batch]
        labels = torch.tensor([sample[1] for sample in batch])

        # get sequence lengths
        lengths = torch.tensor([text.shape[0] for text in texts])

        # pad
        padded = pad_sequence(sequences=texts, batch_first=False, padding_value=self.pad_value)
        packed = pack_padded_sequence(padded, lengths=lengths, batch_first=False, enforce_sorted=False)

        # Goes to
        return packed, lengths, labels


class PerceiverClassificationPackedDecoder(PerceiverClassificationDecoder):
    """
    TODO.
    """
    def __init__(self, config: PerceiverConfig):
        """
        TODO.
        """
        super().__init__(config=config)

    @property
    def num_query_channels(self) -> int:
        """
        TODO.
        """
        return self.decoder.num_query_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        """
        TODO.
        """
        return self.decoder.decoder_query(
            inputs, modality_sizes, inputs_without_pos, subsampled_points=subsampled_points
        )

    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
    ) -> PerceiverDecoderOutput:
        """
        TODO.
        """
        unpack before calling super.forward
        # decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)

        # # B x 1 x num_classes -> B x num_classes
        # logits = decoder_outputs.logits[:, 0, :]

        # return PerceiverDecoderOutput(logits=logits, cross_attentions=decoder_outputs.cross_attentions)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import os
    from pathlib import Path

    import objsize
    from torchtext.datasets import IMDB
    DEVICE = "cpu"

    BASE_DATA_PATH = "C:/Users/willf/DataScience/Repos/deep-embedding/models/prepare/text/imdb_data"
    # make splits for data
    train_set, test_set = IMDB(root=BASE_DATA_PATH, split=(str("train"), str("test")))  # type: ignore
    # for sample_idx, train_sample in enumerate(train_dataset):
    #     with open(base_data_path + f"/train/{sample_idx}.txt", 'w', encoding="utf-8") as f:
    #         f.write(f"{str(train_sample[0])}, {train_sample[1]}")

    # for sample_idx, test_sample in enumerate(test_dataset):
    #     with open(base_data_path + f"/test/{sample_idx}.txt", 'w', encoding="utf-8") as f:
    #         f.write(f"{str(test_sample[0])}, {test_sample[1]}")

    # print("Finished writing text files.")
    # test_dataset, = IMDB(tokenizer=tokenizer, vocab=vocab, data_select='test')

    train_paths = [str(path) for path in Path(BASE_DATA_PATH).glob("train/*.txt")]
    TOKENISER_PATH = BASE_DATA_PATH + "/imdb_bpe_tokenizer.json"
    TOKENISER = load_tokeniser(tokeniser_path=TOKENISER_PATH) if os.path.exists(TOKENISER_PATH) else train_tokeniser(
        data_paths=train_paths, save_path=TOKENISER_PATH
    )
    vocab_size = TOKENISER.get_vocab_size()
    # tokenise

    # Generate a list of tuples of text length, index, label, text
    data_len = [(len(txt), idx, label, txt) for idx, (label, txt) in enumerate(train_set)]
    data_len.sort()  # sort by length and pad sequences with similar lengths
    text_preprocessor = TextPreprocessor(tokeniser=TOKENISER, device=DEVICE)
    # Generate the pad id
    pad_id = TOKENISER.token_to_id("[PAD]")

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

    train_dataset = IMDBDataset(data=train_set, preprocessor=text_preprocessor, device=DEVICE)

    dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=train_dataset.pad_batch)
    PERCEIVER_CONFIG = PerceiverConfig(
        vocab_size=vocab_size,
        num_labels=2,
        num_latents=32,
        d_latents=160,
        d_model=96,
        num_blocks=1,
        num_self_attends_per_block=3,
        num_self_attention_heads=2,
        num_cross_attention_heads=2
    )  # , image_size=224
    perceiver_text_preprocessor = PerceiverTextPreprocessor(config=PERCEIVER_CONFIG)
    # image_preprocess = PerceiverImagePreprocessor(
    #     config=PERCEIVER_CONFIG,
    #     prep_type="conv1x1",
    #     spatial_downsample=1,
    #     out_channels=256,
    #     position_encoding_type="trainable",
    #     concat_or_add_pos="concat",
    #     project_pos_dim=256,
    #     trainable_position_encoding_kwargs=dict(
    #         num_channels=256,
    #         index_dims=PERCEIVER_CONFIG.image_size**2,
    #     ),
    # )
    decoder = PerceiverClassificationDecoder(
        PERCEIVER_CONFIG,
        num_channels=PERCEIVER_CONFIG.d_latents,
        trainable_position_encoding_kwargs=dict(num_channels=PERCEIVER_CONFIG.d_latents, index_dims=1),
        use_query_residual=True,
    )
    perceiver = PerceiverModel(
        PERCEIVER_CONFIG,
        input_preprocessor=perceiver_text_preprocessor,
        decoder=decoder
    ).to(DEVICE)
    model_memory_size = objsize.get_deep_size(perceiver.base_model)
    model_parameter_size = count_parameters(model=perceiver.base_model)
    for idx, (texts, lengths, labels) in enumerate(dataloader):
        texts: PackedSequence

        # print(idx, texts.data.size(), labels)

    # you can then do a forward pass as follows:
    TEXT_BATCH = [
        "There are some unseen words here banana zen 😁 but also some seen text",
        "Lots of unknowns in this one I expect",
        "text and words and tokens and stuff"
    ]
    inputs = torch.tensor([encoding.ids for encoding in TOKENISER.encode_batch(TEXT_BATCH)])

    with torch.no_grad():
        outputs = perceiver(inputs=inputs)
    logits = outputs.logits
    print(list(logits.shape))

    # to train, one can train the model using standard cross-entropy:
    criterion = torch.nn.CrossEntropyLoss()

    LABELS = torch.tensor([1])
    loss = criterion(logits, LABELS)
    print(loss)
