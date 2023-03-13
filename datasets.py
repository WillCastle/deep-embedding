"""
TODO.
"""

from abc import ABCMeta, abstractmethod

from bidict import bidict
import torch
from torch.utils.data import Dataset


class Preprocessor(ABCMeta):
    """
    TODO.
    """
    @abstractmethod
    def __call__(self,) -> torch.Tensor:
        """
        TODO.
        """
        raise NotImplementedError

class TextPreprocessor(Preprocessor):
    """
    TODO.
    """
    def __init__(self,):
        """
        TODO.
        """

    def __call__(self,) -> torch.Tensor:
        """
        TODO.
        """
        processed_text =
        return torch.tensor(processed_text)

class ImagePreprocessor(Preprocessor):
    """
    TODO.
    """
    def __init__(self,):
        """
        TODO.
        """

    def __call__(self,) -> torch.Tensor:
        """
        TODO.
        """
        processed_image =
        return torch.tensor(processed_image)


class DatasetPair:
    """
    TODO.
    """
    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        """
        TODO.
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2


class DatasetNLP(Dataset):
    """
    TODO.
    """
    def __init__(self, data_path: str, preprocessor: TextPreprocessor, class_labels: list[str]):
        """
        TODO.
        """
        self.data_path = data_path
        self.label_dict = bidict({label: label_code for label_code, label in enumerate(class_labels)})

    def __getitem__(self,) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO.
        """
        data_features =
        label_code =
        return torch.tensor(data_features), torch.tensor([label_code])

    def code_from_label(self, label) -> int:
        return self.label_dict[label]

    def label_from_code(self, label_code) -> str:
        return self.label_dict.inverse[label_code]

class DatasetCV(Dataset):
    """
    TODO.
    """
    def __init__(self, data_path: str, preprocessor: ImagePreprocessor, class_labels: list[str]):
        """
        TODO.
        """
        self.data_path = data_path
        self.label_dict = bidict({label: label_code for label_code, label in enumerate(class_labels)})

    def __getitem__(self,) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO.
        """
        data_features =
        label_code =
        return torch.tensor(data_features), torch.tensor([label_code])

    def code_from_label(self, label) -> int:
        return self.label_dict[label]

    def label_from_code(self, label_code) -> str:
        return self.label_dict.inverse[label_code]
