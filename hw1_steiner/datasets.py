import numpy as np

import torch
from torch.utils.data import Dataset


class RandomImageDataset(Dataset):
    """
    A dataset returning random noise images of specified dimensions
    """

    def __init__(self, num_samples, num_classes, C, W, H):
        """
        :param num_samples: Number of samples (labeled images in the dataset)
        :param num_classes: Number of classes (labels)
        :param C: Number of channels per image
        :param W: Image width
        :param H: Image height
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image_dim = (C, W, H)

    def __getitem__(self, index):

        # TODO: Create a random image tensor and return it.
        # Bonus if you make sure to always return the same image for the
        # same index (make it deterministic per index), but don't mess-up
        # RNG state outside this method.
        # ====== YOUR CODE: ======
        if index >= self.num_samples or index < 0:
            raise ValueError()
        if not hasattr(self, 'dict1'):
            self.dict1 = {}
        if index not in self.dict1:
            image = torch.randint(255, self.image_dim)
            x = np.random.random(1)
            x *= self.num_classes
            self.dict1[index] = (image, int(x))

        return self.dict1[index]
        # ========================

    def __len__(self):

        # ====== YOUR CODE: ======
        return self.num_samples
        # ========================


class SubsetDataset(Dataset):
    """
    A dataset that wraps another dataset, returning a subset from it.
    """
    def __init__(self, source_dataset: Dataset, subset_len, offset=0):
        """
        Create a SubsetDataset from another dataset.
        :param source_dataset: The dataset to take samples from.
        :param subset_len: The total number of sample in the subset.
        :param offset: The offset index to start taking samples from.
        """
        if offset + subset_len > len(source_dataset):
            raise ValueError("Not enough samples in source dataset")

        self.source_dataset = source_dataset
        self.subset_len = subset_len
        self.offset = offset

    def __getitem__(self, index):
        # TODO: Return the item at index + offset from the source dataset.
        # Make sure to raise an IndexError if index is out of bounds.

        # ====== YOUR CODE: ======
        if index < 0 or index >= self.subset_len:
            raise IndexError()
        return self.source_dataset[self.offset + index]
        # ========================

    def __len__(self):
        # ====== YOUR CODE: ======
        return self.subset_len
        # ========================