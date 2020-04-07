import torch
import hw1.knn_classifier as knn
import torchvision
import os
import torchvision.transforms as tvtf

import cs236781.dataloader_utils as dataloader_utils
import hw1.datasets as hw1datasets
import hw1.transforms as hw1tf

if __name__ == '__main__':
    # Test distance calculation
    y1 = torch.tensor([0, 1, 2, 3])
    y2 = torch.tensor([2, 2, 2, 2])
    tf_ds = tvtf.Compose([
        tvtf.ToTensor(),  # Convert PIL image to pytorch Tensor
        hw1tf.TensorView(-1),  # Reshape to 1D Tensor
    ])

    # Define how much data to load (only use a subset for speed)
    num_train = 10000
    num_test = 1000
    batch_size = 1024
    data_root = os.path.expanduser('~/.pytorch-datasets')
    ds_train = hw1datasets.SubsetDataset(
        torchvision.datasets.MNIST(root=data_root, download=True, train=True, transform=tf_ds), num_train)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size)
    data = knn.KNNClassifier(dl_train)
    data.train(dl_train)
    y_pred = data.predict(x_test)
