import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import cs236781.dataloader_utils as dataloader_utils
from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.
        # ====== YOUR CODE: ======
        # x_train, y_train = dataloader_utils.flatten(dl_train)
        # a = y_train.unique()
        # n_classes = a.nelement()
        # ========================
        x_train, y_train = dataloader_utils.flatten(dl_train)
        n_classes = len(torch.unique(y_train))
        # ========================

        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        for i in range(n_test):
            # TODO:
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            #  - Don't use an explicit loop.
            # ====== YOUR CODE: ======
            nearest_neighbors, indices = torch.topk(dist_matrix[:, i], k=self.k, dim=0, largest=False)
            nearest_labels = torch.index_select(self.y_train, dim=0, index=indices)
            label, index = torch.mode(nearest_labels)
            y_pred[i] = label
            # ========================

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Notes:
    #  - Use only basic pytorch tensor operations, no external code.
    #  - Solution must be a fully vectorized implementation, i.e. use NO
    #    explicit loops (yes, list comprehensions are also explicit loops).
    #    Hint: Open the expression (a-b)^2. Use broadcasting semantics to
    #    combine the three terms efficiently.

    dists = None
    # ====== YOUR CODE: ======(
    dist1 = -2*torch.mm(x1, x2.t())
    dist2 = torch.sum(x1.pow(2), dim=1)
    dist3 = torch.sum(x2.pow(2), dim=1)
    d2 = dist2.repeat(dist3.size()[0], 1)
    d3 = dist3.repeat(dist2.size()[0], 1)
    dists = torch.sqrt(dist1 + d2.t() + d3)
    # ========================
    # n1 = x1.size(0)
    # n2 = x2.size(0)
    #
    # norm_x1 = torch.sum(x1**2, dim=1, keepdim=True)
    # norm_x2 = torch.sum(x2**2, dim=1, keepdim=True)
    #
    # norm_x1 = norm_x1.expand(n1, n2)
    # norm_x2 = norm_x2.transpose(0, 1).expand(n1, n2)
    #
    # norm = norm_x1 + norm_x2
    #
    # dists = torch.sqrt(norm - 2 * x1.mm(x2.transpose(0, 1)))
    # ========================
    # new_x1 = torch.sum(x1**2, 1)
    # new_x2 = torch.sum(x2**2, 1)
    #
    # transposed_x2 = x2.t()
    # mat = torch.mm(x1, transposed_x2)
    #
    # reshaped_x1 = torch.reshape(new_x1, [x1.shape[0], 1])
    # reshaped_x2 = torch.reshape(new_x2, [1, x2.shape[0]])
    #
    # dists = torch.sqrt(reshaped_x1 -2*mat + reshaped_x2)
    # ========================
    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    accuracy = None
    # ====== YOUR CODE: ======
    diff = torch.eq(y, y_pred)
    sum = torch.sum(diff).item()

    accuracy = sum / diff.nelement()
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        tmp_list = []
        for j in range(num_folds):
            dl_train, dl_valid = dataloaders.create_train_validation_loaders(ds_train, validation_ratio=(1/num_folds))
            model.train(dl_train)
            x_test, y_test = dataloader_utils.flatten(dl_valid)
            y_pred = model.predict(x_test)
            acc = accuracy(y_test, y_pred)
            tmp_list.append(acc)

        accuracies.append(tmp_list)
    # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
