"""Util functions for data used during training"""

import torch
import sklearn
import numpy as np

class MGRFeatures(torch.utils.data.Dataset):
    """Class to load the data

    Arguments:
    __________
    features: numpy.ndarray
        Features of the data
    labels: numpy.ndarray
        Labels of the data
    transform: torchvision.transforms
        Transformation to be applied to the data
    """
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """Length of the dataset
        
        Returns:
        ________
        int
        """
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        """Get item from the dataset
        
        Arguments:
        __________
        idx: int
            Index of the item
        
        Returns:
        ________
        features: torch.Tensor
            Features of the item
        label: torch.Tensor
            Label of the item
        """
        features = self.features[idx]
        if self.transform:
            features = self.transform(features)
        return features, self.labels[idx]

def get_data(features, labels, transform=None, valid_size=0.15, train_BS=64, valid_BS=64):
    """Get the data loaders
    
    Arguments:
    __________
    features: numpy.ndarray
        Features of the data
    labels: numpy.ndarray
        Labels of the data
    transform: torchvision.transforms
        Transformation to be applied to the data
    valid_size: float
        Size of the validation set
    train_BS: int
        Batch size for training
    valid_BS: int
        Batch size for validation

    Returns:
    ________
    train_loader: torch.utils.data.DataLoader
        Data loader for training
    valid_loader: torch.utils.data.DataLoader
        Data loader for validation
    """

    train_features, val_features, train_labels, val_labels = sklearn.model_selection.train_test_split(
        features, labels, shuffle=True, test_size=valid_size)
    train_dataset = MGRFeatures(train_features, train_labels, transform)
    val_dataset = MGRFeatures(val_features, val_labels, transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_BS, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=valid_BS, shuffle=True)
    return train_loader, val_loader

def load_data(features_path,  labels_path):
    """Load the data
    
    Arguments:
    __________
    features_path: str
        Path to the features
    labels_path: str
        Path to the labels
    
    Returns:
    ________
    features: torch.Tensor
        Features of the data
    labels: torch.Tensor
        Labels of the data
    """

    features = np.load(features_path)
    labels = np.load(labels_path)

    features  = torch.FloatTensor(features)
    return features, labels