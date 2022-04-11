"""File containing util functions for training"""

import torch


def get_model_configs(model, train_size, learning_rate, num_epochs):
    """Get the configs to triain the model

    Arguments:
    __________
    model: torch.nn.Module
        Model to be trained
    train_size: int
        Size of the training set
    learning_rate: float
        Learning rate

    Returns:
    ________
    criterion: torch.nn.CrossEntropyLoss
        Loss function
    optimizer: torch.optim.Adam
        Optimizer
    scheduler: torch.optim.lr_scheduler.StepLR
        Learning rate scheduler
    """

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, learning_rate, epochs=num_epochs, steps_per_epoch=train_size)
    return criterion, optimizer, scheduler
