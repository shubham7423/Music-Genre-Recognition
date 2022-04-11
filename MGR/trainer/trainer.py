"""Trainer file for the model."""

import os
import torch
import torch.nn as nn
import numpy as np
import tqdm as tq
from sklearn.metrics import accuracy_score


class Trainer:
    """Trainer class for the model.

    Arguments
    _________
    model: torch.nn.Module
        The model to train.
    device: torch.device
        The device to train the model on.
    optimizer: torch.optim.Optimizer
        The optimizer to use for training.
    criterion: torch.nn.Module
        The loss function to use for training.
    scheduler: torch.optim.lr_scheduler
        The scheduler to use for training.
    """

    def __init__(self,
                 model,
                 device,
                 optimizer,
                 criterion,
                 scheduler=None,
                 model_name="abc.pt"
                 ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.model_name = model_name

    def train(self, train_loader, epoch):
        """Training loop for a specific epoch.

        Arguments
        _________
        train_loader: torch.utils.data.DataLoader
            The training data loader.
        epoch: int
            The epoch number.

        Returns
        _______
        loss: float
            The loss for the epoch.
        accuracy: float
            The accuracy for the epoch.
        """
        self.model.train()
        losses = []
        predicts = []
        correct = []
        description = "\nEPOCH: {} training".format(epoch + 1)
        train_bar = tq.tqdm_notebook(
            train_loader,
            total=len(train_loader),
            desc=description,
            position=0,
            leave=True)
        for images, labels in train_bar:
            images, labels = images.to(
                self.device), labels.type(
                torch.LongTensor).to(
                self.device)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)

            correct.extend(labels.cpu().numpy())
            predicts.extend(torch.argmax(output, dim=1).cpu().numpy())
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_bar.set_postfix(loss=np.mean(losses))
        return np.mean(losses), accuracy_score(predicts, correct)

    def val(self, val_loader, epoch):
        """Validation loop for a specific epoch.

        Arguments
        _________
        val_loader: torch.utils.data.DataLoader
            The validation data loader.
        epoch: int
            The epoch number.

        Returns
        _______
        loss: float
            The loss for the epoch.
        accuracy: float
            The accuracy for the epoch.
        """
        self.model.eval()
        losses = []
        predicts = []
        correct = []
        description = "EPOCH: {} validation".format(epoch + 1)
        with torch.no_grad():
            valid_bar = tq.tqdm_notebook(
                val_loader,
                total=len(val_loader),
                desc=description,
                position=0,
                leave=True)
            for images, labels in valid_bar:

                images, labels = images.to(
                    self.device), labels.type(
                    torch.LongTensor).to(
                    self.device)
                output = self.model(images)
                loss = self.criterion(output, labels)

                correct.extend(labels.cpu().numpy())
                predicts.extend(torch.argmax(output, dim=1).cpu().numpy())

                losses.append(loss.item())
                valid_bar.set_postfix(loss=np.mean(losses))

        return np.mean(losses), accuracy_score(predicts, correct)

    def fit(
            self,
            ES,
            model_path,
            train_loader,
            val_loader=None,
            epochs=1,
            start_epoch=0,
            fold=0,
            train_BS=64,
            valid_BS=64,
            grad_clip=None):
        """Training loop for the model.

        Arguments
        _________
        ES: EarlyStopping
            The early stopping object.
        model_path: str
            The path to save the model to.
        train_loader: torch.utils.data.DataLoader
            The training data loader.
        val_loader: torch.utils.data.DataLoader
            The validation data loader.
        epochs: int
            The number of epochs to train for.
        start_epoch: int
            The epoch to start training from.
        fold: int
            The fold number.
        train_BS: int
            The batch size for training.
        valid_BS: int
            The batch size for validation.
        grad_clip: float
            The gradient clipping value.

        Returns
        _______
        history: dict
            The training history (contains train accuracy, train loss, validation accuracy and validation loss for each epoch).
        """
        training_loss_lst = []
        training_acc_lst = []
        val_loss_lst = []
        val_acc_lst = []
        i = 1
        validation = False

        if val_loader is not None:
            validation = True

        min_loss = np.Inf
        print("=> Training on {}".format(self.device))
        for epoch in range(start_epoch, epochs):

            training_loss, training_acc = self.train(train_loader, epoch)
            training_loss_lst.append(training_loss)
            training_acc_lst.append(training_acc)

            if validation:
                val_loss, val_acc = self.val(val_loader, epoch)
                val_loss_lst.append(val_loss)
                val_acc_lst.append(val_acc)

                # self.x.append([training_loss, training_acc, val_loss, val_acc])
                print(
                    'Train Loss: {:.4f} \tTrain Acc: {:.4f} \tVal Loss: {:.4f} \tVal Acc: {:.4f}'.format(
                        training_loss,
                        training_acc,
                        val_loss,
                        val_acc))
                i += 1
                if i > ES:
                    print('\nEarly Stopping')
                    break
                if val_loss <= min_loss:
                    i = 0
                    if self.scheduler is not None:
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict()
                        }
                    else:
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()
                        }

                    model_path_1 = os.path.join(model_path, self.model_name)
                    torch.save(checkpoint, model_path_1)
                    print(
                        '\nVal loss decreased ({:.4f} -> {:.4f}), Model saved'.format(min_loss, val_loss))
                    min_loss = val_loss

            else:
                print(
                    'EPOCH: {} \tTrain Loss: {:.4f} \tTrain Acc: {:.4f} '.format(
                        epoch + 1, training_loss, training_acc))
                i += 1
                if i >= ES:
                    print('Early Stopping')
                    break
                if training_loss <= min_loss:
                    i = 0
                    print(
                        'Loss decreased ({:.4f} -> {:.4f}), model saved at {}'.format(
                            min_loss, training_loss, model_path))
                    min_loss = training_loss

                    if self.scheduler is not None:
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict()
                        }
                    else:
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()
                        }
                    model_path_1 = os.path.join((model_path, self.model_name))
                    torch.save(checkpoint, model_path_1)

            if self.scheduler is not None:
                self.scheduler.step()

            if grad_clip is not None:
                nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)
            torch.cuda.empty_cache()

        history = {
            'train_loss': training_loss_lst,
            'train_acc': training_acc_lst,
            'val_loss': val_loss_lst,
            'val_acc': val_acc_lst
        }

        return history
