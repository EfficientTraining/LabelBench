# Ported from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, path='checkpoint.pt'):
        """
        :param int patience: How long to wait after last time validation loss improved.
        :param bool verbose: If True, prints a message for each validation loss improvement. Default: False.
        :param float delta: Minimum change in the monitored quantity to qualify as an improvement. Default: 0.
        :param str path: Path for the checkpoint to be saved to. Default: 'checkpoint.pt'.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.train_loss_min = np.Inf
        self.delta = delta
        self.path = path

        if self.patience is None:
           print("EarlyStopping is disabled.")

    def __call__(self, val_loss, model = None, train_loss = None):
        if self.patience is None:
            pass
        else:
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, train_loss, model) 
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, train_loss, model) 
                self.counter = 0

    def save_checkpoint(self, val_loss, train_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). ')
            if train_loss is not None:
                print(f'Train loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}). ')
        if model is not None:
            torch.save(model.state_dict(), self.path) 
        self.val_loss_min = val_loss
        if train_loss is not None:
            self.train_loss_min = train_loss