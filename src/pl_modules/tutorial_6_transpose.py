import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


class TransformerPredictor_pl(pl.LightningModule):

    def __init__(self, mdl: nn.Module, optimizer, scheduler,
                 train_loader, val_loader, test_loader, **kwargs):
        """ The pytorch lighting module that configures the model and its training configuration.

        Inputs:
            mdl: the model to be trained or tested
            optimizer: the optimizer e.g. Adam
            scheduler: scheduler for learning rate schedule
            train_loader: Dataloader for training dataset
            val_loader: Dataloader for validation dataset
            test_loader: Dataloader for test dataset
        """
        super().__init__()
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.mdl = mdl
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = kwargs['num_classes']
        self.validation_epoch_outputs = []
        self.validation_preds = []
        self.save_hyperparameters()
        # print(self.num_classes)

    def configure_optimizers(self):
        """ configure the optimizer and scheduler """
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def _calculate_loss(self, batch, mode="train"):
        """ Calculation of loss and prediction accuracy using output and label"""
        # Fetch data and transform categories to one-hot vectors
        inp_data, labels = batch
        inp_data = F.one_hot(inp_data, num_classes=self.num_classes).float()

        # Perform prediction and calculate loss and accuracy
        preds = self.mdl.forward(inp_data)[0]
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logging
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        if mode == 'val' or mode == 'test':
            return loss, acc, preds.argmax(dim=-1)
        return loss, acc

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def test_dataloader(self):
        """test dataloader"""
        return self.test_loader

    def training_step(self, batch, batch_idx):
        """ Calculate training loss and accuracy after each batch """
        loss, acc = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        """ Calculate validation loss and accuracy after each batch
            Also store the intermediate validation accuracy and prediction results of first sample of the batch
        """
        val_loss, val_output, preds = self._calculate_loss(batch, mode="val")
        self.validation_preds.append([batch[1][0], preds[0]])
        self.validation_epoch_outputs.append(val_output)

    def on_validation_epoch_end(self) -> None:
        """ Calculate the validation accuracy after an entire epoch.

        Returns: validation accuracy of an entire epoch

        """
        val_acc = sum(self.validation_epoch_outputs) / len(self.validation_epoch_outputs)
        self.validation_epoch_outputs.clear()
        self.validation_preds.clear()
        self.log("val_acc", val_acc, on_step=False, on_epoch=True)

        # inp = F.one_hot(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7]], device='cuda'), num_classes=self.num_classes).float()
        # print(self.mdl.forward(inp)[0].argmax(-1))
        print(f'val_acc at epoch {self.current_epoch}:', float(val_acc.item()))

        return val_acc

    def test_step(self, batch, batch_idx):
        print(batch[0])
        """ Calculate test accuracy after each batch """
        test_acc, preds = self._calculate_loss(batch, mode="test")
        print(preds[0])
