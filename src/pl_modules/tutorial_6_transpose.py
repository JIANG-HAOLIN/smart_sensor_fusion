import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


class TransformerPredictor(pl.LightningModule):

    def __init__(self, mdl: nn.Module, optimizer, scheduler, train_loader, val_loader, **kwargs):
        """
        Inputs:
            mdl: the model to be trained or tested
            optimizer: the optimizer e.g. Adam
            scheduler: scheduler for learning rate schedule
            train_loader: Dataloader for training dataset
            val_loader: Dataloader for validation dataset
        """
        super().__init__()
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.mdl = mdl
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = kwargs['num_classes']
        self.validation_epoch_outputs = []
        print(self.num_classes)

    def configure_optimizers(self):
        # return [self.optimizer], [self.scheduler]
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform categories to one-hot vectors
        inp_data, labels = batch
        inp_data = F.one_hot(inp_data, num_classes=self.num_classes).float()

        # Perform prediction and calculate loss and accuracy
        preds = self.mdl.forward(inp_data, add_positional_encoding=True)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logging
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss, acc

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss, val_output = self._calculate_loss(batch, mode="val")
        self.validation_epoch_outputs.append(val_output)

    def on_validation_epoch_end(self) -> None:
        val_acc = sum(self.validation_epoch_outputs)/len(self.validation_epoch_outputs)
        self.validation_epoch_outputs.clear()
        self.log("val_acc", val_acc, on_step=False, on_epoch=True)
        return val_acc

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
