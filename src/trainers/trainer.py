from abc import ABCMeta, abstractmethod

import mlconfig
import mlflow
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from ..metrics import Accuracy, Average, MIOU


class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


@mlconfig.register
class Trainer(AbstractTrainer):

    def __init__(self, device, model, criterion, optimizer, scheduler, train_loader, test_loader, num_epochs):
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.best_loss = 99999
        self.epoch = 1

    def fit(self):
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
            if self.epoch > self.num_epochs*0.7:
                self.train_loader.dataset.train = False
            train_loss = self.train()
            test_loss = self.evaluate()
            if test_loss.value < self.best_loss:
                self.best_loss = test_loss.value
                torch.save(self.model, './model/best_loss_vov39_3.pt')
            self.scheduler.step()


            metrics = dict(train_loss=train_loss.value,
                           test_loss=test_loss.value,)
            mlflow.log_metrics(metrics, step=self.epoch)
            format_string = 'Epoch: {}/{}, '.format(self.epoch, self.num_epochs)
            format_string += 'train loss: {}, '.format(train_loss)
            format_string += 'test loss: {}'.format(test_loss)
            tqdm.write(format_string)

    def train(self):
        self.model.train()

        train_loss = Average()

        for x, cls_targets, cnt_targets, reg_targets in tqdm(self.train_loader):
            x = x.to(self.device)
            cls_targets = cls_targets.to(self.device)
            cnt_targets = cnt_targets.to(self.device)
            reg_targets = reg_targets.to(self.device)

            cls_logits, reg_preds, cnt_logits = self.model(x)
            loss = self.criterion(cls_logits, reg_preds, cnt_logits, cls_targets, cnt_targets, reg_targets)
            
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=x.size(0))

        return train_loss

    def evaluate(self):
        self.model.eval()

        test_loss = Average()

        with torch.no_grad():
            for x, cls_targets, cnt_targets, reg_targets in tqdm(self.test_loader):
                x = x.to(self.device)
                cls_targets = cls_targets.to(self.device)
                cnt_targets = cnt_targets.to(self.device)
                reg_targets = reg_targets.to(self.device)

                cls_logits, reg_preds, cnt_logits = self.model(x)
                loss = self.criterion(cls_logits, reg_preds, cnt_logits, cls_targets, cnt_targets, reg_targets)

                test_loss.update(loss.item(), number=x.size(0))
        
        return test_loss

    def save_checkpoint(self, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
        }

        torch.save(checkpoint, f)
        mlflow.log_artifact(f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1
