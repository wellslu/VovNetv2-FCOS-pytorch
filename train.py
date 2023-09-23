import argparse

import mlconfig
import mlflow
import numpy as np
import torch

import src


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config.yaml')
    parser.add_argument('-r', '--resume', type=str, default=None)
    return parser.parse_args()


def manual_seed(seed=0):
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    config = mlconfig.load(args.config)
    mlflow.log_artifact(args.config)
    mlflow.log_params(config.flat())

    manual_seed()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.model().to(device)
#     model = torch.load('./model/best_loss_vov.pt')

    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)

    criterion = config.fcosloss()

    train_loader = config.dataset(root='./data/train', list_file='./data/train.txt', train=True)

    test_loader = config.dataset(root='./data/test', list_file='./data/test.txt', train=False)

    trainer = config.trainer(device, model, criterion, optimizer, scheduler, train_loader, test_loader)
    
    if args.resume is not None:
        trainer.resume(args.resume)
    
    trainer.fit()
    torch.save(model, './model/last_vov39_3.pt')


if __name__ == '__main__':
    main()
