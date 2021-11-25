import numpy as np
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from dataloader import myDataSet
from network import myNetwork
from loss import myLoss
def parsing_hyperparas(args):
    hypara = {}
    hypara['E'] = {}
    hypara['L'] = {}
    # 挨个赋值
    for arg in vars(args):
        hypara[str(arg)[0]][str(arg)] = getattr(args, arg)
    return hypara

def main(args):
    hypara = parsing_hyperparas(args)
    # Choose the CUDA device
    if 'E_CUDA' in hypara['E']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hypara['E']['E_CUDA'])
    # create dataset
    train_dataset = myDataSet('train')
    test_dataset = myDataSet('test')
    # create dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=hypara['L']['L_batch_size'],
                                  shuffle=True,
                                  num_workers=int(hypara['E']['E_workers']),
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=hypara['L']['L_batch_size'],
                                  shuffle=True,
                                  num_workers=int(hypara['E']['E_workers']),
                                  pin_memory=True)
    # Create Model
    Network = myNetwork(hypara).cuda()
    Network.train()

    # Create Loss Function
    loss_func = myLoss(hypara).cuda()

    # Create Optimizer
    optimizer = optim.Adam(Network.parameters(), lr=hypara['L']['L_base_lr'],betas=(hypara['L']['L_adam_beta1'], 0.999))

    # train
    for epoch in range(hypara['L']['L_epochs']):
        for i, data in enumerate(train_dataloader, 0):
            faceimg, label = data
            faceimg, label = faceimg.cuda(), label.cuda()
            optimizer.zero_grad()
            output = Network(faceimg)
            loss = loss_func(output,label)
            loss.backward()
            optimizer.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--E_CUDA', default=0, type=int, help='Index of CUDA')
    parser.add_argument('--E_workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--L_epochs', default=1000, type=int, help='Number of epochs')
    parser.add_argument('--L_batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--L_base_lr', default=6e-4, type=float, help='Learning rate')
    parser.add_argument('--L_adam_beta1', default=0.9, type=float, help='Adam beta1')
    args = parser.parse_args()
    main(args)