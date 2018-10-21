from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as Data


import os
import argparse

from models import DPN26 as net
from utils import loader, train, val



parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--model_name', default='model', type=str, help='name of the saved model')
parser.add_argument('--num_epoch', default=100, type=int, help='number of epoch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

best_F2 = 0


net = net().to('cuda')
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}.t7'.format(args.model_name))
    net.load_state_dict(checkpoint['net'])
    best_F2 = checkpoint['F2']


train_loader = Data.DataLoader(loader('datafile/train.txt'),
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=8, 
                              drop_last=True)
test_loader = Data.DataLoader(loader('datafile/val.txt',test=True), 
                              batch_size=args.batch_size,
                              num_workers=8)
def loss_F(outputs, targets, w=10):
    p = 1/(1+torch.exp(-outputs))
    loss = targets*torch.log(p)+(1-targets)*torch.log(1-p)*w
    loss = -loss.mean()/(w+1)
    return loss

criterion = loss_F
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-6)




for epoch in range(args.num_epoch):
    print('Epoch %d'%(epoch))
    train(train_loader, net, criterion, optimizer)
    if epoch%5==0: F2 = val(test_loader, net, criterion)
    if F2 > best_F2:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'F2': F2}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.t7'.format(args.model_name))
        best_F2 = F2
