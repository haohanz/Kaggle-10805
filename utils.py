import cv2
import numpy as np
import torch
import torchvision.transforms as Tran
import torch.utils.data as Data


class loader(Data.Dataset):
    def __init__(self, list_file, test=False, num_class=7178):
        self.list_file = open(list_file).readlines()
        self.num_class = num_class
        self.transform = Tran.Compose([
               Tran.RandomCrop(224),
               Tran.ColorJitter(0.2,0.2,0.2,0.05),
               Tran.RandomHorizontalFlip(),
               Tran.ToTensor(),
               Tran.Normalize((0.396, 0.431, 0.455), (0.241, 0.237, 0.243)),])
        if test:
            self.transform = Tran.Compose([
               Tran.ToTensor(),
               Tran.Normalize((0.396, 0.431, 0.455), (0.241, 0.237, 0.243)),])


    def __getitem__(self, index):
        file, labels = self.list_file[index].split('\t')
        img = cv2.imread(file)
        img = self.transform(img)
        labels = list(map(int,labels.strip().split(',')))
        lable = np.zeros(self.num_class)
        for i in labels:
            lable[i] = 1
        return img, lable.astype('float')

    def __len__(self):
        return len(self.list_file)


def train(train_loader, net, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, TP，TN，FP，FN = 0, 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = (outputs>0).float()
        TP += ((pred == 1) & (target.data == 1)).sum().item()
        TN += ((pred == 0) & (target.data == 0)).sum().item()
        FN += ((pred == 0) & (target.data == 1)).sum().item()
        FP += ((pred == 1) & (target.data == 0)).sum().item()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F2 = 5 * r * p / (r + 4*p)

    print('Loss: %.3f | F2 score'%(train_loss/(batch_idx+1), F2))
    return




def val(test_loader, net):
    net.eval()
    val_loss, TP，TN，FP，FN = 0, 0, 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        val_loss += loss.item()
        pred = (outputs>0).float()
        TP += ((pred == 1) & (target.data == 1)).sum().item()
        TN += ((pred == 0) & (target.data == 0)).sum().item()
        FN += ((pred == 0) & (target.data == 1)).sum().item()
        FP += ((pred == 1) & (target.data == 0)).sum().item()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F2 = 5 * r * p / (r + 4*p)

    print('Loss: %.3f | F2 score'%(val_loss/(batch_idx+1), F2))
    return F2