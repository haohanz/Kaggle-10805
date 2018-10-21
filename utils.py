from PIL import Image
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
               Tran.Normalize((0.455, 0.430, 0.396), (0.244, 0.237, 0.241)),])
        if test:
            self.transform = Tran.Compose([
               Tran.ToTensor(),
               Tran.Normalize((0.455, 0.430, 0.396), (0.244, 0.237, 0.241)),])


    def __getitem__(self, index):
        file, labels = self.list_file[index].split('\t')
        img = Image.open(file)
        img = self.transform(img)
        labels = list(map(int,labels.strip().split(',')))
        lable = np.zeros(self.num_class)
        for i in labels:
            lable[i] = 1
        return img, lable.astype('float32')

    def __len__(self):
        return len(self.list_file)


def train(train_loader, net, criterion, optimizer):
    net.train()
    train_loss, TP, TN, FP, FN = 0, 0, 0, 0, 0
    n = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = (outputs>-1).float()
        TP += ((pred == 1) & (targets.data == 1)).sum().item()
        TN += ((pred == 0) & (targets.data == 0)).sum().item()
        FN += ((pred == 0) & (targets.data == 1)).sum().item()
        FP += ((pred == 1) & (targets.data == 0)).sum().item()
        p = TP / (TP + FP + 1e-10)
        r = TP / (TP + FN + 1e-10)
        F2 = 5 * r * p / (r + 4*p + 1e-10)
        if batch_idx % (n//20) == 0:
            print('{}/20 passed, loss is {}, p is {}%, r is {}%, F2 is {}'.format(
                           batch_idx//(n//20), 
                           round(train_loss/(batch_idx+1)*1000,3),
                           round(p*100,3),
                           round(r*100,3),
                           round(F2,3))
                    )


    print('Epoch finished!')
    return




def val(test_loader, net, criterion):
    net.eval()
    val_loss, TP, TN, FP, FN = 0, 0, 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        val_loss += loss.item()
        pred = (outputs>0).float()
        TP += ((pred == 1) & (targets.data == 1)).sum().item()
        TN += ((pred == 0) & (targets.data == 0)).sum().item()
        FN += ((pred == 0) & (targets.data == 1)).sum().item()
        FP += ((pred == 1) & (targets.data == 0)).sum().item()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F2 = 5 * r * p / (r + 4*p)

    print('Test set loss: %.3f | F2 score: %.3f'%(val_loss/(batch_idx+1)*1000, F2))
    return F2
