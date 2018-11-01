from PIL import Image
import numpy as np
import torch
import torchvision.transforms as Tran
import torch.utils.data as Data

from torch.autograd import Variable


def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class loader(Data.Dataset):
    def __init__(self, list_file, test=False, num_class=7172):
        self.list_file = open(list_file).readlines()
        self.num_class = num_class
        self.transform = Tran.Compose([
               Tran.Resize(128),
               Tran.RandomCrop(112),
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
        labels = list(map(int, labels.strip().split(',')))
        label = np.zeros(self.num_class)
        for i in labels:
            label[i] = 1
        return img, label.astype('float32')

    def __len__(self):
        return len(self.list_file)


def train(train_loader, net, criterion, optimizer, alpha):
    net.train()
    train_loss, TP, TN, FP, FN = 0, 0, 0, 0, 0
    n = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha)

        optimizer.zero_grad()

        inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)

        outputs = net(inputs)

        loss_func = mixup_criterion(targets_a, targets_b, lam)
        loss = loss_func(criterion, outputs)

#        loss = criterion(outputs, targets)



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




def val(val_loader, net, criterion):
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
