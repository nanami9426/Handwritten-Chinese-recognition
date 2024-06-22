import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
from dataset import get_data_loader
from net import HWNet


def train(net,data_loader,loss_fn,optimizer,device):
    size = len(data_loader.dataset)
    net.train()
    for batch,(X,y) in enumerate(data_loader):
        X = X.to(device)
        y = y.float().to(device)
        y_hat = net(X)
        loss = loss_fn(y_hat,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch%2==0:
            loss = loss.item()
            current_batch = (batch+1)*len(X)
            print(f' loss:{loss:>.5f},[{current_batch:>5d}/{size:>5d}]',end='\r')
    print(f"Already trained an epoch,waiting for validate ...")

def test(net,data_loader,loss_fn,device):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    net.eval()
    correct,test_loss = 0,0
    with torch.no_grad():
        for X,y in data_loader:
            X,y = X.to(device),y.float().to(device)
            y_hat = net(X)
            test_loss += loss_fn(y_hat,y).item()
            for i,j in zip(y,y_hat):
                if i.argmax() == j.argmax():
                    correct += 1
        test_loss /= num_batches
        correct /= size
        print(f"Test error:\n accuracy: {(100*correct):0.1f}%, avg loss: {test_loss:.5f}")

def run(net,loss_fn,optimizer,epochs):
    for t in range(epochs):
        print(f"Epoch {t+1}")
        start = time.time()
        train(net,trn_loader , loss_fn, optimizer,device)
        test(net,tst_loader, loss_fn,device)
        torch.save(net.state_dict(), 'handwriting.params')
        end = time.time()
        interval = (end-start)
        print(f"Time : {interval:.3f}s\n-------------------------------")
    print('***End of train***')
    test(net,val_loader,loss_fn,device)

if __name__ == '__main__':
    device = torch.device("cuda")
    batch_size=16
    lr = 3e-3
    epochs = 12
    tst_data_dir = '../data_for_test/test'
    trn_data_dir = '../data_for_test/train'
    trn_loader,num_labels_trn,tsn_set = get_data_loader(trn_data_dir,batch_size,True)
    val_loader,tst_loader,num_labels_tst,val_set,tst_set = get_data_loader(tst_data_dir,batch_size,True,False)
    net = HWNet(num_labels_trn).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    run(net=net,loss_fn=loss_fn,optimizer=optimizer,epochs=epochs)