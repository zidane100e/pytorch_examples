"""
# this should be called in main file
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

def get_MLP(n_hiddens, activation=nn.ReLU(), dropout=0.1):
    def get_a_layer(n_in, n_out, activation, dropout):
        seq = [nn.Dropout(dropout), nn.Linear(n_in, n_out),
                activation]
        return seq
    layers = [get_a_layer(n_in, n_out, activation, dropout) for 
              n_in, n_out in zip(n_hiddens, n_hiddens[1:])]
    layers = [ x for xs in layers for x in xs ]
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self, model=None, loss=None, 
                 optimizer=None):
        super(Net, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
    
    def run_batch(self, i_batch, data):
        self.optimizer.zero_grad()
        data_in, tgt = data
        data_in = data_in.to(device)
        tgt = tgt.to(device)
        out = self.model(data_in)
        loss = self.loss(out, tgt)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()
    
    def run_train(self, n_epoch, data):
        self.model.train()
        for i_epoch in range(n_epoch):
            loss = 0
            n_batch = len(data)
            for i_batch, data_batch in enumerate(data):
                loss_temp = self.run_batch(i_batch, data_batch)
                loss += loss_temp
                #print(i_batch, loss_temp)
            loss /= 1.0*n_batch
            print('epoch', i_epoch, 'loss', loss)
            
    def run_eval(self, data):
        self.model.eval()
        loss = 0
        for i_batch, data_batch in enumerate(data):
            data_in, tgt = data_batch
            out = self.model(data_in)
            loss += self.loss(out, tgt).detach().cpu()
        loss /= 1.0*i_batch
        return out, loss
    
class Autoencoder(Net):
    def __init__(self, model=None, loss=None, 
                 optimizer=None):
        super(Autoencoder, self).__init__(model, loss, optimizer)
    
    def run_batch(self, i_batch, data):
        self.optimizer.zero_grad()
        data_in, _ = data
        data_in = data_in.to(device)
        out = self.model(data_in)
        loss = self.loss(out, data_in)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()    



## TEST
if __name__ == '__main__':
    device = torch.device("cuda")
    
    # classifier
    dim_mnist = 784

    encoder = get_MLP([784, 300, 100, 10])
    #decoder = get_MLP([100, 300, 784])
    #ae_model = nn.Sequential(encoder, decoder)
    encoder = encoder.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(encoder.parameters())
    classifier = Net(model=encoder, 
                    loss=loss, optimizer=optimizer)
    classifier.run_train(20, train_loader)
    
    # autoencoder
    dim_mnist = 784
    #dims = [784, 300, 300]
    dims = [784, 128, 64, 32]
    encoder = get_MLP(dims)
    decoder = get_MLP(list(reversed(dims)))
    ae_model = nn.Sequential(encoder, decoder)
    ae_model = ae_model.to(device)

    loss = nn.MSELoss()
    optimizer = optim.Adam(ae_model.parameters())
    ae = Autoencoder(model=ae_model, 
                    loss=loss, optimizer=optimizer)
    ae.run_train(100, train_loader)
    
    # reconstruction check
    import matplotlib.pyplot as plt

    for i_batch, data_batch in enumerate(test_loader):
        if i_batch > 0: 
            break
        with torch.no_grad():
            data_in, tgt = data_batch
            ii = 50
            #data_in0 = data_in[:1]
            data_in0 = data_in[ii:ii+1]
            data_in0 = data_in0.to(device)
            data_in1 = data_in0.cpu().squeeze(0).view(28,28).numpy()
            tgt0 = tgt[ii]
            out = ae.model(data_in0)
            out = out.cpu()
            out = out.squeeze(0)
            out = out.view(28,28).numpy()
            plt.subplot(1,2,1)
            plt.imshow(data_in1, cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(out, cmap='gray')
            print(tgt0.item())