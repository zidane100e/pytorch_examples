"""
# this should be called in main file
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"
"""
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import os, pathlib, sys
module_dir = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(module_dir))
from kbutils.evaluation import accuracy

def get_MLP(dim_hiddens, activation=nn.ReLU(), dropout=0.1, end=False):
    """
    help to easily build an MLP model
    >>> encoder = get_MLP([784, 128, 64, 32])
    
    :param dim_hiddens: list of hidden layer sizes
    :param end: True = regression(no activation at the end), False = activation at the end
    :return: Sequential of pytorch MLP
    """
    def get_a_layer(n_in, n_out, activation, dropout, end=False):
        seq = [nn.Dropout(dropout), nn.Linear(n_in, n_out),
                activation]
        if end is True:
            return seq[:-1]
        else:
            return seq
    
    layers = []
    ii = 0
    n_hidden = len(dim_hiddens)
    for n_in, n_out in zip(dim_hiddens, dim_hiddens[1:]):
        if ii == n_hidden-1-1: # at last layer
            layer = get_a_layer(n_in, n_out, activation, dropout, end=end)
        else:
            layer = get_a_layer(n_in, n_out, activation, dropout, end=False)
        layers.append(layer)
        ii += 1
        
    layers = [ x for xs in layers for x in xs ]
    return nn.Sequential(*layers)

class Net(nn.Module):
    """
    This class defines network learning and test behaviors. 
    It gets model, loss and optimizer
    """
    def __init__(self, model=None, loss=None, 
                 optimizer=None, device='cuda'):
        super(Net, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
    
    def set_train(self):
        """
        initialization for training
        """
        self.model.train()
    
    def set_eval(self):
        """
        initialization for evaluation
        """
        self.model.eval()
    
    def init_weights(self):
        """
        describe weight initialization method
        """
        pass
    
    def forward(self, x):
        """
        describe network forward action
        """
        return self.model(x)
    
    def run_batch(self, i_batch, data):
        """
        learning method for a batch.
        You can override this in sub-class
        """
        self.optimizer.zero_grad()
        data_in, tgt = data
        data_in = data_in.to(self.device)
        tgt = tgt.to(self.device)
        out = self.forward(data_in)
        loss = self.loss(out, tgt)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item()
    
    def run_train(self, n_epoch, data, test_data=None, eval_step=1):
        """
        training method definition
        :param data: training data
        :param test_data: validation data
        """
        for i_epoch in range(n_epoch):
            self.set_train()
            loss = 0
            for i_batch, data_batch in enumerate(data):
                loss_temp = self.run_batch(i_batch, data_batch)
                loss += loss_temp
            loss /= 1.0*len(data)
            print('epoch', i_epoch, 'loss', loss)
            
            if eval_step>0 and (i_epoch+1)%eval_step==0:
                if test_data is None:
                    print('eval_train', end=' ')
                    self.run_eval(data)
                else:
                    self.run_eval(test_data)
        
    def run_eval(self, data):
        """
        test method definition
        :return: (prediction, target, loss)
        """
        self.set_eval()
        loss = 0
        outs = None
        tgts = None
        with torch.no_grad():
            for i_batch, data_batch in enumerate(data):
                data_in, tgt = data_batch
                data_in = data_in.to(self.device)
                tgt = tgt.to(self.device)
                out = self.forward(data_in)
                loss += self.loss(out, tgt).detach().cpu()
                softmaxout = self.softmax(out).cpu().numpy()
                tgt = tgt.cpu().numpy()
                if outs is None:
                    outs = softmaxout
                    tgts = tgt
                else:
                    outs = np.concatenate((outs, softmaxout), axis=0)
                    tgts = np.concatenate((tgts, tgt), axis=0)
        loss /= 1.0*(i_batch+1)
        print('evaluate', 'loss', loss, 'accuracy', accuracy(outs, tgts))
        return outs, tgts, loss
    
class Autoencoder(Net):
    """
    class for Autoencoder
    
    >>> dims = [784, 128, 64, 32]
    >>> encoder = get_MLP(dims)
    >>> decoder = get_MLP(list(reversed(dims)))
    >>> ae_model = nn.Sequential(encoder, decoder)
    >>> loss = nn.MSELoss()
    >>> optimizer = optim.Adam(ae_model.parameters())
    >>> ae = Autoencoder(model=ae_model, loss=loss, optimizer=optimizer)
    >>> ae.run_train(100, train_loader)
    """
    def __init__(self, model=None, loss=None, 
                 optimizer=None, device='cuda'):
        super(Autoencoder, self).__init__(model, loss, optimizer, device)
    
    def run_batch(self, i_batch, data):
        self.optimizer.zero_grad()
        data_in, _ = data
        data_in = data_in.to(self.device)
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
