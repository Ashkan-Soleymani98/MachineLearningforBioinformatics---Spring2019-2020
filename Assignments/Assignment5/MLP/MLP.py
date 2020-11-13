from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
import torch
import numpy as np
from sklearn.datasets import load_iris
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

class MLP:
    def __init__(self):
        input_size = 4
        hidden_size = 5
        output_size = 3
        self.hist = {'loss':[], 'acc':[]}
        
        self.W1 = Variable(torch.randn(input_size, hidden_size))
        self.b1 = Variable(torch.randn(hidden_size))
        self.W2 = Variable(torch.randn(hidden_size, output_size))
        self.b2 = Variable(torch.randn(output_size))

    def softmax(self, x):
        e = torch.exp(x - torch.max(x))
        return e / e.sum(dim=1)[:, None]
    
    def relu(self, x):
        return torch.max(x, torch.zeros_like(x))
    
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def cross_entropy(self, y, o):
        t = y * torch.log(o + 1e-10)
        return -torch.sum(t) / y.shape[0]
    
    def forward(self, x):
        h = self.sigmoid(torch.matmul(x, self.W1) + self.b1)
        o = self.softmax(torch.matmul(h, self.W2) + self.b2)
        
        self.x = x
        self.h = h
        self.activation1 = torch.matmul(x, self.W1) + self.b1
        return o
    
    @staticmethod
    def one_hot(y):
        y_numpy = y.numpy().astype(int)
        return torch.tensor(np.eye(int(np.max(y_numpy)) + 1)[y_numpy]).float()
    
    @staticmethod
    def one_hot_to_label(y):
        y_numpy = y.numpy().astype(float)
        return torch.tensor(np.argmax(y_numpy, axis=1))
    
    def backward(self, y, o, lr=3e-6):
        loss_activation2_grad = (o-y)
        loss_b2_grad = torch.sum(loss_activation2_grad, dim=0)
        loss_w2_grad = torch.matmul(self.h.t(), loss_activation2_grad)
        loss_activation1_grad = torch.matmul(loss_activation2_grad, self.W2.t()) * (torch.exp(-self.activation1) / (1 + torch.exp(-self.activation1)))

        loss_b1_grad = torch.sum(loss_activation1_grad, dim=0)
        loss_w1_grad = torch.matmul(self.x.t(), loss_activation1_grad)
        
        self.b1 -= lr * loss_b1_grad
        self.W1 -= lr * loss_w1_grad
        self.b2 -= lr * loss_b2_grad
        self.W2 -= lr * loss_w2_grad
    
    def train(self, x, y, epochs):
        y1 = self.one_hot(y)
        for epoch in tqdm(range(1, epochs+1)):
            o = self.forward(x)
            self.backward(y1, o)
            
            loss = self.cross_entropy(y1, o)
            acc = accuracy_score(y, self.one_hot_to_label(o))
            self.hist['loss'] += [loss]
            self.hist['acc'] += [acc]
            print(epoch, 'loss:', loss, 'acc:', acc)