import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from datetime import date
import pickle

is_cuda = torch.cuda.is_available()

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        if is_cuda:
            self.lstm = self.lstm.cuda()
            self.fc = self.fc.cuda()
            self.relu = self.relu.cuda()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        #Propogate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        out = self.relu(out)
        out = self.dropout(out)

        return out

def model_training(m, learning_rate, num_epochs, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

    trn_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        trn_loss = 0.0
        for i, data in enumerate(train_loader):
            x, label = data
            if is_cuda:
                x = x.cuda()
                label = label.cuda()
            # grad init
            optimizer.zero_grad()
            # forward propagation
            model_output = m(x)
            # cacualte loss
            loss = criterion(model_output, label)
            # back propogation
            loss.backward()
            # weight update
            optimizer.step()

            # trn_loss summary
            trn_loss += loss.item()

            # #del (memory issue)
            # del loss
            # del model_output

        # validation
        with torch.no_grad():
            val_loss = 0.0
            for i, data in enumerate(val_loader):
                x, val_label = data
                if is_cuda:
                    x = x.cuda()
                    label = label.cuda()
                val_output = m(x)
                v_loss = criterion(val_output, val_label)
                val_loss += v_loss

        # del v_loss
        # del val_output

        if epoch % 100 == 0:
            print("Epoch: {} / {} | train_loss: {:.5f} | val_loss: {:.4f}".format(epoch, num_epochs, trn_loss, val_loss))

        trn_loss_list.append(trn_loss)
        val_loss_list.append(val_loss)

    return trn_loss_list, val_loss_list

def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        x.append(x)
        y.append(y)

    return np.array(x), np.array(y)
