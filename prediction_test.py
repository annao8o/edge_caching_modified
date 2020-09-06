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
import xlsxwriter

# is_cuda = torch.cuda.is_available()
is_cuda = False

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)
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
        # h_02 = Variable(torch.zeros(x.size(0), self.hidden_size))
        # c_02 = Variable(torch.zeros(x.size(0), self.hidden_size))
        #
        # for i, input in enumerate(x.chunk(x.size(1), dim=1)):

        # Propogate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        out = self.relu(out)
        out = self.dropout(out)

        return out


def split(dataframe, border, col):
    return dataframe[dataframe["date"] < date(border, 1, 1)].loc[:, col], dataframe[dataframe["date"] >= date(border, 1, 1)].loc[:,col]


def split_data(df, content_id):
    # req_df = df[:, [content_id]]
    # train_data, test_data = split(req_df, 2019, "count")

    req_data = df[[content_id]].values.astype(float)
    return req_data


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def training(model, train_loader, num_batches, epochs, criterion, optimizer):
    trn_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        trn_loss = 0.0
        for i, data in enumerate(train_loader):
            x, label = data
            if is_cuda:
                x = x.cuda()
                label = label.cuda()
            # grad init
            optimizer.zero_grad()
            # forward propagation
            model_output = model(x)
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

        #     if (i+1) % 100 == 0:
        #         print('epoch: {} / {} | step: {}/{} | trn_loss: {:.4f}'.format(epoch+1, num_epochs, i+1, batch_size, trn_loss/100))
        #         trn_loss = 0.0
        #

        '''
        # validation
        with torch.no_grad():
            val_loss = 0.0
            for i, data in enumerate(val_loader):
                x, val_label = data
                if is_cuda:
                    x = x.cuda()
                    label = label.cuda()
                val_output = model(x)
                v_loss = criterion(val_output, val_label)
                val_loss += v_loss

        # del v_loss
        # del val_output

        if epoch % 100 == 0:
            print("Epoch: {} / {} | train_loss: {:.5f} | val_loss: {:.4f}".format(epoch, epochs, trn_loss, val_loss))
        '''

        if epoch % 100 == 0:
            print("Epoch: {} / {} | train_loss: {:.5f}".format(epoch, epochs, trn_loss))

            # #학습과정 출력
            # if (i+1) % num_batches == 0:
            #     with torch.no_grad():
            #         val_loss = 0.0
            #         for j, val in enumerate(val_loader):
            #             val_x, val_label = val
            #             if is_cuda:
            #                 val_x = val_x.cuda()
            #                 val_label = val_label.cuda()
            #             val_output = model(val_x)
            #             v_loss = criterion(val_output, val_label)
            #             val_loss += v_loss
            #
            #     del val_output
            #     del v_loss
            #
            #     print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f}".format(epoch+1, num_epochs, i+1, num_batches, trn_loss/10, val_loss/len(val_loader)))

        trn_loss_list.append(trn_loss)
        # val_loss_list.append(val_loss)

    return trn_loss_list


if __name__ == "__main__":
    # dir = './data/'
    # req_file = 'requests of each content(10%).pickle'
    req_file = 'cluster_1_req_cnt'

    with open(req_file, 'rb') as f:
        req_all_contents = pickle.load(f)

    contents_idx = [i for i in range(2000)]
    df = pd.DataFrame(req_all_contents, columns=contents_idx)

    print(df.tail())
    train_loss_list = [0 for _ in range(2000)]
    result_dict = dict()
    # wb = Workbook()
    # sheet1 = wb.active
    # sheet1.title = 'loss'
    with xlsxwriter.Workbook('trn_loss_list2.xlsx') as workbook:
        worksheet = workbook.add_worksheet()

        for i in contents_idx:
            content_id = i
            print('id: ', content_id)

            data = split_data(df, content_id)

            sc = MinMaxScaler()
            training_data = sc.fit_transform(data)

            seq_length = 7
            x, y = sliding_windows(training_data, seq_length)

            train_size = 60
            test_size = 30
            test_size = len(y) - train_size
            val_size = 0.2

            dataX = Variable(torch.Tensor(np.array(x)))
            dataY = Variable(torch.Tensor(np.array(y)))

            trainX = Variable(torch.Tensor(np.array(x[:round(train_size)])))
            trainY = Variable(torch.Tensor(np.array(y[:round(train_size)])))
            train_data = TensorDataset(trainX, trainY)

            # valX = Variable(torch.Tensor(np.array(x[round(train_size * (1 - val_size)):])))
            # valY = Variable(torch.Tensor(np.array(y[round(train_size * (1 - val_size)):])))
            # val_data = TensorDataset(valX, valY)

            testX = Variable(torch.Tensor(np.array(x[train_size:90])))
            testY = Variable(torch.Tensor(np.array(y[train_size:90])))
            test_data = TensorDataset(testX, testY)

            batch_size = 10
            train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
            # val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
            test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
            print("train_loader_size: {}".format(len(train_loader)))

            num_batches = len(train_loader)
            num_epochs = 2000
            learning_rate = 0.001
            input_size = 1
            hidden_size = 100
            num_layers = 1
            output_size = 1

            model = LSTM(input_size, output_size, hidden_size, num_layers)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            trn_loss_list = training(model, train_loader, num_batches, num_epochs, criterion, optimizer)
            for k in range(len(train_loss_list)):
                train_loss_list[k] += trn_loss_list[k]

            worksheet.write_row(i, 0, trn_loss_list)
            print(trn_loss_list)
            # print(val_loss_list)

            # ## Loss graph
            # plt.figure(1)
            # x_range = range(len(trn_loss_list))
            # plt.plot(x_range, trn_loss_list, label = "train loss")
            # plt.plot(x_range, val_loss_list, label = "validation loss", linestyle = '--')
            # plt.legend()
            # plt.xlabel("Epochs")
            # plt.ylabel("Loss")
            # plt.grid(True, color='gray', alpha=0.5, linestyle='--')
            # plt.title('Training loss vs Validation loss')
            # plt.show()
            #
            # ## Validation accracy graph
            # plt.figure(2)

            ##testing
            model.eval()
            test_predict = model(testX)

            test_predict = test_predict.data.numpy()
            testY_plot = testY.data.numpy()

            test_predict = sc.inverse_transform(test_predict)
            testY_plot = sc.inverse_transform(testY_plot)
            print(test_predict)

            test_predict = test_predict.tolist()
            t = list()
            for j in range(len(test_predict)):
                for z in test_predict[j]:
                    t.append(z)
            result_dict[content_id] = t
            '''
            ## prediction result graph
            plt.figure(2)
            # plt.axvline(x=train_size, c='r', linestyle='--')
            plt.plot(testY_plot)
            plt.plot(test_predict)
            plt.autoscale(axis='x', tight=True)
            plt.ylabel('Counts')
            plt.title('Request counts prediction')
            plt.grid(True, color='gray', alpha=0.5, linestyle='--')
            plt.legend(["Actual", "Predicted"])
            plt.show()
            '''
    workbook.close()
    for k in range(len(train_loss_list)):
        train_loss_list[k] /= len(contents_idx)

    print("loss_result: ", train_loss_list)
    print(result_dict)
    result_df = pd.DataFrame(result_dict)
    print(result_df.head())

    with open('prediction_result_mat2.pickle', 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)