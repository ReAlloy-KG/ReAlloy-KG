import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy import sparse
from torch.utils.data import DataLoader
import csv
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# log日志
class Logger(object):
    def __init__(self, file="Default.log"):
        self.terminal = sys.stdout
        self.log = open(file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()

sys.stdout = Logger('Results/Console_log.txt')
# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def r2_mae(y_act, y_pre):
    error = []
    for ap in range(len(y_act)):
        error.append(y_act[ap] - y_pre[ap])
        '''
        print("Errors: ", Error)
        print(Error)
        '''
    squared_error = []
    abs_error = []
    for ae in error:
        squared_error.append(ae * ae)  # target-prediction之差平方
        abs_error.append(abs(ae))  # 误差绝对值

    target_deviation = []
    target_mean = sum(y_act) / len(y_act)  # target平均值
    for ac in y_act:
        target_deviation.append((ac - target_mean) * (ac - target_mean))

    r1 = []
    for n in range(len(y_act)):
        r1.append(y_act[n] * y_pre[n])
    r2 = np.square(len(y_act) * sum(r1) - sum(y_act) * sum(y_pre))
    r3 = []
    for n in range(len(y_act)):
        r3.append(y_pre[n] * y_pre[n])
    r4 = len(y_act) * sum(r3) - sum(y_pre) * sum(y_pre)
    r5 = []
    for n in range(len(y_act)):
        r5.append(y_act[n] * y_act[n])
    r6 = len(y_act) * sum(r5) - sum(y_act) * sum(y_act)
    rr2 = r2 / (r4 * r6)
    mae = sum(abs_error) / len(abs_error)
    r2_2 = r2_score(y_act, y_pre)
    mae2 = mean_absolute_error(y_act, y_pre)
    return rr2, mae, r2_2, mae2

csv_file2 = csv.reader(open('Data_classfication_random/Random1.csv',
                            encoding='utf-8-sig'))
Random11 = []
for content in csv_file2:
    content = list(map(float, content))
    if len(content) != 0:
        Random11.append(content)
Random11 = np.array(Random11)

start1 = time.perf_counter()

df = pd.read_excel("data.xlsx")
feature_name = [column for column in df][:-1]
sample_data = df.values[:, :]  # 多尺度数据集
Data_feature = sample_data[:, :-1]
Data_target = sample_data[:, -1]

ss_feature = preprocessing.StandardScaler()
Data_feature_nor = ss_feature.fit_transform(Data_feature)

ss_target = preprocessing.StandardScaler()
Data_target_nor = ss_target.fit_transform(Data_target.reshape(-1, 1))

# 邻接矩阵
neighbor_matrix = []
csv_file1 = csv.reader(open('data_matrix.csv',encoding='utf-8-sig'))
for content in csv_file1:
    content = list(map(float, content))
    if len(content) != 0:
        neighbor_matrix.append(content)

print('neighbor_matrix=', neighbor_matrix)
neighbor_matrix = np.array(neighbor_matrix)

def manipulate_feature(feature, max_node, features):

    feature = feature.reshape(-1, 1)
    feature[:, [0]] = (feature[:, [0]]) #
    result = np.zeros((max_node, features))
    result[:feature.shape[0], :feature.shape[1]] = feature
    feature = result
    feature = sparse.csr_matrix(feature)
    return feature

def normalize_adj(neighbor, max_node, feature):
    np.fill_diagonal(neighbor, 1)
    neighbor = sparse.csr_matrix(neighbor)
    return neighbor

def tensor_to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x.float())
    #return Variable(x)

def variable_to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    return x

class GraphTrainSet(Dataset):
    def __init__(self, train_x, train_y):
        max_node = 19
        num_features = 1
        neighbor_matrix_nor = neighbor_matrix

        for i in range(len(train_x)):
            # feature data manipulation
            train_feature = manipulate_feature(train_x[i], max_node, num_features)
            # normalize the adjacency matrix
            train_adj_matrix = normalize_adj(neighbor_matrix_nor, max_node, train_x[i])
            train_label = [train_y[i]]

            train_multiple_neighbor = [train_adj_matrix]
            train_multiple_feature = [train_feature]

            if i == 0:
                train_adjacency_matrix, train_node_attr_matrix, train_label_matrix = train_multiple_neighbor, train_multiple_feature, train_label
            else:
                train_adjacency_matrix, train_node_attr_matrix, train_label_matrix = np.concatenate((train_adjacency_matrix, train_multiple_neighbor)), \
                                                                                     np.concatenate((train_node_attr_matrix,train_multiple_feature)), \
                                                                                     np.concatenate((train_label_matrix,train_label))
        train_label_matrix = train_label_matrix.reshape(len(train_x), 1)
        self.train_adjacency_matrix = np.array(train_adjacency_matrix)
        self.train_node_attr_matrix = np.array(train_node_attr_matrix)
        self.train_label_matrix = np.array(train_label_matrix)
        print('Training Data:')
        print('train_adjacency matrix:\t', self.train_adjacency_matrix.shape)
        print('train_node attribute matrix:\t', self.train_node_attr_matrix.shape)
        print('train_label name:\t\t', self.train_label_matrix.shape)

    def __len__(self):
        return len(self.train_adjacency_matrix)

    def __getitem__(self, idx):
        train_adjacency_matrix = self.train_adjacency_matrix[idx].todense()
        train_node_attr_matrix = self.train_node_attr_matrix[idx].todense()
        train_label_matrix = self.train_label_matrix[idx]
        train_adjacency_matrix = torch.from_numpy(train_adjacency_matrix)
        train_node_attr_matrix = torch.from_numpy(train_node_attr_matrix)
        train_label_matrix = torch.from_numpy(train_label_matrix)
        return train_adjacency_matrix, train_node_attr_matrix, train_label_matrix

class GraphTestSet(Dataset):
    def __init__(self, test_x, test_y):
        max_node = 19
        num_features = 1
        neighbor_matrix_nor = neighbor_matrix
        for i in range(len(test_x)):
            test_feature = manipulate_feature(test_x[i], max_node, num_features)
            test_adj_matrix = normalize_adj(neighbor_matrix_nor, max_node, test_x[i])
            test_label = [test_y[i]]

            test_multiple_neighbor = [test_adj_matrix]
            test_multiple_feature = [test_feature]

            if i == 0:
                test_adjacency_matrix, test_node_attr_matrix, test_label_matrix = test_multiple_neighbor, test_multiple_feature, test_label
            else:
                test_adjacency_matrix, test_node_attr_matrix, test_label_matrix = np.concatenate((test_adjacency_matrix, test_multiple_neighbor)), \
                                                                                     np.concatenate((test_node_attr_matrix,test_multiple_feature)), \
                                                                                     np.concatenate((test_label_matrix,test_label))
        test_label_matrix = test_label_matrix.reshape(len(test_x), 1)
        self.test_adjacency_matrix = np.array(test_adjacency_matrix)
        self.test_node_attr_matrix = np.array(test_node_attr_matrix)
        self.test_label_matrix = np.array(test_label_matrix)
        print('Testing Data:')
        print('test_adjacency matrix:\t', self.test_adjacency_matrix.shape)
        print('test_node attribute matrix:\t', self.test_node_attr_matrix.shape)
        print('test_label name:\t\t', self.test_label_matrix.shape)

    def __len__(self):
        return len(self.test_adjacency_matrix)

    def __getitem__(self, idx):
        test_adjacency_matrix = self.test_adjacency_matrix[idx].todense()
        test_node_attr_matrix = self.test_node_attr_matrix[idx].todense()
        test_label_matrix = self.test_label_matrix[idx]
        # print('--------------------')
        test_adjacency_matrix = torch.from_numpy(test_adjacency_matrix)
        test_node_attr_matrix = torch.from_numpy(test_node_attr_matrix)
        test_label_matrix = torch.from_numpy(test_label_matrix)

        return test_adjacency_matrix, test_node_attr_matrix, test_label_matrix

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 初始化 1.414

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化

        self.act = nn.LeakyReLU(0.4)

    def forward(self, inp, adj):

        h = torch.matmul(inp, self.W)  # [B, N, out_features]
        N = h.size()[1]
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features),
                             h.repeat(1, N, 1)], dim=-1).view(-1, N, N, 2 * self.out_features)
        e = self.act(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -1e20 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GATModel_1layer(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GATModel_1layer, self).__init__()
        self.n_class = n_class
        self.dropout = 0.1
        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        if self.n_class != 0 :
            self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, concat=False)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 将每个head得到的表示进行拼接
        # x = F.dropout(x, self.dropout, training=self.training)
        if self.n_class != 0 :
            x = F.elu(self.out_att(x, adj)) # 输出并激活
            # x = F.dropout(x, self.dropout, training=self.training)
        # return F.log_softmax(x, dim=2)  # log_softmax速度变快，保持数值稳定
        return x

class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()

        self.graph_modules = nn.Sequential(OrderedDict([
            ('GAT_layer_0', GATModel_1layer(n_feat=1, n_hid=1, n_class=0, n_heads=8)),
        ]))

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=8 * 1 * 19, out_features=64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )

        self.pre = nn.Sequential(
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            # nn.LayerNorm(16),
            nn.Dropout(0.1),
        )

        self.predict = nn.Sequential(
            nn.Linear(in_features=16, out_features=1),
        )  # output layer


    def forward(self, node_attr_matrix, adjacency_matrix):
        adjacency_matrix = adjacency_matrix.float()
        train_x = node_attr_matrix.float()
        x = train_x[:, :19, :]

        for (name, module) in self.graph_modules.named_children():
            if 'GAT_layer' in name:
                x = module(x, adj=adjacency_matrix)
            else:
                x = module(x)

        x = x.view(x.size()[0], -1)
        xc = self.fc1(x)
        # get y_mean
        x = self.pre(xc)
        y = self.predict(x)

        return y

def train(model, train_data_loader, val_data_loader, epochs, optimizer, criterion, scheduler=None, num_ens=1, beta_type=0.1,):
    print(' ')
    print("*** Training started! ***")
    print(' ')

    for epoch in range(epochs):
        model.train()

        for batch_id, (adjacency_matrix, node_attr_matrix, label_matrix) in enumerate(train_data_loader):
            adjacency_matrix = tensor_to_variable(adjacency_matrix)
            node_attr_matrix = tensor_to_variable(node_attr_matrix)
            label_matrix = tensor_to_variable(label_matrix)

            optimizer.zero_grad()
            y_pre = model(adjacency_matrix=adjacency_matrix, node_attr_matrix=node_attr_matrix)
            loss = criterion(label_matrix, y_pre)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print('Epoch: {} \tTraining Loss: {:.4f} \t'.format(epoch, loss.item()))

            scheduler.step()
    return

def get_MC_samples(network, data_loader):

    label_list, pre_list = [], []

    for batch_id, (adjacency_matrix, node_attr_matrix, label_matrix) in enumerate(data_loader):
        adjacency_matrix = tensor_to_variable(adjacency_matrix)

        inputs = tensor_to_variable(node_attr_matrix)
        labels = tensor_to_variable(label_matrix)

        network.eval()
        prediction = network(adjacency_matrix=adjacency_matrix, node_attr_matrix=inputs)

        prediction = prediction.detach().cpu().numpy()
        prediction = ss_target.inverse_transform(prediction)

        label_list.extend(variable_to_numpy(labels))
        pre_list.extend(prediction)

    label_list = np.array(label_list)
    label_list = ss_target.inverse_transform(label_list)
    pre_list = np.array(pre_list)

    return label_list, pre_list














