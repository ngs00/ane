import numpy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas
from torch_geometric.data import DataLoader
from sklearn.preprocessing import scale
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import correlation
from util.data import normalize
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from geomagnetic_strom_forecasting.storm_detection import detect_storm


class FCNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(dim_in, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, dim_out)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        out = self.fc3(h)

        return out

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        sum_train_losses = 0

        for data, targets in data_loader:
            data = data.cuda()
            targets = targets.cuda()

            preds = self(data)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_train_losses += loss.item()

        return sum_train_losses / len(data_loader)

    def predict(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.cuda()

                preds = self(data)
                list_preds.append(preds)

        return torch.cat(list_preds, dim=0)


class RNNModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, n_layers, len_seq, dim_emb, rnn_model):
        super(RNNModel, self).__init__()
        self.dim_hidden = dim_hidden
        self.n_layers = n_layers
        self.rnn_model = rnn_model

        if self.rnn_model == 'rnn':
            self.rnn = nn.RNN(dim_in, dim_hidden, n_layers, batch_first=True)
        elif self.rnn_model == 'lstm':
            self.rnn = nn.LSTM(dim_in, dim_hidden, n_layers, batch_first=True)
        elif self.rnn_model == 'gru':
            self.rnn = nn.GRU(dim_in, dim_hidden, n_layers, batch_first=True)

        self.bn1 = nn.BatchNorm1d(dim_hidden * len_seq)
        self.fc1 = nn.Linear(dim_hidden * len_seq, dim_emb)

    def forward(self, x):
        h_0 = torch.zeros(self.n_layers, x.shape[0], self.dim_hidden).cuda()

        if self.rnn_model == 'lstm':
            c_0 = torch.zeros(self.n_layers, x.shape[0], self.dim_hidden).cuda()
            z, _ = self.rnn(x, (h_0, c_0))
        else:
            z, _ = self.rnn(x, h_0)

        z = self.bn1(z.reshape(z.shape[0], -1))
        embs = self.fc1(z)

        return embs


class Transformer(nn.Module):
    def __init__(self, dim_in, dim_emb, n_layers=None, len_seq=None):
        super(Transformer, self).__init__()
        self.rnn_model = 'transformer'
        self.n_layers = n_layers
        self.rnn = nn.Transformer(dim_in, 7, 2, 2, 2, 0.05, batch_first=True)
        self.bn1 = nn.BatchNorm1d(dim_in * len_seq)
        self.fc1 = nn.Linear(dim_in * len_seq, dim_emb)

    def forward(self, src, tgt):
        z = self.rnn(src, tgt)
        z = self.bn1(z.reshape(z.shape[0], -1))
        embs = self.fc1(z)

        return embs


def get_seq_dataset(dataset, len_seq):
    seq_dataset = list()
    seq = list()

    for i in range(0, len_seq):
        seq.append(numpy.zeros(dataset.shape[1]))

    for i in range(0, dataset.shape[0]):
        seq.pop(0)
        seq.append(dataset[i, :])
        seq_dataset.append(numpy.hstack(seq))

    return numpy.vstack(seq_dataset)


def get_data_loader(*data, batch_size, shuffle=False, dtype=torch.float):
    tensors = [torch.tensor(d, dtype=dtype) for d in data]
    return DataLoader(TensorDataset(*tuple(tensors)), batch_size=batch_size, shuffle=shuffle)


def get_pos_neg_samples(seq, targets, tgt=None):
    pos_seq = list()
    pos_tgt = list()
    pos_targets = list()
    neg_seq = list()
    neg_tgt = list()
    neg_targets = list()

    for i in range(0, seq.shape[0]):
        idx = random.sample(range(0, seq.shape[0]), 2)

        if torch.norm(targets[i] - targets[idx[0]]) < torch.norm(targets[i] - targets[idx[1]]):
            pos_seq.append(seq[idx[0], :, :].view(1, seq.shape[1], seq.shape[2]))
            pos_targets.append(targets[idx[0]])
            neg_seq.append(seq[idx[1], :, :].view(1, seq.shape[1], seq.shape[2]))
            neg_targets.append(targets[idx[1]])

            if tgt is not None:
                pos_tgt.append(tgt[idx[0], :, :].view(1, tgt.shape[1], tgt.shape[2]))
                neg_tgt.append(tgt[idx[1], :, :].view(1, tgt.shape[1], tgt.shape[2]))
        else:
            pos_seq.append(seq[idx[1], :, :].view(1, seq.shape[1], seq.shape[2]))
            pos_targets.append(targets[idx[1]])
            neg_seq.append(seq[idx[0], :, :].view(1, seq.shape[1], seq.shape[2]))
            neg_targets.append(targets[idx[0]])

            if tgt is not None:
                pos_tgt.append(tgt[idx[1], :, :].view(1, tgt.shape[1], tgt.shape[2]))
                neg_tgt.append(tgt[idx[0], :, :].view(1, tgt.shape[1], tgt.shape[2]))
    if tgt is None:
        return torch.vstack(pos_seq), torch.vstack(pos_targets), torch.vstack(neg_seq), torch.vstack(neg_targets)
    else:
        return torch.vstack(pos_seq), torch.vstack(pos_tgt), torch.vstack(pos_targets),\
               torch.vstack(neg_seq), torch.vstack(neg_tgt), torch.vstack(neg_targets)


def loss_ane(seq, targets, emb_net, tgt=None):
    rand_idx = torch.randperm(seq.shape[0])
    seq_sample = seq[rand_idx, :, :]
    targets_sample = targets[rand_idx]

    if tgt is None:
        emb_data = F.normalize(emb_net(seq), p=2, dim=1)
        emb_sample = F.normalize(emb_net(seq_sample), p=2, dim=1)
    else:
        tgt_sample = tgt[rand_idx, :, :]
        emb_data = F.normalize(emb_net(seq, tgt), p=2, dim=1)
        emb_sample = F.normalize(emb_net(seq_sample, tgt_sample), p=2, dim=1)

    dist_emb = torch.sum((emb_data - emb_sample)**2, dim=1)
    dist_target = torch.sum((targets - targets_sample)**2, dim=1)

    return torch.mean((dist_emb - dist_target)**2)


def train(emb_net, data_loader, optimizer):
    if emb_net.rnn_model == 'transformer':
        return __train_transformer(emb_net, data_loader, optimizer)
    else:
        return __train_rnn(emb_net, data_loader, optimizer)


def __train_rnn(emb_net, data_loader, optimizer):
    emb_net.train()
    train_loss = 0

    for seq, targets in data_loader:
        seq = seq.cuda()
        targets = targets.cuda()
        loss = loss_ane(seq, targets, emb_net)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(data_loader)


def __train_transformer(emb_net, data_loader, optimizer):
    emb_net.train()
    train_loss = 0

    for batch in data_loader:
        src = batch[0].cuda()
        tgt = batch[1].cuda()
        targets = batch[2].cuda()
        loss = loss_ane(src, targets, emb_net, tgt=tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(data_loader)


def test(emb_net, data_loader):
    if emb_net.rnn_model == 'transformer':
        return __test_transformer(emb_net, data_loader)
    else:
        return __test_rnn(emb_net, data_loader)


def __test_rnn(emb_net, data_loader):
    emb_net.eval()
    list_embs = list()

    with torch.no_grad():
        for seq, _ in data_loader:
            seq = seq.cuda()

            embs = F.normalize(emb_net(seq), p=2, dim=1)
            list_embs.append(embs)

    return torch.cat(list_embs, dim=0)


def __test_transformer(emb_net, data_loader):
    emb_net.eval()
    list_embs = list()

    with torch.no_grad():
        for batch in data_loader:
            src = batch[0].cuda()
            tgt = batch[1].cuda()

            embs = F.normalize(emb_net(src, tgt), p=2, dim=1)
            list_embs.append(embs)

    return torch.cat(list_embs, dim=0)


def get_seq_data(x, y, len_seq):
    x_seq = []
    y_seq = []

    for i in range(0, x.shape[0] - len_seq):
        x_seq.append(x[i:i+len_seq, :])
        y_seq.append(y[i+len_seq])

    return torch.FloatTensor(x_seq), torch.FloatTensor(y_seq).view([-1, 1])


def exec_dml(dataset_name, rnn_model, init_lr_emb, l2_emb, bs_emb, dim_emb,
             init_lr_pred, l2_pred, bs_pred):
    # Experiment settings.
    path_save_file = 'results/' + dataset_name
    len_seq = 8

    # Load dataset.
    data = numpy.array(pandas.read_excel('datasets/' + dataset_name + '.xlsx'))
    x = scale(data[:, 1:-1])
    y = data[:, -1]
    y_norm = normalize(y)
    n_train = int(0.8 * x.shape[0])

    if rnn_model == 'transformer':
        x_seq, y_pred = get_seq_data(x, y_norm, len_seq=len_seq)
        y_seq = torch.vstack([x_seq[1:, :], torch.zeros((1, len_seq, x.shape[1]))])
        loader_train = DataLoader(TensorDataset(x_seq[:n_train], y_seq[:n_train], y_pred[:n_train]), batch_size=bs_emb)
        loader_test = DataLoader(TensorDataset(x_seq[n_train:], y_seq[n_train:], y_pred[n_train:]), batch_size=bs_emb)
        emb_net = Transformer(dim_in=x.shape[1], dim_emb=dim_emb, len_seq=len_seq).cuda()
    else:
        x_seq, y_seq = get_seq_data(x, y_norm, len_seq=len_seq)
        loader_train = DataLoader(TensorDataset(x_seq[:n_train], y_seq[:n_train]), batch_size=bs_emb, shuffle=False)
        loader_test = DataLoader(TensorDataset(x_seq[n_train:], y_seq[n_train:]), batch_size=bs_emb, shuffle=False)
        emb_net = RNNModel(rnn_model=rnn_model, dim_emb=dim_emb, dim_in=x.shape[1],
                           dim_hidden=16, n_layers=2, len_seq=len_seq).cuda()

    optimizer = torch.optim.Adam(emb_net.parameters(), lr=init_lr_emb, weight_decay=l2_emb)

    # Train the embedding network.
    for epoch in range(0, 500):
        train_loss = train(emb_net, loader_train, optimizer)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, 500, train_loss))

    # Generate the latent embeddings of the data using the trained embedding network.
    emb_train = test(emb_net, loader_train).cpu().numpy()
    emb_test = test(emb_net, loader_test).cpu().numpy()

    # Save embedding results.
    df_train = pandas.DataFrame(numpy.hstack([emb_train, y[:n_train].reshape(-1, 1)]))
    df_test = pandas.DataFrame(numpy.hstack([emb_test, y[n_train:-len_seq].reshape(-1, 1)]))
    df_train.to_excel(path_save_file + '/embs_train.xlsx', header=None, index=None)
    df_test.to_excel(path_save_file + '/embs_test.xlsx', header=None, index=None)

    # Load embedding results.
    dataset_train = numpy.array(pandas.read_excel(path_save_file + '/embs_train.xlsx', header=None))
    dataset_test = numpy.array(pandas.read_excel(path_save_file + '/embs_test.xlsx', header=None))
    dataset_x = scale(numpy.vstack([dataset_train[:, :-1], dataset_test[:, :-1]]))
    dataset_train_x = dataset_x[:dataset_train.shape[0], :]
    dataset_test_x = dataset_x[dataset_train.shape[0]:, :]
    dataset_train_x_seq = get_seq_dataset(dataset_train_x, len_seq)
    dataset_test_x_seq = get_seq_dataset(dataset_test_x, len_seq)
    dataset_train_y = dataset_train[:, -1].reshape(-1, 1)
    dataset_test_y = dataset_test[:, -1].reshape(-1, 1)

    # Load the dataset on the embedding space.
    data_loader_train = get_data_loader(dataset_train_x_seq, dataset_train_y, batch_size=bs_pred, shuffle=True)
    data_loader_test = get_data_loader(dataset_test_x_seq, dataset_test_y, batch_size=bs_pred)

    # Define prediction model and optimizer.
    model = FCNN(dataset_train_x_seq.shape[1], dim_out=1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr_pred, weight_decay=l2_pred)
    criterion = torch.nn.MSELoss()

    # Train the prediction model.
    for epoch in range(0, 500):
        train_loss = model.fit(data_loader_train, optimizer, criterion)
        print(epoch, train_loss)

    # Evaluate the trained model.
    preds_test = model.predict(data_loader_test).cpu().numpy()
    mae = mean_absolute_error(dataset_test_y, preds_test)
    rmse = numpy.sqrt(mean_squared_error(dataset_test_y, preds_test))
    corr = correlation(dataset_test_y, preds_test)

    # Save the trained model and the extrapolation results.
    df = pandas.DataFrame(numpy.hstack([dataset_test_y, preds_test]))
    df.to_excel(path_save_file + '/preds.xlsx', index=None, header=['targets', 'preds'])
    precision, recall, f1 = detect_storm(path_save_file)

    return mae, rmse, corr, precision, recall, f1
