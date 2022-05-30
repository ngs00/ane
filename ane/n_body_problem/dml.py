import numpy
import torch
import pandas
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import correlation
from n_body_problem.nbody import load_dataset
from n_body_problem.nbody import get_data_loader
from n_body_problem.nbody import get_pos


class FCNN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(dim_in, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
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


def train(emb_net, optimizer, data_loader):
    emb_net.train()
    train_loss = 0

    for inputs, targets in data_loader:
        inputs = inputs.cuda()
        targets = targets.cuda()

        rand_idx = torch.randperm(inputs.shape[0])
        inputs_sample = inputs[rand_idx, :]
        targets_sample = targets[rand_idx, :]

        emb_data = F.normalize(emb_net(inputs), p=2, dim=1)
        emb_sample = F.normalize(emb_net(inputs_sample), p=2, dim=1)

        dist_emb = torch.sum((emb_data - emb_sample)**2, dim=1)
        dist_target = torch.sum((targets - targets_sample)**2, dim=1)
        loss = torch.mean((dist_emb - dist_target)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(data_loader)


def test(emb_net, data_loader):
    emb_net.eval()
    list_embs = list()

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.cuda()

            embs = F.normalize(emb_net(inputs), p=2, dim=1)
            list_embs.append(embs)

    return torch.cat(list_embs, dim=0)


def exec_dml(idx_dataset, init_lr_emb, l2_emb, bs_emb, dim_emb, init_lr_pred, l2_pred, bs_pred):
    # Experiment settings.
    idx_dataset = str(idx_dataset)
    path_save_files = 'results/' + idx_dataset + '/'
    dt = 0.1

    # load training and test dataset.
    dataset_x, dataset_y = load_dataset('datasets/' + idx_dataset)
    dataset_train_x = dataset_x[:800, :]
    dataset_train_y = dataset_y[:800, :]
    dataset_test_x = dataset_x[800:, :]
    dataset_test_y = dataset_y[800:, :]
    data_loader_train = get_data_loader(dataset_train_x, dataset_train_y,
                                        batch_size=bs_emb, shuffle=True, dtype=torch.double)
    data_loader_test = get_data_loader(dataset_test_x, dataset_test_y, batch_size=bs_emb, dtype=torch.double)
    data_loader_emb = get_data_loader(dataset_train_x, dataset_train_y, batch_size=bs_emb, dtype=torch.double)

    # Define embedding network and optimization algorithm.
    emb_net = FCNN(dataset_train_x.shape[1], dim_out=dim_emb).double().cuda()
    optimizer = torch.optim.Adam(emb_net.parameters(), lr=init_lr_emb, weight_decay=l2_emb)

    # Train the embedding network.
    for epoch in range(0, 1000):
        train_loss = train(emb_net, optimizer, data_loader_train)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, 1000, train_loss))

    # Calculate embedding of the data using the trained embedding network.
    emb_train = test(emb_net, data_loader_emb).cpu().numpy()
    emb_test = test(emb_net, data_loader_test).cpu().numpy()

    # Save embedding results.
    df_train = pandas.DataFrame(numpy.hstack([emb_train, dataset_train_y]))
    df_test = pandas.DataFrame(numpy.hstack([emb_test, dataset_test_y]))
    df_train.to_excel(path_save_files + 'embs_train.xlsx', header=None, index=None)
    df_test.to_excel(path_save_files + 'embs_test.xlsx', header=None, index=None)

    # Load embedding results.
    dataset_train = numpy.array(pandas.read_excel(path_save_files + 'embs_train.xlsx', header=None))
    dataset_train_x = dataset_train[:, :dim_emb]
    dataset_train_y = dataset_train[:, dim_emb:]
    dataset_test = numpy.array(pandas.read_excel(path_save_files + 'embs_test.xlsx', header=None))
    dataset_test_x = dataset_test[:, :dim_emb]
    dataset_test_y = dataset_test[:, dim_emb:]

    # Load the dataset on the embedding space.
    data_loader_train = get_data_loader(dataset_train_x, dataset_train_y, batch_size=bs_pred, shuffle=True)
    data_loader_test = get_data_loader(dataset_test_x, dataset_test_y, batch_size=bs_pred)

    # Define prediction model and optimizer.
    model = FCNN(dim_emb, dim_out=dataset_train_y.shape[1]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr_pred, weight_decay=l2_pred)
    criterion = torch.nn.L1Loss()

    # Train the prediction model.
    for epoch in range(0, 500):
        train_loss = model.fit(data_loader_train, optimizer, criterion)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, 500, train_loss))

    # Evaluate the trained model.
    dataset_x, dataset_y = load_dataset('datasets/' + idx_dataset)
    dataset_test_x = dataset_x[800:, :]
    dataset_test_y = dataset_y[800:, :]
    pos_init = dataset_test_x[0, :9]
    pos_test = dataset_test_x[1:, :9]
    preds_vel_test = model.predict(data_loader_test).cpu().numpy()
    preds_pos_test = get_pos(pos_init, preds_vel_test, dt=dt)

    mae = mean_absolute_error(dataset_test_y, preds_vel_test)
    rmse = numpy.sqrt(mean_squared_error(dataset_test_y, preds_vel_test))
    corr_dist = correlation(dataset_test_y.flatten(), preds_vel_test.flatten())

    # Save the trained model and the extrapolation results.
    df = pandas.DataFrame(numpy.hstack([pos_test, preds_pos_test]))
    df.to_excel(path_save_files + 'preds_pos.xlsx', index=None, header=None)
    df = pandas.DataFrame(numpy.hstack([dataset_test_y, preds_vel_test]))
    df.to_excel(path_save_files + 'preds_vel.xlsx', index=None, header=None)

    return mae, rmse, corr_dist
