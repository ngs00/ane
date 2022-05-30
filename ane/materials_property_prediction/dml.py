import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from util.crystal import *
from materials_property_prediction.gnn import get_gnn
from util.data import normalize
from util.data import get_numerical_data_loader


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


def get_rand_samples(batch):
    return Batch.from_data_list([batch[idx] for idx in numpy.random.permutation(batch.y.shape[0])])


def loss_ane(batch, emb_net):
    samples = get_rand_samples(batch)
    batch.batch = batch.batch.cuda()
    samples.batch = samples.batch.cuda()

    emb_data = F.normalize(emb_net(batch), p=2, dim=1)
    emb_sample = F.normalize(emb_net(samples), p=2, dim=1)

    dist_emb = torch.sum((emb_data - emb_sample)**2, dim=1)
    dist_target = torch.sum((batch.y - samples.y)**2, dim=1)

    return torch.mean((dist_emb - dist_target)**2)


def train(emb_net, optimizer, data_loader):
    emb_net.train()
    train_loss = 0

    for batch in data_loader:
        batch.cuda()
        loss = loss_ane(batch, emb_net)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(data_loader)


def test(emb_net, data_loader):
    emb_net.eval()
    list_embs = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.cuda()
            embs = F.normalize(emb_net(batch), p=2, dim=1)
            list_embs.append(embs)

    return torch.cat(list_embs, dim=0)


def exec_dml(dataset_name, idx_target, gnn_model,
             init_lr_emb, l2_emb, bs_emb, dim_emb,
             init_lr_pred, l2_pred, bs_pred):
    # Experiment settings.
    path_save_file = 'results/' + dataset_name

    # Load dataset.
    dataset_train = load_dataset('datasets/' + dataset_name, 'metadata_train.xlsx', idx_target=idx_target,
                                 n_bond_feats=128, radius=5)
    dataset_test = load_dataset('datasets/' + dataset_name, 'metadata_test.xlsx', idx_target=idx_target,
                                n_bond_feats=128, radius=5)
    data_loader_train = DataLoader(dataset_train, batch_size=bs_emb, shuffle=True)
    data_loader_test = DataLoader(dataset_test, batch_size=bs_emb)
    data_loader_emb = DataLoader(dataset_train, batch_size=bs_emb)

    # Normalize target values.
    targets_train = numpy.array([x.y.item() for x in dataset_train]).reshape(-1, 1)
    targets_train_norm = normalize(targets_train)
    for i in range(0, targets_train.shape[0]):
        dataset_train[i].y.data = torch.tensor(targets_train_norm[i], dtype=torch.float).view(1, 1).cuda()

    # Define embedding network and optimization algorithm.
    emb_net = get_gnn(n_elem_feats, n_edge_feats=128, dim_out=dim_emb, gnn_model=gnn_model).cuda()
    optimizer = torch.optim.Adam(emb_net.parameters(), lr=init_lr_emb, weight_decay=l2_emb)

    # Train the embedding network.
    for epoch in range(0, 500):
        train_loss = train(emb_net, optimizer, data_loader_train)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, 500, train_loss))

    # Generate the latent embeddings of the data using the trained embedding network.
    emb_train = test(emb_net, data_loader_emb).cpu().numpy()
    emb_test = test(emb_net, data_loader_test).cpu().numpy()
    targets_test = numpy.array([x.y.item() for x in dataset_test]).reshape(-1, 1)

    # Save embedding results.
    df_train = pandas.DataFrame(numpy.hstack([emb_train, targets_train]))
    df_test = pandas.DataFrame(numpy.hstack([emb_test, targets_test]))
    df_train.to_excel(path_save_file + 'embs_train.xlsx', header=None, index=None)
    df_test.to_excel(path_save_file + 'embs_test.xlsx', header=None, index=None)

    # Load embedding results.
    dataset_train = numpy.array(pandas.read_excel(path_save_file + 'embs_train.xlsx', header=None))
    dataset_train_x = dataset_train[:, :-1]
    dataset_train_y = dataset_train[:, -1].reshape(-1, 1)
    dataset_test = numpy.array(pandas.read_excel(path_save_file + 'embs_test.xlsx', header=None))
    dataset_test_x = dataset_test[:, :-1]
    dataset_test_y = dataset_test[:, -1].reshape(-1, 1)

    # Load the dataset on the embedding space.
    data_loader_train = get_numerical_data_loader(dataset_train_x, dataset_train_y, batch_size=bs_pred, shuffle=True)
    data_loader_test = get_numerical_data_loader(dataset_test_x, dataset_test_y, batch_size=bs_pred)

    # Define prediction model and optimizer.
    model = FCNN(dataset_train_x.shape[1], dim_out=1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr_pred, weight_decay=l2_pred)
    criterion = torch.nn.L1Loss()

    # Train the prediction model.
    for epoch in range(0, 500):
        train_loss = model.fit(data_loader_train, optimizer, criterion)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, 500, train_loss))

    # Evaluate the trained model.
    preds_test = model.predict(data_loader_test).cpu().numpy()
    mae = mean_absolute_error(dataset_test_y, preds_test)
    rmse = numpy.sqrt(mean_squared_error(dataset_test_y, preds_test))
    r2 = r2_score(dataset_test_y, preds_test)

    # Save the trained model and the extrapolation results.
    df = pandas.DataFrame(numpy.hstack([dataset_test_y, preds_test]))
    df.to_excel(path_save_file + 'preds.xlsx', index=None, header=['targets', 'preds'])

    return mae, rmse, r2
