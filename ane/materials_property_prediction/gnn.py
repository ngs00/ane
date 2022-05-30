import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import CGConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool
from util.crystal import *


class GCN(nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(n_node_feats, 128)
        self.gc2 = GCNConv(128, 128)
        self.gc3 = GCNConv(128, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index))
        h = F.relu(self.gc2(h, g.edge_index))
        h = F.relu(self.gc3(h, g.edge_index))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))
        out = self.fc2(h)

        return out

    def _emb(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index))
        h = F.relu(self.gc2(h, g.edge_index))
        h = F.relu(self.gc3(h, g.edge_index))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))

        return h

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch.batch = batch.batch.cuda()

                preds = self(batch)
                list_preds.append(preds)

        return torch.cat(list_preds, dim=0)

    def emb(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch.batch = batch.batch.cuda()

                preds = self._emb(batch)
                list_preds.append(preds)

        return torch.cat(list_preds, dim=0)


class CGCNN(nn.Module):
    def __init__(self, n_node_feats, n_bond_feats, dim_out):
        super(CGCNN, self).__init__()
        self.fc1 = nn.Linear(n_node_feats, 128)
        self.gc1 = CGConv(128, n_bond_feats)
        self.gc2 = CGConv(128, n_bond_feats)
        self.gc3 = CGConv(128, n_bond_feats)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, dim_out)

    def forward(self, g):
        hx = F.relu(self.fc1(g.x))
        h = F.relu(self.gc1(hx, g.edge_index, g.edge_attr))
        h = F.relu(self.gc2(h, g.edge_index, g.edge_attr))
        h = F.relu(self.gc3(h, g.edge_index, g.edge_attr))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc2(hg))
        out = self.fc3(h)

        return out

    def _emb(self, g):
        hx = F.relu(self.fc1(g.x))
        h = F.relu(self.gc1(hx, g.edge_index, g.edge_attr))
        h = F.relu(self.gc2(h, g.edge_index, g.edge_attr))
        h = F.relu(self.gc3(h, g.edge_index, g.edge_attr))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc2(hg))

        return h

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch.batch = batch.batch.cuda()

                preds = self(batch)
                list_preds.append(preds)

        return torch.cat(list_preds, dim=0)

    def emb(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch.batch = batch.batch.cuda()

                preds = self._emb(batch)
                list_preds.append(preds)

        return torch.cat(list_preds, dim=0)


class MPNN(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, dim_out):
        super(MPNN, self).__init__()
        self.nn1 = nn.Sequential(nn.Linear(n_edge_feats, 64), nn.ReLU(), nn.Linear(64, n_node_feats * 128))
        self.gc1 = NNConv(n_node_feats, 128, self.nn1)

        self.nn2 = nn.Sequential(nn.Linear(n_edge_feats, 64), nn.ReLU(), nn.Linear(64, 128 * 64))
        self.gc2 = NNConv(128, 64, self.nn2)

        self.nn3 = nn.Sequential(nn.Linear(n_edge_feats, 64), nn.ReLU(), nn.Linear(64, 64 * 64))
        self.gc3 = NNConv(64, 64, self.nn3)

        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index, g.edge_attr))
        h = F.relu(self.gc2(h, g.edge_index, g.edge_attr))
        h = F.relu(self.gc3(h, g.edge_index, g.edge_attr))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))
        out = self.fc2(h)

        return out

    def _emb(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index, g.edge_attr))
        h = F.relu(self.gc2(h, g.edge_index, g.edge_attr))
        h = F.relu(self.gc3(h, g.edge_index, g.edge_attr))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))

        return h

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch.batch = batch.batch.cuda()

                preds = self(batch)
                list_preds.append(preds)

        return torch.cat(list_preds, dim=0)

    def emb(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch.batch = batch.batch.cuda()

                preds = self._emb(batch)
                list_preds.append(preds)

        return torch.cat(list_preds, dim=0)


class TFNN(nn.Module):
    def __init__(self, n_node_feats, n_edge_feats, dim_out):
        super(TFNN, self).__init__()
        self.gc1 = TransformerConv(in_channels=n_node_feats, out_channels=128, edge_dim=n_edge_feats)
        self.gc2 = TransformerConv(in_channels=128, out_channels=128, edge_dim=n_edge_feats)
        self.gc3 = TransformerConv(in_channels=128, out_channels=128, edge_dim=n_edge_feats)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, dim_out)

    def forward(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index, g.edge_attr))
        h = F.relu(self.gc2(h, g.edge_index, g.edge_attr))
        h = F.relu(self.gc3(h, g.edge_index, g.edge_attr))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))
        out = self.fc2(h)

        return out

    def _emb(self, g):
        h = F.relu(self.gc1(g.x, g.edge_index, g.edge_attr))
        h = F.relu(self.gc2(h, g.edge_index, g.edge_attr))
        h = F.relu(self.gc3(h, g.edge_index, g.edge_attr))
        hg = global_mean_pool(h, g.batch)
        h = F.relu(self.fc1(hg))

        return h

    def fit(self, data_loader, optimizer, criterion):
        self.train()
        train_loss = 0

        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            preds = self(batch)
            loss = criterion(batch.y, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch.batch = batch.batch.cuda()

                preds = self(batch)
                list_preds.append(preds)

        return torch.cat(list_preds, dim=0)

    def emb(self, data_loader):
        self.eval()
        list_preds = list()

        with torch.no_grad():
            for batch in data_loader:
                batch.batch = batch.batch.cuda()

                preds = self._emb(batch)
                list_preds.append(preds)

        return torch.cat(list_preds, dim=0)


def get_gnn(n_node_feats, n_edge_feats, dim_out, gnn_model):
    if gnn_model == 'gcn':
        return GCN(n_node_feats, dim_out)
    elif gnn_model == 'cgcnn':
        return CGCNN(n_node_feats, n_edge_feats, dim_out)
    elif gnn_model == 'mpnn':
        return MPNN(n_node_feats, n_edge_feats, dim_out)
    elif gnn_model == 'tfnn':
        return TFNN(n_node_feats, n_edge_feats, dim_out)
    else:
        return None
