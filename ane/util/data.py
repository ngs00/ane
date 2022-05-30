import numpy
import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


def get_numerical_data_loader(*data, batch_size, shuffle=False, dtype=torch.float):
    tensors = [torch.tensor(d, dtype=dtype) for d in data]
    return DataLoader(TensorDataset(*tuple(tensors)), batch_size=batch_size, shuffle=shuffle)


def get_graph_data_loader(dataset, batch_size, shuffle, collate):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def get_pos_neg_samples(batch):
    pos_samples = list()
    neg_samples = list()

    for i in range(0, len(batch)):
        idx = random.sample(range(0, len(batch)), 2)

        if torch.norm(batch[i].y - batch[idx[0]].y) < torch.norm(batch[i].y - batch[idx[0]].y):
            pos_samples.append(batch[idx[0]])
            neg_samples.append(batch[idx[1]])
        else:
            pos_samples.append(batch[idx[1]])
            neg_samples.append(batch[idx[0]])

    return Batch.from_data_list(pos_samples), Batch.from_data_list(neg_samples)


def normalize(data):
    data_min = numpy.min(data)
    data_max = numpy.max(data)

    return (data - data_min) / (data_max - data_min)
