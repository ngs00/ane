import numpy
import pandas
import torch
import ast
from tqdm import tqdm
from mendeleev.fetch import fetch_table
from mendeleev import element
from sklearn import preprocessing
from pymatgen.core.structure import Structure
from torch_geometric.data import Data


atom_nums = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O':8, 'F': 9, 'Ne': 10,
             'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
             'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
             'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
             'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
             'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
             'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
             'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
             'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
             'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
             'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110}
atom_syms = {v: k for k, v in atom_nums.items()}
elem_feat_names = ['atomic_number', 'period', 'en_pauling', 'covalent_radius_bragg',
                   'electron_affinity', 'atomic_volume', 'atomic_weight', 'fusion_heat']
n_elem_feats = len(elem_feat_names) + 1


def get_elem_feats():
    tb_atom_feats = fetch_table('elements')
    elem_feats = numpy.nan_to_num(numpy.array(tb_atom_feats[elem_feat_names]))
    ion_engs = numpy.zeros((elem_feats.shape[0], 1))

    for i in range(0, ion_engs.shape[0]):
        ion_eng = element(i + 1).ionenergies
        if 1 in ion_eng:
            ion_engs[i, 0] = element(i + 1).ionenergies[1]
        else:
            ion_engs[i, 0] = 0

    elem_feats = numpy.hstack([elem_feats, ion_engs])

    return preprocessing.scale(elem_feats)


def load_dataset(dataset_file_name, comp_idx, target_idx):
    data = numpy.array(pandas.read_excel(dataset_file_name))
    comps = data[:, comp_idx]
    targets = data[:, target_idx].reshape(-1, 1)
    mat_feats = list()
    elem_feats = get_elem_feats()

    for i in tqdm(range(0, comps.shape[0])):
        mat_feats.append(calc_mat_feat(elem_feats, comps[i]))

    return numpy.hstack([numpy.vstack(mat_feats), targets])


def calc_mat_feat(elem_feats, comp):
    elems = ast.literal_eval(str(parse_formula(comp)))
    e_sum = numpy.sum([float(elems[key]) for key in elems])
    w_sum_vec = numpy.zeros(elem_feats.shape[1])
    atom_feats = list()

    for e in elems:
        atom_vec = elem_feats[atom_nums[e] - 1, :]
        atom_feats.append(atom_vec)
        w_sum_vec += (float(elems[e]) / e_sum) * atom_vec

    return numpy.hstack([w_sum_vec, numpy.std(atom_feats, axis=0), numpy.min(atom_feats, axis=0), numpy.max(atom_feats, axis=0)])


def parse_formula(comp):
    elem_dict = dict()
    elem = comp[0]
    num = ''

    for i in range(1, len(comp)):
        if comp[i].islower():
            elem += comp[i]
        elif comp[i].isupper():
            if num == '':
                elem_dict[elem] = 1.0
            else:
                elem_dict[elem] = float(num)

            elem = comp[i]
            num = ''
        elif comp[i].isnumeric() or comp[i] == '.':
            num += comp[i]

        if i == len(comp) - 1:
            if num == '':
                elem_dict[elem] = 1.0
            else:
                elem_dict[elem] = float(num)

    return elem_dict


def rbf(x, mu, beta):
    return numpy.exp(-(x - mu)**2 / beta**2)


def load_dataset(path, id_target_file, idx_target, n_bond_feats, radius=5):
    elem_feats = get_elem_feats()
    list_cgs = list()
    id_target = numpy.array(pandas.read_excel(path + '/' + id_target_file))
    id_target = numpy.hstack([id_target, numpy.arange(id_target.shape[0]).reshape(-1, 1)])
    targets = id_target[:, idx_target]

    for i in tqdm(range(0, id_target.shape[0])):
        cg = read_cif(elem_feats, path, str(id_target[i, 0]), n_bond_feats, radius, targets[i])

        if cg is not None:
            cg.gid = len(list_cgs)
            list_cgs.append(cg)

    return list_cgs


def read_cif(elem_feats, path, m_id, n_bond_feats, radius, target):
    crys = Structure.from_file(path + '/' + m_id + '.cif')
    atom_feats = get_atom_feats(crys, elem_feats)
    bonds, bond_feats = get_bonds(crys, n_bond_feats, radius)

    if bonds is None:
        return None

    atom_feats = torch.tensor(atom_feats, dtype=torch.float).cuda()
    bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous().cuda()
    bond_feats = torch.tensor(bond_feats, dtype=torch.float).cuda()
    target = torch.tensor(target, dtype=torch.float).view(1, -1).cuda()
    gid = torch.tensor(-1, dtype=torch.long).cuda()

    return Data(x=atom_feats, y=target, edge_index=bonds, edge_attr=bond_feats, gid=gid)


def get_atom_feats(crys, elem_feats):
    atoms = crys.atomic_numbers
    atom_feats = list()

    for i in range(0, len(atoms)):
        atom_feats.append(elem_feats[atoms[i] - 1, :])

    return numpy.vstack(atom_feats)


def get_bonds(crys, n_bond_feats, radius):
    rbf_means = numpy.linspace(start=1.0, stop=radius, num=n_bond_feats)
    list_nbrs = crys.get_all_neighbors(radius, include_index=True)
    bonds = list()
    bond_feats = list()

    for i in range(0, len(list_nbrs)):
        nbrs = list_nbrs[i]

        for j in range(0, len(nbrs)):
            bonds.append([i, nbrs[j][2]])
            bond_feats.append(rbf(numpy.full(n_bond_feats, nbrs[j][1]), rbf_means, beta=0.5))

    if len(bonds) == 0:
        return None, None

    return numpy.vstack(bonds), numpy.vstack(bond_feats)
