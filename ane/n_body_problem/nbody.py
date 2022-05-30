import numpy
import torch
import pandas
import copy
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def get_data_loader(*data, batch_size, shuffle=False, dtype=torch.float):
    tensors = [torch.tensor(d, dtype=dtype) for d in data]
    return DataLoader(TensorDataset(*tuple(tensors)), batch_size=batch_size, shuffle=shuffle)


def load_dataset(path_dataset):
    pos_data = numpy.array(pandas.read_excel(path_dataset + '/n_body_pos.xlsx', header=None))
    vel_data = numpy.array(pandas.read_excel(path_dataset + '/n_body_vel.xlsx', header=None))
    dataset = get_dataset(pos_data, vel_data)

    return dataset


def get_dataset(pos_data, vel_data):
    states = list()
    next_vel = list()

    for i in range(0, pos_data.shape[0] - 1):
        states.append(numpy.hstack([pos_data[i, :], vel_data[i, :]]))
        next_vel.append(vel_data[i + 1, :])

    return numpy.vstack(states), numpy.vstack(next_vel)


def get_pos(pos_init, vel, dt):
    list_pos = list()
    pos = pos_init

    for i in range(0, vel.shape[0] - 1):
        pos += vel[i, :] * dt
        list_pos.append(copy.deepcopy(pos))

    return numpy.vstack(list_pos)


def getAcc(pos, mass, G, softening):
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0] ** (-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass
    az = G * (dz * inv_r3) @ mass

    # pack together the acceleration components
    a = numpy.hstack((ax, ay, az))

    return a


def getEnergy(pos, vel, mass, G):
    # Kinetic Energy:
    KE = 0.5 * numpy.sum(numpy.sum(mass * vel ** 2))

    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    z = pos[:, 2:3]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y
    dz = z.T - z

    # matrix that stores 1/r for all particle pairwise particle separations
    inv_r = numpy.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    inv_r[inv_r > 0] = 1.0 / inv_r[inv_r > 0]

    # sum over upper triangle, to count each interaction only once
    PE = G * numpy.sum(numpy.sum(numpy.triu(-(mass * mass.T) * inv_r, 1)))

    return KE, PE


def simulate(n_particles, path_dataset):
    N = n_particles  # Number of particles
    t = 0  # current time of the simulation
    tEnd = 100.0  # time at which simulation ends
    dt = 0.1  # timestep
    softening = 0.1  # softening length
    G = 1.0  # Newton's Gravitational Constant
    plotRealTime = True  # switch on for plotting as the simulation goes along

    list_vel = list()
    list_pos = list()
    list_acc = list()

    mass = numpy.full((N, 1), 0.1)
    pos = numpy.random.randn(N, 3) / 2
    vel = numpy.random.randn(N, 3) / 5

    # Convert to Center-of-Mass frame
    vel -= numpy.mean(mass * vel, 0) / numpy.mean(mass)

    # Calculate accelerations
    acc = getAcc(pos, mass, G, softening)

    # calculate initial energy of system
    KE, PE = getEnergy(pos, vel, mass, G)

    # number of timesteps
    Nt = int(numpy.ceil(tEnd / dt))

    # save energies, particle orbits for plotting trails
    pos_save = numpy.zeros((N, 3, Nt + 1))
    pos_save[:, :, 0] = pos
    KE_save = numpy.zeros(Nt + 1)
    KE_save[0] = KE
    PE_save = numpy.zeros(Nt + 1)
    PE_save[0] = PE
    t_all = numpy.arange(Nt + 1) * dt

    list_pos.append(copy.deepcopy(pos.reshape(1, -1)))
    list_vel.append(copy.deepcopy(vel.reshape(1, -1)))
    list_acc.append(copy.deepcopy(acc.reshape(1, -1)))

    # prep figure
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])

    # Simulation Main Loop
    for i in range(Nt):
        print(i, Nt)

        # update accelerations and velocity
        vel += acc * dt

        # drift
        pos += vel * dt

        acc = getAcc(pos, mass, G, softening)

        list_pos.append(copy.deepcopy(pos.reshape(1, -1)))
        list_vel.append(copy.deepcopy(vel.reshape(1, -1)))
        list_acc.append(copy.deepcopy(acc.reshape(1, -1)))

        # update time
        t += dt

        # get energy of system
        KE, PE = getEnergy(pos, vel, mass, G)

        # save energies, positions for plotting trail
        pos_save[:, :, i + 1] = pos
        KE_save[i + 1] = KE
        PE_save[i + 1] = PE

        # plot in real time
        if i % 10 == 0:
            if plotRealTime or (i == Nt - 1):
                plt.sca(ax1)
                plt.cla()
                xx = pos_save[:, 0, max(i - 50, 0):i + 1]
                yy = pos_save[:, 1, max(i - 50, 0):i + 1]
                plt.scatter(xx, yy, s=1, color=[.7, .7, 1])
                plt.scatter(pos[:, 0], pos[:, 1], s=10, color='blue')
                ax1.set(xlim=(-2, 2), ylim=(-2, 2))
                ax1.set_aspect('equal', 'box')
                ax1.set_xticks([-2, -1, 0, 1, 2])
                ax1.set_yticks([-2, -1, 0, 1, 2])

                plt.pause(0.001)

    pandas.DataFrame(numpy.vstack(list_pos)).to_excel(path_dataset + '/n_body_pos.xlsx', header=None, index=None)
    pandas.DataFrame(numpy.vstack(list_vel)).to_excel(path_dataset + '/n_body_vel.xlsx', header=None, index=None)
    pandas.DataFrame(numpy.vstack(list_acc)).to_excel(path_dataset + '/n_body_acc.xlsx', header=None, index=None)
