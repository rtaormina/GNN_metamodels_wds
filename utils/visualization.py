# Libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import wntr
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from training.test import testing


# Plot the distribution of the resiliences
def plot_res_distr(dataset):
    '''
    Plot the distribution of the resilience values
    '''
    ress = []
    n_graphs = len(dataset)

    for i in range(n_graphs):
        if dataset[i].res > 0:
            ress.append(dataset[i].res.double())
    ress = np.array(torch.stack(ress))

    plt.hist(ress, bins=20)
    plt.xlabel('Resilience index')
    plt.ylabel('Occurences')

    mean_res = np.round(ress.mean(), 4)

    print(f'Mean resilience: {mean_res}\t Positive values: {len(ress)}/{n_graphs}')


# Create plot for losses
def plot_loss(train_losses, val_losses=None, name=None, scale='log'):
    plt.plot(train_losses, 'b-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale(scale)
    if val_losses is not None:
        plt.plot(val_losses, 'r-')
        plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    if name is not None:
        plt.savefig(name)


# Create plot for R2
def plot_R2(model, loader, name=None, show=True, normalization=None):
    '''
    Calculate R^2 (coefficient of determination) between real and predicted pressures
    ------
    model: nn.Module
        GNN model
    loader: DataLoader
        batched list of several graphs to be evaluated by the model
    name: str
        if given, a figure is saved with that name
    normalization: dict
        used to normalize pressures by the maximum
    '''    
    _, pred, real = testing(model, loader, normalization=normalization)

    
    if show:
        if normalization is not None:
            pressure_max = normalization['pressure'].item()
            pred = pred * pressure_max
            real = real * pressure_max

        MIN = min(pred.min(), real.min())
        MAX = max(pred.max(), real.max())

        plt.scatter(pred, real, alpha=0.01)
        plt.plot([MIN, MAX], [MIN, MAX], 'k-')
        plt.xlim([0, MAX])
        plt.ylim([0, MAX])
        plt.title("Prediction vs Real")
        plt.xlabel('Predicted pressure [m]')
        plt.ylabel('Real pressure [m]');

        if name is not None:
            plt.savefig('results/' + name)

    return r2_score(real, pred)


def plot_pressure(database, normalization=None):
    '''
    Plot pressure distribution over entire dataset
    The model takes wither the whole database or its division in training, validation, and testing
    ------
    database: list
    normalization: dict
        contains the information to scale the pressures (e.g., graph_norm = {pressure:pressure_max})
    '''

    if len(database) == 3:
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle('Pressure distribution')
        titles = ["Training dataset", "Validation dataset", "Testing dataset"]

        for ax, dataset, title in zip(axs, database, titles):
            # Plot pressure distribution over entire dataset
            pressure = dataset[0].y.squeeze()

            for i in dataset[1:]:
                pressure = torch.cat((pressure, i.y.squeeze()), dim=0)

            if normalization is not None:
                pressure *= normalization['pressure']

            ax.hist(np.array(pressure), bins=100)
            ax.set_title(title)
            ax.set(xlabel='Pressure [m]', ylabel='Occurrences')
            # ax.label_outer()

    else:
        pressure = database[0].pressure.squeeze()

        for i in database[1:]:
            pressure = torch.cat((pressure, i.pressure.squeeze()), dim=0)

        plt.hist(np.array(pressure), bins=100)
        plt.title("Pressure distribution")
        plt.xlabel("Pressure [m]")
        plt.ylabel("Occurrences")
        plt.show()


# Spatial representation of pressure errors
def plot_diff(model, dataset, wdns, normalization=None, max_error=5):
    '''
    Plot pressure differences across a water distribution network (WDN)
    In case several graphs are given as inputs, the difference will be an averaged
    ------
    model: nn.Module
        GNN model
    dataset: Data or list
        Data: single graph object
        list: list of several graphs. It can encompass different WDNs
    wdns: dict
        keys: name of the WDN, generally expressed by three letters, e.g., 'PES'
        values: number of nodes in the WDN
    normalized: Boolean
        True: pressure values are normalized by the maximum pressure
    max_error: float
        represents the maximum error displayed in the plots
    '''
    if wdns is not None:
        # create dictionary for each water network
        graphs = dict.fromkeys(wdns)
        graphs_info = dict.fromkeys(wdns)

    if isinstance(dataset, Data):
        # Difference between real and predicted pressure
        label = list(wdns.keys())[0]
        graphs = {label: wdns[label]}
        graphs[label] = (dataset.y - model(dataset))

        # save graph information: position, edge_index, node_type
        graphs_info[label] = [dataset.pos]
        graphs_info[label].append(dataset.edge_index)
        graphs_info[label].append(dataset.x[:, 2])

    elif isinstance(dataset, list):
        # Average out all predictions of list of graphs
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        index = 0
        pred = []
        real = []

        for batch in loader:
            # calculate differencs in pressure for all graphs
            preds = model(batch)
            pred.append(model(batch).detach())
            real.append(batch.y)

            for i in range(batch.num_graphs):
                # iterate over graphs in a batch to diferrentiate between them
                mask = batch.batch == i
                real_i = batch.y[mask]
                pred_i = preds[mask]
                diff_i = (real_i - pred_i)
                info_i = torch.stack((diff_i, real_i, pred_i), dim=0)

                label = next(key for key, value in wdns.items() if value == diff_i.shape[0])
                try:
                    graphs[label] = torch.cat((graphs[label], info_i), dim=-1)
                except:
                    graphs[label] = info_i
                    # for the first dictionary creation, save graph information
                    # position, edge_index, node_type, colors for each graph
                    graphs_info[label] = [batch.pos[mask]]
                    mask_edge = ((batch.edge_index < (index + diff_i.shape[0])) & (batch.edge_index >= index))[0]
                    edge_index = batch.edge_index[:, mask_edge]
                    if edge_index.min() != 0:
                        edge_index -= edge_index.min()
                    graphs_info[label].append(edge_index)
                    graphs_info[label].append(batch.x[mask, 2])
                    # random not blue color 
                    graphs_info[label].append(np.hstack((np.random.rand(2), np.array(0))))

                index += diff_i.shape[0]

        if normalization is not None:
            pressure_max = normalization['pressure'].item()
            pred = torch.cat(pred) * pressure_max
            real = torch.cat(real) * pressure_max

    else:
        raise ("dataset type must be either Data or list")

    for label in graphs:
        # normalization
        if normalization is not None:
            pressure_max = normalization['pressure'].item()
            graphs[label] *= pressure_max

        # Plot results on the network    
        # Create graph
        G = nx.Graph()

        # Nodes
        G.add_nodes_from(range(graphs[label].shape[1]))

        # Edges
        edges = zip(graphs_info[label][1][1].tolist(), graphs_info[label][1][0].tolist())
        G.add_edges_from(edges)

        # Position
        pos = {}
        for i, (x, y) in enumerate(graphs_info[label][0]):
            pos[i] = (x.item(), y.item())

        storages = np.array(graphs_info[label][2])
        junctions = 1 - storages

        if isinstance(dataset, Data):
            fig = plt.figure(figsize=(7, 4))
            plt.title(label)
            axg = plt.gca()

        elif isinstance(dataset, list):
            # [axr, axg] = R^2, averaged graph
            fig, [axr, axg] = plt.subplots(ncols=2, figsize=(15, 4))
            fig.suptitle(label)

        # average pressure difference
        diffs = graphs[label][0].mean(-1).detach().numpy()

        # storage are represented with boxes
        for shape, nodelist, size in zip(["o", "s"], [junctions, storages], [30, 100]):
            nodes = nx.draw_networkx_nodes(G, pos, node_color=diffs[nodelist == 1], node_size=size, cmap='jet',
                                           vmin=-max_error, vmax=max_error, ax=axg,
                                           node_shape=shape,
                                           nodelist=[node for i, node in enumerate(G.nodes) if nodelist[i] == 1])

        edges = nx.draw_networkx_edges(G, pos, width=0.8, ax=axg)

        # labels = dict(zip(list(G.nodes), list(G.nodes)))
        # nx.draw_networkx_labels(G, pos, labels, font_size=20, ax=ax)

        axg.set_title('Average error')
        clb = fig.colorbar(nodes, ax=axg)
        clb.ax.set_title('Difference in \npressure [m]')

        if isinstance(dataset, list):
            ### Plot R2
            MIN = min(pred.min(), real.min())
            MAX = max(pred.max(), real.max())

            real_label = graphs[label][1].reshape(-1, 1).squeeze().detach().numpy()
            pred_label = graphs[label][2].reshape(-1, 1).squeeze().detach().numpy()

            axr.scatter(pred_label, real_label, color=graphs_info[label][3], alpha=0.2)
            axr.scatter(pred, real, alpha=0.01)
            axr.plot([MIN, MAX], [MIN, MAX], 'k-')
            axr.set(title="Correlation plot", xlabel='Predicted pressure [m]', ylabel='Real pressure [m]',
                    xlim=[0, MAX], ylim=[0, MAX])

        plt.show()
