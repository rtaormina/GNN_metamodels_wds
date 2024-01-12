import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import wntr
import networkx as nx
from tqdm import tqdm
import math
from scipy import stats
import torch
from torch_geometric.utils import convert
import os
from pathlib import Path


def load_water_network(inp_file):
    '''
    This function loads a water network inputfile (.inp) and returns a WNTR WaterNetworkModel object.
    ------
    inp_file: .inp file
        file with information of wdn
    '''
    return wntr.network.WaterNetworkModel(inp_file)


def run_wntr_simulation(wn, headloss='H-W'):
    '''
    This function runs a simulation after changing the hydraulic options.
    ------
    wn: WNTR object
    headloss: str
        options: 'H-W'= Hazen-Williams, 'D-W'=Darcy-Weisbach
    '''
    wn.options.hydraulic.viscosity = 1.0
    wn.options.hydraulic.specific_gravity = 1.0
    wn.options.hydraulic.demand_multiplier = 1.0
    wn.options.hydraulic.demand_model = 'PDD'
    wn.options.hydraulic.minimum_pressure = 0
    wn.options.hydraulic.required_pressure = 1
    wn.options.hydraulic.pressure_exponent = 0.5
    wn.options.hydraulic.headloss = headloss
    wn.options.hydraulic.trials = 50
    wn.options.hydraulic.accuracy = 0.001
    wn.options.hydraulic.unbalanced = 'CONTINUE'
    wn.options.hydraulic.unbalanced_value = 10
    wn.options.hydraulic.checkfreq = 2
    wn.options.hydraulic.maxcheck = 10
    wn.options.hydraulic.damplimit = 0.0
    wn.options.hydraulic.headerror = 0.0
    wn.options.hydraulic.flowchange = 0.0
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim(version=2.2)

    return results


def get_attribute_all_nodes(wn, attr_str):
    '''
    This function retrieves an attribute (e.g., base demand) from all nodes in the network.

    output: a pandas Series indexed by node_id and containing the attribute as values.
    ------
    wn: WNTR object
    attr_str: str
        name of the selected attribute e.g., 'base_demand', 'elevation'
    '''
    temp = {}

    for id in wn.node_name_list:
        node = wn.get_node(id)
        try:
            attr = getattr(node, attr_str)
        except AttributeError:
            # e.g., tanks/reservoirs have no base demand
            attr = np.nan
        temp[id] = attr

    return pd.Series(temp)


def get_attribute_all_links(wn, attr_str):
    '''
    This function retrieves an attribute (e.g., diameter) from all links in the network.

    output: a pandas Series indexed by link_id and containing the attribute as values.
    ------
    wn: WNTR object
    attr_str: str
        name of the selected attribute e.g., 'diameter', 'length', 'roughness'
    '''
    temp = {}
    for id in wn.link_name_list:
        link = wn.get_link(id)
        try:
            attr = getattr(link, attr_str)
        except AttributeError:
            # e.g., pumps have no roughness
            attr = np.nan
        temp[id] = attr
    return pd.Series(temp)


def get_attribute_from_networks(attr_str, wn_path, wn_list, plot=True, n_cols=5):
    '''
    This function retrieves and plots the distribution of attribute attr_str across all networks in wn_list.
    The attribute can be either from edges or nodes. At most n_cols histograms are displayed per each row.

    Output: a dictionary with the retrieved attributes across all networks.
    ------
    attr_str: str
        name of the selected attribute e.g., 'diameter', 'length', 'base_demand', 'roughness'
    wn_path: str
        path to the network folder location
    wn_list: list
        names of the networks considered
    plot: bool
        if True, plot the distribution of attr_str for all considered networks
    n_cols: int
        number of plots displayed per each row
    '''
    d_attr = {}

    for network in wn_list:
        inp_file = f'{wn_path}/{network}.inp'
        wn = load_water_network(inp_file)
        # check if attr_str is node attribute
        s_attr = get_attribute_all_nodes(wn, attr_str)
        if s_attr.isnull().all() == True:
            # no? is it a link attribute?
            s_attr = get_attribute_all_links(wn, attr_str)
        if s_attr.isnull().all() == True:
            # no? then the attribute doesn't exist
            raise AttributeError(f'Attribute {attr_str} not existing.')
        d_attr[network] = s_attr

    # plot
    if plot == True:
        n_networks = len(wn_list)
        n_cols = min(n_cols, n_networks)
        n_rows = math.ceil(n_networks / n_cols)
        f, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        for ax, network in zip(axes.reshape(-1), d_attr.keys()):
            d_attr[network].hist(ax=ax)
            ax.set_title(network)
        # adjust overall figure
        f.suptitle(attr_str, fontsize=18, color='r')
        f.tight_layout(rect=[0, 0.05, 1, 0.95])  # rect takes into account suptitle

    return d_attr


def get_number_of_components_from_network(wn):
    '''
    This function returns a dictionary with number of links and nodes for a WNTR network model.
    The following total counts are returned:
    Nodes: overall, junctions, reservoirs, storage tanks
    Links: overall, pipes, valves, pumps
    '''
    return {'nodes': wn.num_nodes, 'junctions': wn.num_junctions, 'reservoirs': wn.num_reservoirs,
            'tanks': wn.num_tanks,
            'links': wn.num_links, 'pipes': wn.num_pipes, 'valves': wn.num_valves, 'pumps': wn.num_pumps, }


# get number of components for each network
def get_wdn_components(networks, path):
    '''
    Returns a dataframe with number of network components for each wdn in networks
    ------
    networks: list
        list of wdn names
    path: str
        path to the folder with .inp of the networks
    '''
    for ix, network in enumerate(networks):
        inp_file = f'{path}/{network}.inp'
        wn = load_water_network(inp_file)
        d_counts = get_number_of_components_from_network(wn)
        if ix == 0:
            # create dataframe
            df_counts = pd.DataFrame(index=networks, columns=d_counts.keys())
        df_counts.loc[network, :] = d_counts

    return df_counts


# options for each network
def get_wdn_options(networks, path):
    '''
    Returns a dataframe with hydraulic options for each wdn in networks
    ------
    networks: list
        list of wdn names
    path: str
        path to the folder with .inp of the network
    '''
    df_options = pd.DataFrame(index=networks, columns=['headloss', 'inpfile_units'])
    for network in networks:
        inp_file = f'{path}/{network}.inp'
        wn = load_water_network(inp_file)
        df_options.loc[network, :] = (wn.options.hydraulic.headloss, wn.options.hydraulic.inpfile_units)

    return df_options


def alter_water_network(wn, d_attr, d_netw):
    '''
    This function randomly modifies nodes and edges attributes in the water network according to the distributions in d_attr.
    At the moment, these are expressed as arrays containing all possible values. No changes are made if d_attr=None.
    No changes are made to a particular attribute if it is not in the keys of d_attr.
    '''
    for attr in d_attr.keys():
        if attr in ['base_demand']:
            set_attribute_all_nodes_rand(wn, attr, d_attr[attr]['values'], search_range=d_netw['range_bdmnd'],
                                         multiplier=d_netw['dmnd_mlt'])
        elif attr in ['roughness']:
            set_attribute_all_links_rand(wn, attr, d_attr[attr]['values'], search_range=d_netw['range_rough'])
        elif attr in ['diameter']:
            set_attribute_all_links_rand(wn, attr, d_attr[attr]['values'], search_range=d_netw['range_diams'],
                                         prob_exp=d_netw['prob_exp'])
        else:
            raise AttributeError('This attribute cannot be randomly assigned.')
    return None


def set_attribute_all_nodes_rand(wn, attr_str, attr_values, search_range, prob_exp=0, multiplier=1):
    '''
    This function changes an attribute attr_str (e.g., base demand) from all nodes in the network based on their orignal value.
    The list of potential values is contained in attr_values. search_range identifies how many values to the left and to the right
    of the original value are considered for the random selection.

    Tested for: base_demand
    '''
    LPS_TO_GPM = 1

    if attr_str not in ['base_demand']:
        raise AttributeError('You can only change base_demand as node attribute.')
    units = wn.options.hydraulic.inpfile_units
    for id in wn.node_name_list:
        node = wn.get_node(id)
        if hasattr(node, 'base_demand'):
            attr = node.base_demand * multiplier
            if units == 'GPM':
                node.demand_timeseries_list[0].base_value = select_random_value(attr * LPS_TO_GPM, attr_values,
                                                                                search_range, prob_exp)
            else:
                node.demand_timeseries_list[0].base_value = select_random_value(attr, attr_values, search_range,
                                                                                prob_exp)
    return None


def set_attribute_all_links_rand(wn, attr_str, attr_values, search_range, prob_exp=0):
    """
    This function changes an attribute attr_str (e.g., roughness) from all links in the network based on their orignal value.
    The list of potential values is contained in attr_values. search_range identifies how many values to the left and to the right
    of the original value are considered for the random selection.

    Tested for: diameter, roughness
    """
    # IN_TO_M = 0.0254

    if attr_str not in ['diameter', 'roughness']:
        raise AttributeError('You can only change pipe roughness and diameter as link attributes.')
    for id in wn.link_name_list:
        link = wn.get_link(id)
        if hasattr(link, attr_str):
            attr = getattr(link, attr_str)
            setattr(link, attr_str, select_random_value(attr, attr_values, search_range, prob_exp))
    return None


def select_random_value(current_value, possible_values, search_range, prob_exp=0):
    '''
    This function randomly selects a value in a list, centered on the current value.

    prob_exp is the exponent (0=uniform, 1=linear, 2=quadratic,...) use to nudge np.random.choice selection
    towards higher possible_values (prob_exp >= 1); this is useful for diameter (to obtain pressure distribution closer to that of original map).
    '''
    if search_range == 0:
        return (current_value)

    if all(possible_values[i] < possible_values[i + 1] for i in range(len(possible_values) - 1)) == False:
        raise ValueError('List of possible values is not sorted or contains duplicates.')

    if current_value >= possible_values[-1]:
        max_pos = len(possible_values)
        min_pos = max(max_pos - search_range, 0)
    else:
        max_pos = min((np.array(possible_values) > current_value).argmax() + search_range, len(possible_values))
        min_pos = max((np.array(possible_values) > current_value).argmax() - search_range - 1, 0)

    # assign probabilities
    if prob_exp > 0:
        len_seg = max_pos - min_pos
        prob = np.arange(1, len_seg + 1) ** prob_exp / sum(np.arange(1, len_seg + 1) ** prob_exp)
        return np.random.choice(possible_values[min_pos:max_pos], p=prob)

    return np.random.choice(possible_values[min_pos:max_pos])


def get_dataset_entry(network, d_attr, d_netw, path):
    '''
    This function creates a random input/output pair for a single network, after modifying it the original wds model.
    '''
    link_feats = ['roughness', 'diameter', 'length']
    node_feats = ['base_demand', 'node_type', 'elevation', 'base_head']
    res_dict = {}
    # load and alter network
    inp_file = f'{path}/{network}.inp'
    wn = load_water_network(inp_file)
    alter_water_network(wn, d_attr, d_netw)
    # retrieve input features
    for feat in link_feats:
        res_dict[feat] = get_attribute_all_links(wn, feat)
    for feat in node_feats:
        res_dict[feat] = get_attribute_all_nodes(wn, feat)
    # get output == pressure, after running simulation
    sim = run_wntr_simulation(wn, headloss='H-W')
    res_dict['pressure'] = sim.node['pressure'].squeeze()
    # check simulation
    ix = res_dict['node_type'][res_dict['node_type'] == 'Junction'].index.to_list()
    sim_check = ((res_dict['pressure'][ix] > 0).all()) & (sim.error_code == None)
    res_dict['network_name'] = network
    res_dict['network'] = wn

    return res_dict, sim, sim_check


def create_dataset(network, path, n_trials, d_attr, d_netw, max_fails=1e4, show=True):
    '''
    This function creates a dataset of n_trials length for a specific network
    '''
    n_fails = 0
    dataset = []

    if show == True:
        for i in tqdm(range(n_trials), network):
            flag = False
            while not flag:
                res_dict, _, flag = get_dataset_entry(network, d_attr, d_netw, path)
                if not flag:
                    n_fails += 1
                if n_fails >= max_fails:
                    raise RecursionError(f'Max number of fails ({max_fails}) reached.')
            dataset.append(res_dict)

    else:
        for sim_i in range(n_trials):
            flag = False
            while not flag:
                res_dict, _, flag = get_dataset_entry(network, d_attr, d_netw, path)
                if not flag:
                    n_fails += 1
                if n_fails >= max_fails:
                    raise RecursionError(f'Max number of fails ({max_fails}) reached.')
            dataset.append(res_dict)

    return dataset


def plot_dataset_attribute_distribution(dataset, attribute, figsize=(20, 5), bins=20):
    '''
    This function plots the overall distribution of a dataset property (e.g., nodes).
    Different colors are used for different networks within the dataset.
    '''
    df = pd.DataFrame(dataset)
    networks = df.network
    df_attr = pd.DataFrame()
    for network in networks.unique().tolist():
        net_attr = []
        for row in df[df.network == network][attribute]:
            net_attr += row.values.tolist()
            df_attr = pd.concat([df_attr, pd.Series(net_attr)], axis=1)
    df_attr.columns = networks

    # get limits
    min_x = df_attr.min().min()
    max_x = df_attr.max().max()

    # plot
    axes = df_attr.head(100).hist(figsize=figsize, bins=bins)
    for ax in axes.flatten():
        ax.set_xlim([min_x, max_x])
        plt.tight_layout()

    return df_attr


def from_wntr_to_nx(wn):
    '''
    This function converts a WNTR object to networkx
    '''
    wn_links = list(wn.links())
    wn_nodes = list(wn.nodes())

    G_WDS = wn.get_graph()  # directed multigraph
    uG_WDS = G_WDS.to_undirected()  # undirected
    sG_WDS = nx.Graph(uG_WDS)  # Simple graph

    i = 0
    for (u, v, wt) in sG_WDS.edges.data():
        assert isinstance(wn_links[i][1], wntr.network.elements.Pipe), "The link is not a pipe"
        sG_WDS[u][v]['name'] = wn_links[i][1].name
        sG_WDS[u][v]['diameter'] = wn_links[i][1].diameter
        sG_WDS[u][v]['length'] = wn_links[i][1].length
        sG_WDS[u][v]['roughness'] = wn_links[i][1].roughness
        i += 1

    i = 0
    for u in sG_WDS.nodes:
        # Junctions have elevation but no base_head and are identified with a 0
        if sG_WDS.nodes[u]['type'] == 'Junction':
            sG_WDS.nodes[u]['ID'] = wn_nodes[i][1].name
            sG_WDS.nodes[u]['type_1H'] = 0
            sG_WDS.nodes[u]['base_demand'] = list(wn_nodes[i][1].demand_timeseries_list)[0].base_value
            sG_WDS.nodes[u]['elevation'] = wn_nodes[i][1].elevation
            sG_WDS.nodes[u]['base_head'] = 0

        # Reservoirs have base_head but no elevation and are identified with a 1
        elif sG_WDS.nodes[u]['type'] == 'Reservoir':
            sG_WDS.nodes[u]['ID'] = wn_nodes[i][1].name
            sG_WDS.nodes[u]['type_1H'] = 1
            sG_WDS.nodes[u]['base_demand'] = 0
            sG_WDS.nodes[u]['elevation'] = 0
            sG_WDS.nodes[u]['base_head'] = wn_nodes[i][1].base_head
        else:
            print(u)
            raise Exception('Only Junctions and Reservoirs so far')
            break
        i += 1

    return sG_WDS  # df_nodes, df_links, sG_WDS


def convert_to_pyg(dataset):
    '''
    This function converts a list of simulations into a PyTorch Geometric Data type
    ------
    dataset: list
        list of network simulations, as given by create_dataset
    '''
    all_pyg_data = []

    for sample in dataset:
        wn = sample['network']
        # create PyG Data
        pyg_data = convert.from_networkx(from_wntr_to_nx(wn))

        # Add network name
        pyg_data.name = sample['network_name']
        # Add diamters for MLP
        pyg_data.diameters = torch.tensor(sample['diameter']).float()
        # Add simulaton results
        pyg_data.pressure = torch.tensor(sample['pressure'])

        # convert to float where needed
        pyg_data.base_demand = pyg_data.base_demand.float()
        pyg_data.diameter = pyg_data.diameter.float()
        pyg_data.roughness = pyg_data.roughness.float()

        all_pyg_data.append(pyg_data)

    return all_pyg_data


def save_database(database, names, size, out_path):
    '''
    This function saves the geometric database into a pickle file
    The name of the file is given by the used networks and the number of simulations
    ------
    database: list
        list of geometric datasets
    names: list
        list of the network names, possibly ordered by number of nodes
    size: int
        number of simulations per each network
    out_path: str
        output file location
    '''
    if isinstance(names, list):
        name = names + [str(size)]
        name = '_'.join(name)
    elif isinstance(names, str):
        name = names + '_' + str(size)

    Path(out_path).mkdir(parents=True, exist_ok=True)

    pickle.dump(database, open(os.path.join(f"{out_path}",f"{name}.p"), "wb"))

    return None


def create_and_save(networks, net_path, n_trials, d_attr, d_netw, out_path, max_fails=1e5, show=True):
    '''
    Creates and saves dataset given a list of networks and possible range of variable variations
    ------
    networks: list or str
        list or string of wdn names
    net_path: str
        path to the folder with .inp of the networks
    n_trials: int
        number of simulations
    d_attr: dict
        dictionary with values for each attribute
    d_newt: dict
        dictinary with ranges for each network
    out_path: str
        output file location
    max_fails: int
        number of maximum failed simulations per network
    show: bool
        if True, shows a bar progression for each simulation
    '''
    # create dataset
    all_data = []

    if isinstance(networks, list):
        for network in networks:
            all_data += create_dataset(network, net_path, n_trials, d_attr, d_netw[network], max_fails=max_fails,
                                       show=show)

    elif isinstance(networks, str):
        all_data += create_dataset(networks, net_path, n_trials, d_attr, d_netw[networks], max_fails=max_fails,
                                   show=show)

    # Create PyTorch Geometric dataset
    all_pyg_data = convert_to_pyg(all_data)

    # Save database
    save_database(all_pyg_data, names=networks, size=n_trials, out_path=out_path)

    return None


def import_config(config_file):
    '''
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    '''
    with open(config_file) as f:
        data = yaml.safe_load(f)
        networks = data['dataset_names']
        n_trials = data['n_trials']

        # dictionary with values for each attribute
        d_attr = {}
        for ranges in ['diameter_range', 'roughness_range', 'base_demand_range']:
            name = ranges.replace('_range', '')
            d_attr[name] = {'values': np.arange(data[ranges][0], data[ranges][1], data[ranges][2])[1:]}

        # dictinary with ranges for each network
        d_netw = {}
        for network in networks:
            d_netw[network] = {}
            d_netw[network]['range_diams'] = data[network][0]
            d_netw[network]['range_rough'] = data[network][1]
            d_netw[network]['range_bdmnd'] = data[network][2]
            d_netw[network]['prob_exp'] = data[network][3]
            d_netw[network]['dmnd_mlt'] = data[network][4]

    return networks, n_trials, d_attr, d_netw