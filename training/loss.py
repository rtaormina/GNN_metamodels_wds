# Loss functions

# Libraries
import torch
import torch_geometric
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_sparse import SparseTensor


def smooth_loss(preds, batch, alpha=0):
    loss = torch.mean((preds - batch.y)**2)

    if alpha > 0:
        loss += alpha*smoothness(preds, batch)
            
    return loss

def smoothness(preds, graph):
    '''
    Calculates smoothness term, derived as Y.T * L * Y,
    where Y is the output graph signal and L is the graph Laplacian
    ------
    preds: torch.tensor
        predictions
    graph: contains edge_index used for L
    '''
    new_edge_index, new_edge_attr = get_laplacian(graph.edge_index, normalization='sym')

    if isinstance(graph, torch_geometric.loader.dataloader.Batch):      
        Laplacian = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_attr)
        smoothness = 0
        for i in range(graph.num_graphs):
            pred_i = preds[graph.ptr[i]: graph.ptr[i+1]]
            L = Laplacian[graph.ptr[i]:graph.ptr[i+1], graph.ptr[i]:graph.ptr[i+1]]
            smoothness += pred_i.T.matmul(L.matmul(pred_i)).squeeze()
            print(smoothness)

    elif isinstance(graph, torch_geometric.data.data.Data):
        Laplacian = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_attr)
        smoothness = preds.T.matmul(Laplacian.matmul(preds)).squeeze()
            
    return smoothness
    