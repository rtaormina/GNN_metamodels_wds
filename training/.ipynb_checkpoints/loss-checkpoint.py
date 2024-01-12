# Loss functions

# Libraries
import torch
from torch_geometric.utils import get_laplacian, to_dense_adj

def smooth_loss(preds, batch, alpha=0):
    # MSE loss + smoothness term evaluated as x^T * L * x,
    # where x is the graph signal and L is the graph Laplacian
    mask = batch.x[:,2]!=1
    mask_preds = preds[mask]

    # new_edge_index, new_edge_attr = get_laplacian(batch.edge_index, abs(model.pre_mp_edges(batch.edge_attr))[:,0], normalization='sym')
    new_edge_index, new_edge_attr = get_laplacian(batch.edge_index, batch.edge_attr[:,0], normalization='sym')
    Laplacian = to_dense_adj(new_edge_index, edge_attr=new_edge_attr)[0]
    smoothness = torch.matmul(mask_preds.t(), torch.matmul(Laplacian[mask][:, mask], mask_preds))

    loss = torch.mean((preds - batch.y)**2) + alpha*smoothness[0,0]
    
    return loss


# mask1 = torch.isin(batch.edge_index[0], (torch.where(batch.x[:,2]==1)[0]), invert=True)
# mask2 = torch.isin(batch.edge_index[1], (torch.where(batch.x[:,2]==1)[0]), invert=True)
# mask = torch.logical_and(mask1, mask2)

# edge_index_ns = torch.stack((batch.edge_index[0][mask], batch.edge_index[1][mask]))
# edge_attr_ns = batch.edge_attr[mask][:,0]