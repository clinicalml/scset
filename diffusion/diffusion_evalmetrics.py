import torch
import ot


def rbf_kernel(x, y, sigma=1.0):
    """
    Computes the RBF (Gaussian) kernel between two samples.
    x: tensor of shape (n, n_features)
    y: tensor of shape (m, n_features)
    sigma: float, kernel width
    Returns: tensor of shape (n, m)
    """
    xx = x.pow(2).sum(dim=1).unsqueeze(1)
    yy = y.pow(2).sum(dim=1).unsqueeze(0)
    xy = torch.mm(x, y.t())

    sqdist = xx + yy - 2 * xy
    return torch.exp(-sqdist / (2 * sigma**2))

def mmd(x, y, kernel='linear', sigma=1.0):
    """
    Compute the Maximum Mean Discrepancy test statistic between two samples.
    """
    if kernel == 'linear':
        diff = torch.mean(x, dim=0) - torch.mean(y, dim=0)
        return torch.dot(diff, diff)**0.5
    
    elif kernel == 'gaussian':
        
        xx = rbf_kernel(x, x, sigma=sigma)
        yy = rbf_kernel(y, y, sigma=sigma)
        xy = rbf_kernel(x, y, sigma=sigma)

        mmd = xx.mean() + yy.mean() - 2 * xy.mean()

        return mmd**0.5
    

import ot
def ot_sinkhorn(x, y, reg=1e-3):
    """
    Computes the entropy-regularized optimal transport distance between two samples.
    """

    cost = torch.cdist(x, y, p=2).pow(2)
    maxcost = cost.max()
    cost = cost / maxcost

    a = torch.ones(x.shape[0], device=x.device) / x.shape[0]
    b = torch.ones(y.shape[0], device=y.device) / y.shape[0]

    sq_dist = ot.sinkhorn2(a, b, cost, reg=reg)

    return (sq_dist * maxcost)**0.5

def ot_distance(x,y):

    cost = torch.cdist(x, y, p=2).pow(2)
    maxcost = cost.max()
    cost = cost / maxcost

    a = torch.ones(x.shape[0], device=x.device) / x.shape[0]
    b = torch.ones(y.shape[0], device=y.device) / y.shape[0]

    sq_dist = ot.emd2(a, b, cost)

    return (sq_dist * maxcost)**0.5

def compute_distance_matrix(x, y, dist_fn):
    """
    x : tensor of shape n x a x d
    y : tensor of shape m x b x d
    dist_fn : function that computes distance between two samples of shape a x d and b x d
    Returns: D tensor of shape n x m with D[i, j] = dist_fn(x[i], y[j])
    """
    n = x.shape[0]
    m = y.shape[0]
    D = torch.zeros(n, m, device=x.device)
    for i in range(n):
        for j in range(m):
            D[i, j] = dist_fn(x[i], y[j])
    return D


    
