import torch
import numpy as np

class DropCellsBernoulli():

    def __init__(self, p_drop=0.5):
        """
        Simple transformation which drops each instance in bag with probabily p_drop.
        """
        self.p_drop = p_drop

    def __call__(self, *elements):
        bag_size = elements[0].shape[0]
        drop_mask = torch.rand(bag_size) > self.p_drop
        if drop_mask.sum() == 0:
            drop_mask[torch.randint(0, bag_size, (1,))] = True
        return [element[drop_mask] for element in elements]

class DropCellsFraction():

    def __init__(self, fraction_drop=0.5):
        """
        Simple transformation which drops each instance in bag with probabily p_drop.
        """
        if fraction_drop is float:
            self.fraction = (fraction_drop, fraction_drop)
        else:
            self.fraction = fraction_drop

    def __call__(self, *elements):
        bag_size = elements[0].shape[0]
        rnd_fraction = self.fraction[0] + (self.fraction[1] - self.fraction[0]) * torch.rand(size=(1,))
        n_drop = int(bag_size * rnd_fraction)

        drop_mask = torch.ones(bag_size, dtype=torch.bool)
        drop_mask[:n_drop] = False
        drop_mask = drop_mask[torch.randperm(bag_size)]
        
        return [element[drop_mask] for element in elements]

class DropCellsFixedNumber():

    def __init__(self, n_keep=1):
        """
        Simple transformation which drops each instance in bag with probabily p_drop.
        """
        self.n_keep = n_keep

    def __call__(self, *elements):
        bag_size = elements[0].shape[0]
        n_drop = max(bag_size - self.n_keep, 0)

        drop_mask = torch.ones(bag_size, dtype=torch.bool)
        drop_mask[:n_drop] = False
        drop_mask = drop_mask[torch.randperm(bag_size)]
        
        return [element[drop_mask] for element in elements]


class DropCellsBlop():
    """
    Idea: drop cells in a blob-like fashion to shift mean statistics of sample.
    """
    
    def __init__(self, fraction_blop=0.3):
        self.fraction_blop = fraction_blop

    def __call__(self, cells, mask=None):
        bag_size = cells.shape[0]

        center_cell_id = torch.randint(0, bag_size, (1,))

        dist = torch.cdist(cells[center_cell_id].unsqueeze(0), cells)
        dist = dist.squeeze()

        sorted_neighbours = dist.argsort()
        n_keep = int(np.ceil(bag_size * (1-self.fraction_blop))) # keep 70% of cells
        keep = cells[sorted_neighbours[-n_keep:]]
        keep_mask = mask[sorted_neighbours[-n_keep:]]

        return keep, keep_mask

    
class RandomManifoldNoise():

    def __init__(self, k=3, alpha_max=0.5):
        """
        Augmentation which computes convex combination of k-nearest neighbors of each cell and adds a random fraction of it as noise to the cell.
        x' = (1 - alpha) * x + alpha * sum_{i=1}^{k} beta_i * x_i
        """
        self.k = k
        self.alpha_max = 0.5

    def __call__(self, cells):
        dist_matrix = torch.cdist(cells, cells)
        dist_matrix[torch.eye(cells.shape[0]).bool()] = 1e9
        _, topk_ind = torch.topk(dist_matrix, self.k, largest=False) # [N, k]

        neighbours = cells[topk_ind] # [N, k, input_dim]

        convex_coeff = torch.from_numpy(np.random.dirichlet(np.ones(self.k), size=cells.shape[0])).to(cells.device) # [N, k]
        convex_coeff = convex_coeff.unsqueeze(-1).float()
        
        noise = (neighbours * convex_coeff).sum(dim=1) # [N, input_dim]
        alpha = torch.rand(size=(cells.shape[0], 1)) * self.alpha_max

        return (1 - alpha) * cells + alpha * noise
    
