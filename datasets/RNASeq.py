"""
Adjusted from script for Set-MNIST dataset
"""
import os
import numpy as np
import scanpy as sc

import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from datetime import datetime

from .data_augmentations import DropCellsBernoulli, DropCellsFraction, RandomManifoldNoise, DropCellsFixedNumber, DropCellsBlop
from torchvision.transforms import Compose

def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)

def build(args):
    """
    Build torch datasets and dataloaders for the RNASeq dataset
    valfold and testfold are used to specify the fold id (in args.fold_colname) used for validation and test sets
    default is to use the first fold for val and the second fold for test
    """
    full_adata = sc.read_h5ad(args.h5ad_loc)

    print("trainfolds: ", list(np.setdiff1d(full_adata.obs[args.fold_col].unique(), [args.valfold, args.testfold])))
    print("valfolds: ", [args.valfold])
    print("testfolds: ", [args.testfold])

    now = datetime.now()
    print("creating CellBagDatasets...")
    train_dataset = CellBagDataset(adata=full_adata, pid_col=args.pid_col, target_col=args.target_col, fold_col=args.fold_col, adata_layer=args.adata_layer, split_ids=list(np.setdiff1d(full_adata.obs[args.fold_col].unique(), [args.valfold, args.testfold])), latent_idx_tokeep=args.latent_idx_tokeep)
    val_dataset = CellBagDataset(adata=full_adata, pid_col=args.pid_col, target_col=args.target_col, fold_col=args.fold_col, adata_layer=args.adata_layer, split_ids=[args.valfold], latent_idx_tokeep=args.latent_idx_tokeep)
    test_dataset = CellBagDataset(adata=full_adata, pid_col=args.pid_col, target_col=args.target_col, fold_col=args.fold_col, adata_layer=args.adata_layer, split_ids=[args.testfold], latent_idx_tokeep=args.latent_idx_tokeep)
    print("...took ", datetime.now()-now)

    now = datetime.now()
    print("creating DataLoaders...")
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=False, num_workers=args.num_workers,
                              collate_fn=train_dataset.collate_fn, worker_init_fn=init_np_seed)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, drop_last=False, num_workers=args.num_workers,
                            collate_fn=val_dataset.collate_fn, worker_init_fn=init_np_seed)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, drop_last=False, num_workers=args.num_workers,
                            collate_fn=test_dataset.collate_fn, worker_init_fn=init_np_seed)
    print("...took ", datetime.now()-now)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

class CellBagDataset(Dataset):

    def __init__(self, adata, pid_col, target_col, fold_col, adata_layer, split_ids=[0], max_num_cells=None, latent_idx_tokeep=None):

        # encode the targets -- need to do this before splitting data to ensure consistency
        target_classes, adata.obs['targets'] = np.unique(adata.obs[target_col], return_inverse=True)
        self.num_groups = len(target_classes) #number of groups in the full adata, not only in the split for this specific dataset
        self.max_num_cells = max_num_cells

        # filter the dataset for the split
        adata = adata[adata.obs[fold_col].isin(split_ids)]

        # establish an order of pids (I think this will be faster than individual list comprehensions)
        obs = adata.obs[[pid_col, 'targets', fold_col]].drop_duplicates().sort_values(pid_col) #'dataset'
        self.pids = obs[pid_col].values
        self.targets = obs['targets'].values
        self.fold = obs[fold_col].values
        #self.orig_datasets = obs['dataset'].values

        # inverse class frequency weighting
        _, self.target_counts = np.unique(self.targets, return_counts=True)
        self.class_weights = 1.0 / self.target_counts

        if adata_layer == "default":
            self.bags = [torch.from_numpy(adata.X[adata.obs[pid_col] == pid]) for pid in self.pids]
        elif adata_layer == "hvg_lognorm":
            self.bags = [torch.from_numpy(adata.layers['lognorm'][adata.obs[pid_col] == pid][:,adata.var.highly_variable].todense()) for pid in self.pids]
        elif adata_layer == "hvg_raw":
            raise NotImplementedError
        else:
            try:
                if latent_idx_tokeep is None:
                    self.bags = [torch.from_numpy(adata.obsm[adata_layer][adata.obs[pid_col] == pid]) for pid in self.pids]
                else:
                    self.bags = [torch.from_numpy(adata.obsm[adata_layer][adata.obs[pid_col] == pid][:,latent_idx_tokeep]) for pid in self.pids]
            except KeyError:
                raise KeyError(f"adata_layer {adata_layer} not found in adata.obsm")

    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        sample = {
            'idx' : idx,
            'pid' : self.pids[idx],
            'set' : self.bags[idx],
            'target' : self.targets[idx],
            'fold' : self.fold[idx],
            #'orig_dataset' : self.orig_datasets[idx],
        }
        if self.max_num_cells is not None:
            if sample['set'].shape[0] > self.max_num_cells:
                perm = torch.randperm(sample['set'].shape[0])
                sample['set'] = sample['set'][perm[:self.max_num_cells]]
        return sample
    
    def get_all_bags(self):
        return self.bags
    
    def get_all_labels(self):
        return self.targets
    
    # def get_all_orig_datasets(self):
    #     return self.orig_datasets

    def get_all_folds(self):
        return self.fold
    
    def collate_fn(self, batch):

        # pad bags and masks to have the same number of cells
        max_cells_per_sample = max([b['set'].shape[0] for b in batch])

        for b in batch:
            b['set_mask'] = torch.zeros(b['set'].shape[0], dtype=torch.bool)
            b['set_mask'] = torch.cat([b['set_mask'], torch.ones(max_cells_per_sample - b['set_mask'].shape[0], dtype=torch.bool)], dim=0)
            b['set'] = torch.cat([b['set'], torch.zeros(max_cells_per_sample - b['set'].shape[0], b['set'].shape[1])], dim=0)

        ret = dict()
        for k, v in batch[0].items():
            ret.update({k: [b[k] for b in batch]})

        s = torch.stack(ret['set'], dim=0)  # [B, N, dim_input]
        mask = torch.stack(ret['set_mask'], dim=0).bool()  # [B, N]
        cardinality = (~mask).long().sum(dim=-1)  # [B,]

        ret.update({'set': s, 'set_mask': mask, 'cardinality': cardinality, 'target':torch.tensor(ret['target'])})
        return ret


class ContrastiveDataset(CellBagDataset):

    def __init__(self, adata, pid_col, target_col, fold_col, adata_layer, split_ids=[0]):
        super().__init__(adata, pid_col, target_col, fold_col, adata_layer, split_ids)

        self.cell_dropout = DropCellsFixedNumber(n_keep=100)
        

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        original_set = sample['set'] 
        original_mask = sample['set_mask']

        bag, mask = self.cell_dropout(original_set, original_mask)
        bag2, mask2 = self.cell_dropout(original_set, original_mask)

        sample['set'] = bag
        sample['set_mask'] = mask
        sample['set2'] = bag2
        sample['set2_mask'] = mask2

        return sample

    def get_collate_fn(self):
        def collate_fn(batch):
            # pad bags and masks to have the same number of cells
            max_cells_per_sample = max([b['set'].shape[0] for b in batch]+[b['set2'].shape[0] for b in batch])
            for b in batch:
                b['set'] = torch.cat([b['set'], torch.zeros(max_cells_per_sample - b['set'].shape[0], b['set'].shape[1])], dim=0)
                b['set2'] = torch.cat([b['set2'], torch.zeros(max_cells_per_sample - b['set2'].shape[0], b['set2'].shape[1])], dim=0)
                b['set_mask'] = torch.cat([b['set_mask'], torch.ones(max_cells_per_sample - b['set_mask'].shape[0], dtype=torch.bool)], dim=0)
                b['set2_mask'] = torch.cat([b['set2_mask'], torch.ones(max_cells_per_sample - b['set2_mask'].shape[0], dtype=torch.bool)], dim=0)
            ret = dict()
            for k, v in batch[0].items():
                ret.update({k: [b[k] for b in batch]})
            s = torch.stack(ret['set'], dim=0)  # [B, N, input_dim]
            mask = torch.stack(ret['set_mask'], dim=0).bool()  # [B, N]
            s2 = torch.stack(ret['set2'], dim=0)
            mask2 = torch.stack(ret['set2_mask'], dim=0).bool()
            cardinality = (~mask).long().sum(dim=-1)  # [B,]
            ret.update({'set': s, 'set_mask': mask, 'set2': s2, 'set_mask2':mask2, 'cardinality': cardinality,
                        'mean': 0., 'std': 1.})
            return ret
        return collate_fn


class FakePatientDataset(CellBagDataset):

    def __init__(self, adata, pid_col, target_col, fold_col, adata_layer, split_ids=[0]):
        super().__init__(adata, pid_col, target_col, fold_col, adata_layer, split_ids=split_ids)
        self.augmentation = DropCellsBlop()        

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        # apply augmentation with 50% chance
        if torch.rand(1) > 0.5:
            sample['set'] = self.augmentation(sample['set'])
            sample['label'] = 1
        else:
            sample['label'] = 0

        return sample

    def collate_fn(self, batch):
        # pad bags and masks to have the same number of cells
        max_cells_per_sample = max([b['set'].shape[0] for b in batch])
        for b in batch:
            b['set'] = torch.cat([b['set'], torch.zeros(max_cells_per_sample - b['set'].shape[0], b['set'].shape[1])], dim=0)
            b['set_mask'] = torch.cat([b['set_mask'], torch.ones(max_cells_per_sample - b['set_mask'].shape[0], dtype=torch.bool)], dim=0)
        ret = dict()
        for k, v in batch[0].items():
            ret.update({k: [b[k] for b in batch]})
        s = torch.stack(ret['set'], dim=0)  # [B, N, input_dim]
        mask = torch.stack(ret['set_mask'], dim=0).bool()  # [B, N]
        cardinality = (~mask).long().sum(dim=-1)  # [B,]
        labels = torch.tensor([b['label'] for b in batch])
        ret.update({'set': s, 'set_mask': mask, 'cardinality': cardinality,
                    'mean': 0., 'std': 1., 'label': labels})
        return ret

class BaselinePatientEmbDataset(Dataset):
    """
    Dataset for baseline embeddings like cell type proportions
    """
    
    def __init__(self, embs, targets):
        self.embs = embs
        self.targets = targets

        _, self.target_counts = np.unique(self.targets, return_counts=True)
        self.class_weights = 1.0 / self.target_counts

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        return {
            'emb' : self.embs[idx],
            'target' : self.targets[idx],
        }

    def collate_fn(self, batch):
        ret = dict()
        for k, v in batch[0].items():
            ret.update({k: [b[k] for b in batch]})
        emb = torch.tensor(np.array(ret['emb']), dtype=torch.float32)  # [B, emb_dim]
        ret.update({'emb': emb, 'target': torch.tensor(ret['target']) })
        return ret