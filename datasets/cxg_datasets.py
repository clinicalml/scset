import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from anndata import AnnData
from .data_augmentations import DropCellsFixedNumber, DropCellsBernoulli, DropCellsBlop, DropCellsFraction
from tqdm import tqdm
import copy

class PatientDataset(Dataset):

    def __init__(self, folder, pretrain_embedding, min_num_cells=100, preload=False, split_col="random_fold", split_ids=None, patient_csv='patient_index.csv', debug=False, annots_to_load=None, annots_dir='annotations', max_num_cells=10000, latent_idx_tokeep=None, tissue_filter=None, sizefactor_maxfrac=1):
        """
        annots_to_load: list of feature column names to pull from the cell annotation file
        """
        self.folder = folder
        self.pretrain_embedding = pretrain_embedding
        self.patient_df = pd.read_csv(os.path.join(folder, patient_csv), index_col=False)
        self.preload = preload
        self.min_num_cells = min_num_cells
        self.annots_to_load = annots_to_load
        self.annots_dir = annots_dir
        self.max_num_cells = max_num_cells
        self.latent_idx_tokeep = latent_idx_tokeep

        if split_ids is not None:
            self.patient_df = self.patient_df[self.patient_df[split_col].isin(split_ids)]

        # used in some post-hoc analysis when I only need to load a subset of the patients based on tissue
        if tissue_filter is not None:
            self.patient_df = self.patient_df[self.patient_df.tissue_general.isin(tissue_filter)]

        self.patient_df = self.patient_df[self.patient_df.num_cells >= min_num_cells]
        print(f"Number of patients with at least {min_num_cells} cells: {len(self.patient_df)}")

        self.ncells_pp = self.patient_df.num_cells.values
      
        if self.preload:
            print("Preloading patients...")
            self.patients = dict()
            ctr=0
            if debug:
                print("DEBUG MODE: Only loading 10 patients")
                self.patient_df = self.patient_df[:10]
            for id in tqdm(self.patient_df['patient_id']):
                patient_path = os.path.join(self.folder, self.pretrain_embedding, id + '.npy')
                if self.latent_idx_tokeep is None:
                    self.patients[id] = torch.from_numpy(np.load(patient_path))
                else:
                    self.patients[id] = torch.from_numpy(np.load(patient_path)[:, self.latent_idx_tokeep])
                #check for nans
                if torch.isnan(self.patients[id]).any():
                    print(f"Patient {id} has nans. This is patient {ctr}.")
                if False: #WIP
                    #remove cells that are expression 3 or fewer genes
                    if (self.patients[id] == 0).sum()<=3:
                        print(f"removing {(self.patients[id] == 0).all(dim=1).sum()} cells that are all zero from pt index {index}")
                        self.patients[id] = self.patients[id][torch.any(self.patients[id] != 0, dim=1)]
                    # recheck there are min_num_cells after removing cells
                    if self.patients[id].shape[0] < min_num_cells:
                        print(f"Patient {id} has fewer than {min_num_cells} cells. Removing from dataset.")
                        self.patients.pop(id)
                        self.patient_df = self.patient_df[self.patient_df.patient_id != id]
                    # keep blacklist of genes too highly expressed for size factor calculation
                    if sizefactor_maxfrac is not None:
                        # for each cell (row) add genes (column ids) which account for > sizefactor_maxfrac of the total counts for that row
                        self.highlyexpressed_sizefactor_mask = torch.any(self.patients[id] > self.patients[id].sum(dim=1, keepdim=True)*sizefactor_maxfrac, dim=0) #shape genes
                        #self.genes_exclude_sizefactors.update(torch.where(torch.any(self.patients[id] > self.patients[id].sum(dim=1, keepdim=True)*sizefactor_maxfrac, dim=0))[0].tolist())

                ctr+=1

    def __getitem__(self, index):
        id = self.patient_df.iloc[index]['patient_id']
        if self.preload:
            cells = self.patients[id]
        else:
            patient_path = os.path.join(self.folder, self.pretrain_embedding, id + '.npy')
            if self.latent_idx_tokeep is None:
                cells = torch.from_numpy(np.load(patient_path))
            else:
                cells = torch.from_numpy(np.load(patient_path)[:, self.latent_idx_tokeep])

        subsampled_flag = 0
        if cells.shape[0] > self.max_num_cells:
            subsampled_flag = 1
            idx = torch.randperm(cells.shape[0])[:self.max_num_cells]
            cells = cells[idx]

        if self.annots_to_load is not None:
            annotation = os.path.join(self.folder, self.annots_dir, id + '.csv')
            annotations = pd.read_csv(annotation)
            
            if subsampled_flag:
                annotations = annotations.iloc[idx]
            return cells, annotations[self.annots_to_load] # annotations returns pd array
        return cells
    
    def collate_fn(self, batch):
        # pad bags and masks to have the same number of cells
        max_cells_per_sample = max([b.shape[0] for b in batch])
        pad = torch.zeros(len(batch), max_cells_per_sample, dtype=torch.bool)
        for i in range(len(batch)):
            batch[i] = torch.cat([batch[i], torch.zeros(max_cells_per_sample - batch[i].shape[0], batch[i].shape[1])], dim=0)
            pad[i, batch[i].shape[0]:] = 1
        ret = dict()
        ret['set'] = torch.stack(batch, dim=0)
        ret['set_mask'] = pad
        ret['cardinality'] = torch.tensor([b.shape[0] for b in batch])
        return ret
    
    def __len__(self):
        return len(self.patient_df)

    def load_anndata(self):
        cells = []
        annotations = []
        for id in self.patient_df['patient_id']:
            patient_path = os.path.join(self.folder, self.pretrain_embedding, id + '.npy')
            annotation = os.path.join(self.folder, 'annotations', id + '.csv')
            cells.append(np.load(patient_path))
            annotations.append(pd.read_csv(annotation))

        X = np.concatenate(cells, axis=0)
        obs = pd.concat(annotations, axis=0, ignore_index=True) 

        adata = AnnData(X=X)
        adata.obs = obs

        return adata

class AutoregPatientDataset(PatientDataset):

    def __init__(self, folder, pretrain_embedding, min_num_cells, max_num_cells, num_anchor_cells, num_target_cells, preload=False, split_col="random_fold", split_ids=None, patient_csv='patient_index.csv', debug=False, annots_to_load=None, annots_dir='annotations', latent_idx_tokeep=None, tissue_filter=None, target_libnorm=False, target_lognorm=False, sizefactor_maxfrac=1, highly_expressed_mask=None):
        super().__init__(folder, pretrain_embedding, min_num_cells=min_num_cells, preload=preload, split_col=split_col, split_ids=split_ids, patient_csv=patient_csv, debug=debug, annots_to_load=annots_to_load, annots_dir=annots_dir, max_num_cells=max_num_cells, latent_idx_tokeep=latent_idx_tokeep, tissue_filter=tissue_filter, sizefactor_maxfrac=sizefactor_maxfrac)
        self.num_anchor_cells = num_anchor_cells
        self.num_target_cells = num_target_cells
        if False: #WIP
            self.target_lognorm = target_lognorm
            self.target_libnorm = target_libnorm
            if highly_expressed_mask is None: #if no mask passed, use mask calculated on this dataset (want the option to pass e.g. mask calculated on training data)
                self.highly_expressed_mask = self.highlyexpressed_sizefactor_mask

    def __getitem__(self, index):
        if self.annots_to_load is None: #this is the default behavior, and how it's used in the training script
            cells = super().__getitem__(index)
        else: #this is used in post-hoc analysis
            cells, annotations = super().__getitem__(index)
        if self.num_anchor_cells is None:
            # sample as many anchor cells as possible but not more than max_num_cells due to memory constraints
            num_cells = self.max_num_cells
            idx = torch.randperm(cells.shape[0])[:num_cells]
        else:
            # sample num_anchor_cells anchor cells and num_target_cells target cells but not more than max_num_cells+num_target_cells due to memory constraints
            num_cells = min(self.max_num_cells + self.num_target_cells, self.num_anchor_cells + self.num_target_cells)
            idx = torch.randperm(cells.shape[0])[:num_cells]
        cells = cells[idx]

        target_cells = cells[:self.num_target_cells]

        if False:
            # for deconvolution tasks, where input/output is counts, may want target cells to be library- and log-normalized (anchor cells get transformed after they are encoded into pseudobulk)
            if self.target_libnorm:
                #X = X / X.sum(dim=1, keepdim=True)*1e4
                #print("shape target cells before size factor normalization", target_cells.shape)
                #print(f"n genes excluded from size factor calculation {torch.sum(self.highly_expressed_mask)}")
                size_factors = target_cells.masked_fill(self.highly_expressed_mask, 0).sum(dim=1, keepdim=True)
                if (size_factors == 0).any():
                    print("size factors are zero for some cells")
                    print(size_factors)
                #print(f"size factors: {size_factors}")
                target_cells = target_cells / size_factors*1e4
                #print("shape target cells after size factor normalization", target_cells.shape)

            if self.target_lognorm:
                #X = torch.log1p(X)
                target_cells = torch.log1p(target_cells)

        anchor_cells = cells[self.num_target_cells:]
        if self.annots_to_load is None:
            sample = {
                'anchor' : anchor_cells,
                'anchor_mask' : torch.zeros(anchor_cells.shape[0], dtype=torch.bool),
                'target' : target_cells
            }
        else:
            annotations = annotations.iloc[idx].reset_index(drop=True)
            annotations['target_or_anchor'] = ['target']*self.num_target_cells + ['anchor']*(cells.shape[0]-self.num_target_cells)   
            sample = {
                'anchor' : anchor_cells,
                'anchor_mask' : torch.zeros(anchor_cells.shape[0], dtype=torch.bool),
                'target' : target_cells,
                'annotations' : annotations
            }
        return sample
    
    def collate_fn(self, batch):
            # pad bags and masks to have the same number of cells
            max_cells_in_anchor = max([b['anchor'].shape[0] for b in batch])
            for b in batch:
                b['anchor'] = torch.cat([b['anchor'], torch.zeros(max_cells_in_anchor - b['anchor'].shape[0], b['anchor'].shape[1])], dim=0)
                b['anchor_mask'] = torch.cat([b['anchor_mask'], torch.ones(max_cells_in_anchor - b['anchor_mask'].shape[0], dtype=torch.bool)], dim=0)
            ret = dict()
            #for k, _ in batch[0].items():
            for k in ['anchor', 'anchor_mask', 'target']:
                ret[k] = torch.stack([b[k] for b in batch], dim=0)
            if 'annotations' in batch[0].keys():
                # annotations should be stacked into one large array, with additional column for index in batch
                annotations = [pd.concat((b['annotations'], pd.Series(np.repeat(index, b['annotations'].shape[0]), name="idx_in_batch")), axis=1) for index, b in enumerate(batch)]
                annotations = pd.concat(annotations, axis=0)
                ret['annotations'] = annotations
    
            return ret

class DeconvolutionPatientDataset(PatientDataset):

    def __init__(self, folder, pretrain_embedding, min_num_cells, max_num_cells, num_anchor_cells, num_target_cells, preload=False, split_col="random_fold", split_ids=None, patient_csv='patient_index.csv', debug=False, annots_to_load=None, annots_dir='annotations', latent_idx_tokeep=None, tissue_filter=None, train_or_test="train"):
        
        # load pre-saved lognorm+PCA pseudobulks as anchor embedding
        if train_or_test=="train":
            pbulk = pd.read_csv("/data/rna_rep_learning/scset/outputs/processed_data/bulk_pca_train.txt", index_col=0, sep=' ', header=None)
            self.anchor_cells = dict(zip(pbulk.index, pbulk.values.tolist()))

        if train_or_test=="test":
            pbulk = pd.read_csv("/data/rna_rep_learning/scset/outputs/processed_data/bulk_pca_test.txt", index_col=0, sep=' ', header=None)
            self.anchor_cells = dict(zip(pbulk.index, pbulk.values.tolist()))

        # use super().__init__ to load target cells in pretrain_embedding space
        super().__init__(folder, pretrain_embedding, min_num_cells=min_num_cells, preload=preload, split_col=split_col, split_ids=split_ids, patient_csv=patient_csv, debug=debug, annots_to_load=annots_to_load, annots_dir=annots_dir, max_num_cells=max_num_cells, latent_idx_tokeep=latent_idx_tokeep, tissue_filter=tissue_filter)
        self.target_cells = self.patients
        self.num_target_cells = num_target_cells

    def __getitem__(self, index):

        id = self.patient_df.iloc[index]['patient_id']
        anchor_cells = self.anchor_cells[id] #in this case, anchor cells are already pseudobulk vectors per patient
        target_cells = self.target_cells[id]
        #subset to random set of target cells
        idx = torch.randperm(target_cells.shape[0])[:self.num_target_cells]
        target_cells = target_cells[idx]

        ## return sample
        sample = {
            'anchor' : anchor_cells,
            #'anchor_mask' : torch.zeros(anchor_cells.shape[0], dtype=torch.bool),
            'target' : target_cells
        }

        return sample
    
    def collate_fn(self, batch):
            # in the deconv case, no need for padding
            ret = dict()
            for k in ['anchor', 'target']:
                ret[k] = torch.stack([b[k] for b in batch], dim=0)    
            return ret


class ContrastivePatientDataset(PatientDataset):

    def __init__(self, folder, pretrain_embedding, cells_per_view =100, preload=False, split_col="random_fold", split_ids=None, annots_dir='annotations', patient_csv='patient_index.csv', latent_idx_tokeep=None):
        super().__init__(folder, pretrain_embedding, min_num_cells=2*cells_per_view, preload=preload, split_col=split_col, split_ids=split_ids, patient_csv=patient_csv, annots_dir=annots_dir, latent_idx_tokeep=latent_idx_tokeep)
        self.cell_dropout = DropCellsFixedNumber(n_keep=cells_per_view)

    def __getitem__(self, idx):
        cells = super().__getitem__(idx)
        mask = torch.zeros(cells.shape[0], dtype=torch.bool)

        bag1, mask1 = self.cell_dropout(cells, mask)
        bag2, mask2 = self.cell_dropout(cells, mask)

        sample = {
            'set' : bag1,
            'set_mask' : mask1,
            'set2' : bag2,
            'set2_mask' : mask2
        }

        return sample

    def collate_fn(self, batch):
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


class CellSorterPatientDataset(PatientDataset):

    def __init__(self, folder, pretrain_embedding, preload=False, n_positives_per_sample=1, min_num_cells=100, max_num_cells=1000, split_col="random_fold", split_ids=None, patient_csv='patient_index.csv', annots_dir='annotations', debug=False, latent_idx_tokeep=None):
        """
        n_positives_per_sample: number of times each sample is used as a positive example (will also create a balanced number of negative examples)
        """
        super().__init__(folder, pretrain_embedding, preload=preload, min_num_cells=min_num_cells, split_col=split_col, split_ids=split_ids, patient_csv=patient_csv, debug=debug, max_num_cells=max_num_cells, annots_dir=annots_dir, latent_idx_tokeep=latent_idx_tokeep)
        self.n_positives_per_sample = n_positives_per_sample

    def __getitem__(self, idx):
        cells = super().__getitem__(idx)
        mask = torch.zeros(cells.shape[0], dtype=torch.bool)

        # repeat sample n_positives_per_sample*2 (to account for positives and negatives)
        targets = torch.zeros(self.n_positives_per_sample*2, dtype=torch.float)
        targets[:self.n_positives_per_sample] = 1.0

        # draw n_positives_per_sample cells from this sample
        positive_cell_idxs = torch.randint(0, cells.shape[0], (self.n_positives_per_sample,))
        pos_cells = cells[positive_cell_idxs]

        # draw the same number of negative cells from other random samples in the dataset
        negative_sample_idxs = torch.randint(0, len(self), (self.n_positives_per_sample,)).tolist()
        negative_cell_idxs = [torch.randint(0, self.ncells_pp[i], (1,)).item() for i in negative_sample_idxs]
        neg_cells = torch.stack([super().__getitem__(i)[cell_idx,:] for i, cell_idx in zip(negative_sample_idxs, negative_cell_idxs)], dim=0)

        cellstosort = torch.cat([pos_cells, neg_cells], dim=0)

        sample = {
            'set' : cells, #.repeat([self.n_positives_per_sample*2, 1]),
            'set_mask' : mask, #.repeat([self.n_positives_per_sample*2, 1]),
            'cellstosort' : cellstosort,
            'targets' : targets
        }

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
        cellstosort = torch.stack(ret['cellstosort'], dim=0) #[B, num_positives*2, input_dim]
        targets = torch.stack(ret['targets'], dim=0) #[B, num_positives*2]
        ret.update({'set': s, 'set_mask': mask, 'cellstosort': cellstosort, 'targets': targets})
        return ret