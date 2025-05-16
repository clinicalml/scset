import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sample_pt_cells_scvi(obs, var, scvi_model, cell_types, cell_type_dirichlet_conc, total_cells, ptname, groupname, avg_perturb_fc=4, perturb_prob=0.8, celltypesperturb=None, perturbed_gene_set=None, verbose=True):
    """
    sample synthetic cells for a single patient using scvi
    perturb a subset of cells if desired
    """
    
    # all cell types must be in metadata
    assert np.all(pd.Series(cell_types).isin(obs.celltype)), "not all cell types are found in adata.obs"
    
    sample_cell_type_probs = scipy.stats.dirichlet.rvs([cell_type_dirichlet_conc[c] for c in cell_types]).squeeze() #ensures cell_type_concs is indexed in same order as cell_types 
    cell_type_counts = dict(zip(cell_types, scipy.stats.multinomial.rvs(total_cells, sample_cell_type_probs)))
    
    ## get rid of cell types with zero counts
    cell_type_counts = dict([(c, cell_type_counts[c]) for c in cell_type_counts.keys() if cell_type_counts[c]!=0]) 
    cell_types = cell_type_counts.keys()

    ## sample B, T, mono in certain proportions
    #find corresponding inds in adata
    celltype_inds = dict([(c, np.arange(len(obs))[obs.celltype==c]) for c in cell_types])
    
    #sample
    chosen_inds = dict([(c, np.random.choice(celltype_inds[c], size = cell_type_counts[c])) for c in cell_types])

    #generate samples from posterior predictive dist
    samples = dict([(c, scvi_model.posterior_predictive_sample(indices=chosen_inds[c], n_samples=1)) for c in cell_types])

    joint_obs = pd.concat([obs.iloc[chosen_inds[c],:] for c in cell_types])[["patient","celltype","Tcellsubtype"]].reset_index().rename(columns={"index":"orig_cellbarcode", "patient":"patient_orig"})
    joint_obs["group"] = pd.Series(groupname).repeat(total_cells).reset_index(drop=True)
    joint_obs["patient"] = pd.Series(ptname).repeat(total_cells).reset_index(drop=True)

    ## induce fold change if desired ##

    # if perturb
    if perturbed_gene_set is not None: 
        # get list of genes to perturb
        intersect_perturbed_gene_set = np.intersect1d(perturbed_gene_set, var.index)
        if len(intersect_perturbed_gene_set)<len(perturbed_gene_set):
            if verbose:
                print("The following genes were not found in your data: ")
                print(np.setdiff1d(perturbed_gene_set, var.index))
                print("Perturbing {} genes.".format(len(intersect_perturbed_gene_set)))
        if not len(intersect_perturbed_gene_set)==0:         
            # sample FCs from normal distribution
            fc_vec = np.exp(scipy.stats.norm.rvs(np.log(avg_perturb_fc), 0.15, size=len(intersect_perturbed_gene_set))) #generate fold changes

            if celltypesperturb is None: #perturb all cells
                celltypesperturb = cell_types
            #perturb one cell type at a time
            else:
                for c in celltypesperturb:
                    if not c in samples:
                        print("this patient doesn't have any {} cells. skipping this cell type.".format(c))
                        continue
                    # save lib size of each cell, will need it to regenerate counts
                    #print(samples[c].shape)
                    libsizes = np.sum(samples[c], axis=1)
                    
                    # add noise to fold changes across cells
                    fc_mat = np.tile(fc_vec, (samples[c].shape[0], 1))
                    fc_mat += np.random.normal(0,0.5,fc_mat.shape)

                    ## only perturb approx 80% of the cancer genes; 20% remain unaffected at random
                    #fc_mat[:,np.random.rand(len(fc_vec))>perturb_prob] = 1

                    # multiply fold changes
                    full_fc_mat = np.ones((samples[c].shape[0], samples[c].shape[1]))
                    full_fc_mat[:, var.index.isin(intersect_perturbed_gene_set)] = fc_mat
                    samples[c] = np.multiply(samples[c], full_fc_mat)
    
                    #renormalize each cell & regenerate the counts from a poisson
                    samples[c] = np.divide(samples[c].T, samples[c].sum(axis=1)).T
                    #multiply gene rates by library size and sample counts from Poisson
                    samples[c] = np.multiply(samples[c].T, libsizes).T
                    samples[c] = scipy.stats.poisson.rvs(samples[c])
                    samples[c] = samples[c].reshape(-1, len(var))

    return np.concatenate(list(samples.values()), axis=0), joint_obs


def simulate_patient_counts(cell_types, cell_type_concs, total_cells, gene_params, loglibrary_mu, loglibrary_sigma, pt_id, grouplabel, groupval, avg_perturb_fc=4, perturb_prob=0.8, var_names=None, perturbed_gene_set=None, verbose=True):
    """
    cell_types: list of strings with labels for the cell types. Must be present in keys of gene_params and cell_type_concs. 
    cell_type_concs: dictionary of relative likelihood of different cell types, parameter of Dirichlet-Multinomial distribution for generating n for each cell type. Keys should match "cell_types."
    total_cells: the total cells for this patient (likely drawn from a Poisson outside of this function)
    gene_params: a dictionary of Gamma dist parameters per genes, for each cell type of interest. 
    """

    #everything will be pulled in the order of "cell_types" arg
    
    sample_cell_type_probs = scipy.stats.dirichlet.rvs([cell_type_concs[i] for i in cell_types]).squeeze() #ensures cell_type_concs is indexed in same order as cell_types 
    cell_type_counts = scipy.stats.multinomial.rvs(total_cells, sample_cell_type_probs)
    if verbose:
        print("cell type counts: {}".format(cell_type_counts))
    ngenes = len(gene_params[cell_types[0]])
    #simulate library sizes
    libsizes = np.exp(scipy.stats.norm.rvs(loc=loglibrary_mu, scale=loglibrary_sigma, size=total_cells))
    metadata = pd.DataFrame({"cell_type":np.repeat(cell_types, cell_type_counts), "target_libsize":libsizes, "patient":np.repeat(pt_id, total_cells), grouplabel:np.repeat(groupval, total_cells)})
    full_sample_matrix = np.empty((0, ngenes))
    for k in np.arange(len(cell_types)): #iterate over each cell type
        cell_type = cell_types[k]
        if cell_type not in gene_params.keys():
            raise("requested cell type {} not present in gene_params.keys()".format(cell_type))
        if verbose:    
            print("Generating {} for this sample...".format(cell_type))
        cell_by_gene_mat = np.zeros((cell_type_counts[k], ngenes))
        for n in np.arange(cell_type_counts[k]): #simulate one cell at a time            
            #samples all genes for one cell; genes that were 0 everywhere in training data are kept as 0
            cell_by_gene_mat[n,~gene_params[cell_type].iloc[:,0].isna()] = scipy.stats.gamma.rvs(a=gene_params[cell_type].loc[~gene_params[cell_type].iloc[:,0].isna()].a,
                                                                                                 loc=gene_params[cell_type].loc[~gene_params[cell_type].iloc[:,0].isna()].location,
                                                                                                 scale=gene_params[cell_type].loc[~gene_params[cell_type].iloc[:,0].isna()].scale)
            
        #concatenate this cell type expression to full matrix for sample
        full_sample_matrix = np.concatenate((full_sample_matrix, cell_by_gene_mat), axis=0)
    #set negative and very small numbers to 0
    tol = 1e-16
    full_sample_matrix[np.absolute(full_sample_matrix)<tol]=0
    full_sample_matrix[full_sample_matrix<0]=0
    
    # make DataFrame to add columns names
    full_sample_matrix = pd.DataFrame(full_sample_matrix, columns=var_names)
    #option for fold change in gene rates
    if perturbed_gene_set is not None:
        intersect_perturbed_gene_set = np.intersect1d(perturbed_gene_set, full_sample_matrix.columns)
        if len(intersect_perturbed_gene_set)<len(perturbed_gene_set):
            if verbose:
                print("The following genes were not found in your data (did you remember to pass in var_names?) : ")
                print(np.setdiff1d(perturbed_gene_set, full_sample_matrix.columns))
                print("Perturbing {} genes.".format(len(intersect_perturbed_gene_set)))
        if not len(intersect_perturbed_gene_set)==0:         
            fc_vec = np.exp(scipy.stats.norm.rvs(np.log(avg_perturb_fc), 0.15, size=len(intersect_perturbed_gene_set))) #generate fold changes
            
            # only perturb approx 80% of the cancer genes; 20% remain unaffected at random
            fc_vec[np.random.rand(len(fc_vec))>perturb_prob] = 1
            
            full_fc_vec = np.ones(full_sample_matrix.shape[1])
            full_fc_vec[full_sample_matrix.columns.isin(intersect_perturbed_gene_set)] = fc_vec

            full_sample_matrix = np.multiply(full_sample_matrix, full_fc_vec)
    #renormalize each cell
    full_sample_matrix = np.divide(full_sample_matrix.T, full_sample_matrix.sum(axis=1)).T
    #multiply gene rates by library size and sample counts from Poisson
    full_sample_matrix = np.multiply(full_sample_matrix.T, libsizes).T
    full_sample_counts = scipy.stats.poisson.rvs(full_sample_matrix)
    return full_sample_counts, metadata

def viz_props(obs_df, figsize=[5,3]):
    #visualize proportions in this dataset
    #visualize proportions in this dataset
    obs_df.celltype = obs_df.celltype.astype("category").cat.remove_unused_categories()
    celltypecounts = obs_df.groupby(["patient", "celltype"]).count().reset_index().iloc[:,:3]
    celltypecounts = celltypecounts.rename(columns = {celltypecounts.columns[2]:"ncells"})
    pt_totalcells = pd.DataFrame(obs_df.groupby("patient").size()).reset_index().rename(columns={0:'pt_totalcells'})
    celltypecounts = celltypecounts.merge(pt_totalcells, on="patient")
    celltypecounts['fraction'] = celltypecounts.ncells / celltypecounts.pt_totalcells
    celltypecounts = celltypecounts.merge(obs_df[["patient","group"]], on="patient").drop_duplicates().sort_values(["group","patient"]) #separate the two groups visually on the plot
    celltypecounts['patient'] = pd.Categorical(celltypecounts['patient'], categories=celltypecounts.patient.drop_duplicates().astype('str'), ordered=True) # ensure order doesn't change during pivot by explicitly setting the order of the categorical. categories needs to be set with a string, not another categorical, to avoid unexpected reordering
    
    fig, ax = plt.subplots(1,1, figsize=figsize)

    toplot = celltypecounts[["patient","celltype","fraction"]].pivot(index="patient", columns="celltype", values="fraction").fillna(0).reset_index()
    toplot.plot(kind="bar", stacked="true", x="patient", ax=ax, xticks=[], edgecolor=None)
    plt.legend(loc="upper left", bbox_to_anchor=[1,1]);