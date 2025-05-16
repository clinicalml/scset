from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from utils import write_results, calc_auc, init_rnd_seeds, write_results_sampleefficiency
from torch.utils.data import DataLoader
from datasets.RNASeq import CellBagDataset
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_validate, train_test_split

def nested_cross_validate(X, y, fold_ids, param_grid, inner_cv, outer_cv, scoring='accuracy', subsample_size=None, random_state=0):

    # if i wanted to return val scores to use for other hyperparameter tuning, would need to use 'cv_results_' attribute of GridSearchCV
    # currently only returns test performance

    test_accs=[]
    test_aucs=[]
    test_f1s=[]
    test_fold_ids=[]
    
    for fold in np.unique(fold_ids): #outer CV loop

        # Get the data for the current fold (for kmeans encodings, different embeddings for each fold)
        if isinstance(X, dict): 
            X_fold = X[fold]
        else:
            X_fold = X

        # Get train/test for current fold / outer CV loop
        train_idx = np.where(fold_ids != fold)[0]
        test_idx = np.where(fold_ids == fold)[0]

        # Split the data into training and testing for the outer loop
        X_train, X_test = X_fold[train_idx], X_fold[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = fold_ids[train_idx]

        # for sample efficiency experiments
        if subsample_size is not None:
            if isinstance(X, dict):
                print("Subsampling not supported for kmeans encodings (as clustering would need to be relearned for each subsample).")
                return None, None, None, None
            if X_train.shape[0] < subsample_size:
                print("Not enough samples to subsample. subsample size = {} and training data size = {}".format(subsample_size, X_train.shape[0]))
                return None, None, None, None            
            # get indexes stratified by label y
            X_train, _, y_train, _, groups_train, _ = train_test_split(X_train, y_train, groups_train, stratify=y_train, train_size=subsample_size, random_state=random_state) 
            
        # Perform GridSearchCV with inner cross-validation on the training set
        grid_search = GridSearchCV(
            estimator=LogisticRegression(max_iter=1000),
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv.split(X_train, y_train, groups_train),
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        # Evaluate the best model on the outer test set
        best_model = grid_search.best_estimator_

        #predictions, for acc
        preds_test = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, preds_test)
        test_accs.append(test_acc)

        #probabilities, for auc
        scores_test = best_model.predict_proba(X_test)
        test_auc = calc_auc(scores_test, y_test)
        test_aucs.append(test_auc)

        #weighted f1 score
        test_f1 = f1_score(y_test, preds_test, average='weighted', zero_division=np.nan, labels=np.unique(np.concatenate([y_train, y_test])))
        test_f1s.append(test_f1)

        # current test fold
        test_fold_id = fold_ids[test_idx][0]
        test_fold_ids.append(test_fold_id)

        print(f"Outer fold {test_fold_id}, fold test acc: {test_acc:.4f}, Best params: {grid_search.best_params_}")

    return test_accs, test_aucs, test_f1s, test_fold_ids
    
def run_linear_probe(args, encoder, adata, results_dir=None, subsample_ns=None, nreps=5):
    encoder.cuda()
    encoder.eval()

    # create dataset and dataloader (full)
    full_dataset = CellBagDataset(adata=adata, pid_col=args.pid_col, target_col=args.target_col, fold_col=args.fold_col, adata_layer=args.adata_layer, split_ids=list(adata.obs[args.fold_col].unique()), max_num_cells=args.max_num_cells, latent_idx_tokeep=args.latent_idx_tokeep)
    full_dataloader = DataLoader(dataset=full_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=args.num_workers, collate_fn=full_dataset.collate_fn, worker_init_fn=init_rnd_seeds)        

    # get patient embeddings, targets and fold_ids
    embs = []
    targets = []
    folds = []
    with torch.no_grad():
        for batch in full_dataloader:
            b_embs = encoder(batch['set'].cuda(), batch['set_mask'].cuda())
            b_targets = batch['target']
            b_fold = batch['fold']
            embs.append(b_embs.squeeze().cpu().numpy())
            targets.append(b_targets.squeeze().cpu().numpy())
            folds.append(b_fold)

    embs = np.concatenate(embs, axis=0)
    targets = np.concatenate(targets, axis=0)
    folds = np.concatenate(folds, axis=0)

    # run nested cross validation to choose c for linear probe, holding one fold out at a time (in both inner and outer loops)        
    inner_cv = LeaveOneGroupOut()
    outer_cv = LeaveOneGroupOut()
    param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}
    if subsample_ns is not None:
        for n in subsample_ns:
            for rep in range(nreps):
                test_accs, test_aucs, test_f1s, test_fold_ids = nested_cross_validate(embs, targets, folds, param_grid, inner_cv, outer_cv, scoring='accuracy', subsample_size=n, random_state=rep)

                print(f"Test accs: {test_accs}")
                print(f"Test aucs: {test_aucs}")
                print(f"Test f1s: {test_f1s}")

                # write results to file
                for test_acc, test_auc, test_f1, test_id in zip(test_accs, test_aucs, test_f1s, test_fold_ids):
                    write_results_sampleefficiency(args, results_dir, 'linear_probe', 'test_acc', test_acc, test_id, n, rep)
                    write_results_sampleefficiency(args, results_dir, 'linear_probe', 'test_auc', test_auc, test_id, n, rep)
                    write_results_sampleefficiency(args, results_dir, 'linear_probe', 'test_weightedf1', test_f1, test_id, n, rep)

    else:
        test_accs, test_aucs, test_f1s, test_fold_ids = nested_cross_validate(embs, targets, folds, param_grid, inner_cv, outer_cv, scoring='accuracy')

        print(f"Test accs: {test_accs}")
        print(f"Test aucs: {test_aucs}")
        print(f"Test f1s: {test_f1s}")

        # write results to file
        for test_acc, test_auc, test_f1, test_id in zip(test_accs, test_aucs, test_f1s, test_fold_ids):        
            write_results(args, results_dir, 'linear_probe', 'test_acc', test_acc, test_id)
            write_results(args, results_dir, 'linear_probe', 'test_auc', test_auc, test_id)
            write_results(args, results_dir, 'linear_probe', 'test_weightedf1', test_f1, test_id)

    # calculate mean and std of test performance across folds
    test_aucs = np.array(test_aucs)

    test_mean_acc = np.mean(test_accs)
    test_std_acc = np.std(test_accs)
    test_mean_auc = np.mean(test_aucs[test_aucs>=0])
    test_std_auc = np.std(test_aucs[test_aucs>=0])
    test_mean_f1 = np.mean(test_f1s)
    test_std_f1 = np.std(test_f1s)

    return test_mean_acc, test_std_acc, test_mean_auc, test_std_auc, test_mean_f1, test_std_f1