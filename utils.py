import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import torch.optim as optim
import random
import pandas as pd
import json
from sklearn.metrics import roc_auc_score

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_rnd_seeds(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)
    random.seed(seed + worker_id)


def calc_auc(preds_auc, targets_auc):
    
    # if only one class present in the targets, AUC is undefined
    if len(np.unique(targets_auc)) == 1:
        epoch_auc = -1
        print("AUC is not defined when only one target class is present")

    #if targets are binary, roc_auc_score requires 1D array, not 2D
    elif preds_auc.shape[1] == 2:
        preds_auc = preds_auc[:,1]
        epoch_auc = roc_auc_score(targets_auc, preds_auc)

    # for multiclass case (>2), special handling in case a class is missing (sklearn roc_auc_score breaks, but AUC still calculable)
    else:
        class_aucs = []
        class_weights = []
        for group in range(preds_auc.shape[1]):
            if group in np.unique(targets_auc):
                class_aucs.append(roc_auc_score(targets_auc==group, preds_auc[:,group]))
                class_weights.append(np.mean(targets_auc == group))
            else:
                print(f"Class {group} not present in the targets, excluding from AUC calculation")
        epoch_auc = np.average(class_aucs, weights=class_weights)  # Weighted-average AUC

    return epoch_auc


def pad(x, x_mask, max_size):
    if x.size(1) < max_size:
        pad_size = max_size - x.size(1)
        pad = torch.ones(x.size(0), pad_size, x.size(2)).to(x.device) * float('inf')
        pad_mask = torch.ones(x.size(0), pad_size).bool().to(x.device)
        x, x_mask = torch.cat((x, pad), dim=1), torch.cat((x_mask, pad_mask), dim=1)
    else:
        UserWarning(f"pad: {x.size(1)} >= {max_size}")
    return x, x_mask


def extend_batch(b, b_mask, x, x_mask):
    if b is None:
        return x, x_mask
    if b.size(1) >= x.size(1):
        x, x_mask = pad(x, x_mask, b.size(1))
    else:
        b, b_mask = pad(b, b_mask, x.size(1))
    return torch.cat((b, x), dim=0), torch.cat((b_mask, x_mask), dim=0)


def save(model, optimizer, scheduler, epoch, losses, path):
    d = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'losses': losses,
    }
    torch.save(d, path)


def resume(path, model, optimizer=None, scheduler=None, strict=True):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=strict)
    start_epoch = ckpt['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    return start_epoch


def load_pretrained_model_weights(encoder, pretrain, pretrained_ckpt=None, pretrained_ckpt_dir=None):
    if pretrained_ckpt is not None: #if user provided ckpt path
        print(f"Loading pretrained model from {pretrained_ckpt}")
        model_weights = torch.load(pretrained_ckpt, weights_only=True)
    else: #else pull from default loc based on encoder and pretrain type
        pretrained_ckpt = f'{pretrained_ckpt_dir}/pretrained-{encoder}-{pretrain}.pt'
        print(f"Loading pretrained model from default dir: {pretrained_ckpt}")
        try:
            model_weights = torch.load(pretrained_ckpt, weights_only=True)
        except:
            print(f"Pretrained model not found at {pretrained_ckpt}")
    return model_weights


def split_to_generator(splits_per_sample):
    """
    Yields a generator of train-test splits in sklearn format i.e. (train_idx, test_idx)
    Parameters:
    splits_per_sample: numpy.array or pd.Series containing split_id for each sample
    """
    for split_id in sorted(np.unique(splits_per_sample.unique())):
        train = np.flatnonzero(splits_per_sample!=split_id)
        test = np.flatnonzero(splits_per_sample==split_id)
        yield train, test

def get_scheduler(args, optimizer):
    if args.scheduler == 'exponential':
        assert not (args.warmup_epochs > 0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    elif args.scheduler == 'step':
        assert not (args.warmup_epochs > 0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.1)
    elif args.scheduler == 'linear':
        def lambda_rule(ep):
            lr_w = min(1., ep / args.warmup_epochs) if (args.warmup_epochs > 0) else 1.
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
            return lr_l * lr_w
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.scheduler == 'cosine':
        assert not (args.warmup_epochs > 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        # Fake SCHEDULER
        def lambda_rule(ep):
            return 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

def make_optimizer(args, model):
    params = model.parameters()
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.0)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
    else:
        assert 0, "args.optimizer should be either 'adam' or 'sgd'"
    return optimizer

def write_params(file, args):
    if not Path(file).parent.exists():
        Path(file).parent.mkdir(exist_ok=True, parents=True)

    with open(file, 'a') as f:
        #write one arg per line from parser.parse_args()
        for arg in vars(args):
            f.write("{}: {}\n".format(arg, getattr(args, arg)))

def save_config(args, directory):
    """
    Saves the config file for a given checkpoint name as a json dictionnary.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    args_dict = vars(args)
    with open(os.path.join(directory, f"{args.run_name}.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

def read_config(file):
    with open(file, "r") as f:
        return json.load(f)

def write_results(args, dir, experiment, metric, score, fold_id):

    result_dict = {
        'run_name': args.run_name,
        'encoder': args.encoder,
        'pretrain': args.pretrain,
        'experiment': experiment,
        'metric': metric,
        'score': score,
        'fold_id': fold_id
    }

    if not os.path.exists(dir):
        os.makedirs(dir)

    if os.path.exists(os.path.join(dir, f'{args.run_name}.csv')):
        df = pd.read_csv(os.path.join(dir, f'{args.run_name}.csv'))
        df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([result_dict])
    df.to_csv(os.path.join(dir, f'{args.run_name}.csv'), index=False)

def write_results_sampleefficiency(args, dir, experiment, metric, score, fold_id, nsamples, rep_id):

    result_dict = {
        'run_name': args.run_name,
        'encoder': args.encoder,
        'pretrain': args.pretrain,
        'experiment': experiment,
        'metric': metric,
        'score': score,
        'fold_id': fold_id,
        'nsamples': nsamples,
        'rep_id': rep_id
    }

    if not os.path.exists(dir):
        os.makedirs(dir)

    if os.path.exists(os.path.join(dir, f'{args.run_name}.csv')):
        df = pd.read_csv(os.path.join(dir, f'{args.run_name}.csv'))
        df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([result_dict])
    df.to_csv(os.path.join(dir, f'{args.run_name}.csv'), index=False)

def write_results_pretrain_eval(args, dir, metric, score, split):

    result_dict = {
        'encoder': args.encoder,
        'pretrain': args.pretrain,
        'split': split,
        'metric': metric,
        'score': score,
        'tag': args.tag,
        'time': args.cur_time
    }

    if not os.path.exists(dir):
        os.makedirs(dir)

    fname = f'pretraining_results_{args.tag}.csv'
    if os.path.exists(os.path.join(dir, fname)):
        df = pd.read_csv(os.path.join(dir, fname))
        df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
    else:
        df = pd.DataFrame([result_dict])
    df.to_csv(os.path.join(dir, fname), index=False)

def generate_run_name():

    color = [
        'red', 'blue', 'green', 'yellow', 'purple', 
        'orange', 'pink', 'brown', 'black', 'white', 'gold', 'silver', 'gray', 'cyan', 'magenta'
    ]
    
    adj = ['common', 'fair', 'mediocre', 'moderate', 
        'ordinary', 'amazing', 'brilliant', 'creative', 'dynamic', 'elegant', 
        'fantastic', 'gorgeous', 'heroic', 'incredible', 'jubilant', 
        'keen', 'luminous', 'magnificent', 'notable', 'outstanding', 
        'phenomenal', 'radiant', 'spectacular', 'terrific', 'vibrant'
    ]
    
    # 20 random animals
    animals = [
        'dog', 'cat', 'fish', 'bird', 'elephant', 
        'lion', 'tiger', 'bear', 'penguin', 'giraffe', 
        'zebra', 'kangaroo', 'koala', 'dolphin', 'whale', 
        'shark', 'octopus', 'jellyfish', 'butterfly', 'dragonfly'
    ]

    x = random.randint(100, 999)

    return f'{random.choice(adj)}-{random.choice(color)}-{random.choice(animals)}-{x}'

class EarlyStopping(object):
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # reset counter if validation loss improves
        else:
            self.counter += 1  # increment if no improvement
            if self.counter >= self.patience:
                self.early_stop = True