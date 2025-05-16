import torch
import torch.nn as nn

from utils import make_optimizer, get_scheduler, AverageValueMeter, save, init_rnd_seeds, resume, EarlyStopping, calc_auc
from modules.wrappers import HeadedEncoderWrapper
from pathlib import Path
from torch.utils.data import DataLoader
import math
import time
import numpy as np
from sklearn.metrics import f1_score

def train_supervised(args, encoder, classifier_head, train_dataset, val_dataset, test_dataset, device, save_dir, writer=None, freeze_encoder=False):

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=args.num_workers, 
                              collate_fn=train_dataset.collate_fn, worker_init_fn=init_rnd_seeds)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=args.num_workers,
                            collate_fn=val_dataset.collate_fn, worker_init_fn=init_rnd_seeds)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=args.num_workers,
                            collate_fn=test_dataset.collate_fn, worker_init_fn=init_rnd_seeds)

    model = HeadedEncoderWrapper(encoder, classifier_head, freeze_encoder=freeze_encoder)
    model.to(device)

    print("Wrapper built")
    parameters = model.parameters()

    n_parameters = sum(p.numel() for p in parameters if p.requires_grad)
    print(f'number of params: {n_parameters}')

    optimizer = make_optimizer(args, model)
    #cross entropy loss, weighted for class imbalance
    loss = nn.CrossEntropyLoss(weight=torch.tensor(train_dataset.class_weights).float().cuda())

    # initialize the learning rate scheduler
    lr_scheduler = get_scheduler(args, optimizer)

    # main training loop
    avg_meters = {
        'loss_avg_meter': AverageValueMeter(),
        'acc_avg_meter': AverageValueMeter(),
    }

    start_epoch = 1
    epoch = start_epoch
    if epoch == 1: #if training a new model
        best_val_loss = 1e30
        best_val_acc = 0
    else: #if reading in a saved model
        print("NOT YET IMPLEMENTED")
        # TO DO
        #best_val_totalloss #should be calculated based on best saved model

    #initialize early stopping
    if args.earlystop_patience_ft > 0:
        earlystoppatience = args.earlystop_patience_ft / args.val_freq #convert num epochs into num val checks
        early_stopping = EarlyStopping(patience=earlystoppatience, min_delta=args.earlystop_mindelta)
  
    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs+1):
        # train for one epoch
        train_one_epoch_supervised(args, epoch, model, optimizer, loss, train_loader, avg_meters, writer)

        # evaluate on the validation set
        if epoch==1 or epoch % args.val_freq == 0 or epoch == args.epochs:
            model.eval()
            val_losses = evaluate(args, model, val_loader, epoch, loss, writer, prefix='val')            
            #save best model based on accuracy, not loss
            if val_losses['val_acc'] >= best_val_acc:
                best_val_acc = val_losses['val_acc']
                best_val_loss = val_losses['val_loss']
                save(model, optimizer, lr_scheduler, epoch, val_losses,
                    Path(save_dir) / f'testfold{args.testfold}-finetuning-checkpoint-best.pt')
                
            # Check for early stopping
            if args.earlystop_patience_ft > 0:
                early_stopping(1-val_losses['val_acc']) #try stopping based on acc instead of loss #early_stopping(val_losses['val_loss'])
                if early_stopping.early_stop:
                    print("Early stopping triggered at epoch:", epoch)
                    break

        # if epoch % args.save_freq == 0 or epoch == args.epochs:            
        #     # save the latest checkpoint at save_freq interval
        #     save(model, optimizer, lr_scheduler, epoch, None,
        #             Path(save_dir) / 'finetuning-checkpoint-latest.pt')

        # adjust the learning rate
        lr_scheduler.step()
        # log learning rate
        writer.add_scalar('learning rate', lr_scheduler.get_last_lr()[0], epoch)

    # evaluate best model on val set once after training is done, to record best performance
    print("loading best model from: " + str(Path(save_dir) / f'testfold{args.testfold}-finetuning-checkpoint-best.pt'))
    best_epoch = resume(Path(save_dir) / f'testfold{args.testfold}-finetuning-checkpoint-best.pt', model, optimizer=None, strict=True)
    model.eval()

    print("Evaluating best models performance on train, val and test sets...")
    train_losses = evaluate(args, model, train_loader, best_epoch, loss, writer=None, prefix='train', best_model=True)
    val_losses = evaluate(args,model, val_loader, best_epoch, loss, writer=None, prefix='val', best_model=True)
    test_losses = evaluate(args, model, test_loader, best_epoch, loss, writer=None, prefix='test', best_model=True)

    if writer is not None:
        writer.flush()
        writer.close()

    return (train_losses['train_acc'], train_losses['train_auc'], train_losses['train_weightedf1'], train_losses['train_loss'], val_losses['val_acc'], val_losses['val_auc'], val_losses['val_weightedf1'], val_losses['val_loss'], test_losses['test_acc'], test_losses['test_auc'], test_losses['test_weightedf1'], test_losses['test_loss'])

def train_one_epoch_supervised(args, epoch, model, optimizer, criterion, train_loader, avg_meters, writer):
    model.train()
    #bc auc is undefined for single class, will calc once per epoch rather than per batch
    preds_auc = []
    targets_auc = []
    for bidx, data in enumerate(train_loader):
        step = bidx + len(train_loader) * epoch
        bsize = data['set'].size(0)
        preds = model(data['set'].cuda(), data['set_mask'].cuda())
        targets = data['target'].cuda()
        loss = criterion(preds, targets)
        acc = torch.sum(torch.argmax(preds, dim=1) ==  targets)/preds.shape[0]
        # get softmax probabilities for AUC
        preds_auc.append(torch.nn.functional.softmax(preds, dim=1))
        targets_auc.append(targets)
        optimizer.zero_grad()
        loss.backward()
        # compute gradient norm
        total_norm = 0.
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        if writer is not None: #and total_norm > 1000:
            writer.add_scalar('grad_norm', total_norm, step)
        optimizer.step()

        # Only main process writes logs.
        avg_meters['acc_avg_meter'].update(acc.detach().item(), bsize)
        avg_meters['loss_avg_meter'].update(loss.detach().item(), bsize)

        loss_finite = math.isfinite(loss.detach().item())
        assert loss_finite

    #calculate AUC for this epoch
    preds_auc = torch.cat(preds_auc, dim=0).cpu().detach().numpy()
    targets_auc = torch.cat(targets_auc).cpu().detach().numpy()
    epoch_auc = calc_auc(preds_auc, targets_auc)

    if writer is not None:
        writer.add_scalar('Finetuning/Train Accuracy (epoch)', avg_meters['acc_avg_meter'].avg, epoch)
        writer.add_scalar('Finetuning/Train Loss (epoch)', avg_meters['loss_avg_meter'].avg, epoch)
        writer.add_scalar('Finetuning/Train AUC (epoch)', epoch_auc, epoch)
        avg_meters['acc_avg_meter'].reset()
        avg_meters['loss_avg_meter'].reset()


@torch.no_grad()
def evaluate(args, model, loader, epoch, criterion, writer, prefix='train', best_model=False):
    model.eval()
    start_time = time.time()
    acc_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    #bc auc is undefined for single class, will calc once per epoch rather than per batch
    all_preds = []
    preds_auc = []
    targets_auc = []
    for bidx, data in enumerate(loader):
        bsize = data['set'].size(0)
        preds = model(data['set'].cuda(), data['set_mask'].cuda())
        preds_threshoulded = torch.argmax(preds, dim=1)
        all_preds.append(preds_threshoulded)
        targets = data['target'].cuda()
        loss = criterion(preds, targets)
        acc = torch.sum(preds_threshoulded ==  targets)/len(preds_threshoulded ==  targets)
        # get softmax probabilities for AUC
        preds_auc.append(torch.nn.functional.softmax(preds, dim=1))
        targets_auc.append(targets)

        # Only main process writes logs.
        acc_meter.update(acc.detach().item(), bsize)
        loss_meter.update(loss.detach().item(), bsize)

        # assert after logging and optimizing to sync subprocesses
        loss_finite = math.isfinite(loss.detach().item())
        assert loss_finite

    #calculate AUC for full dataset
    preds_auc = torch.cat(preds_auc, dim=0).cpu().detach().numpy()
    targets_auc = torch.cat(targets_auc).cpu().detach().numpy()
    epoch_auc = calc_auc(preds_auc, targets_auc)

    #weighted f1 score
    all_preds = torch.cat(all_preds).cpu().detach().numpy()
    epoch_f1 = f1_score(targets_auc, all_preds, average='weighted', zero_division=np.nan, labels=np.arange(preds.shape[1]))

    # log val set stats for this epoch
    duration = time.time() - start_time
    PRE = '<BEST MODEL> ' if best_model else ''
    if prefix == 'train':
        print(PRE + "<TRAIN> Epoch %d Time [%3.2fs] Acc %2.5f AUC %2.5f weightedf1 %2.5f Loss %2.5f"
            % (epoch, duration, acc_meter.avg, epoch_auc, epoch_f1, loss_meter.avg))
        if writer is not None:
            writer.add_scalar('Finetuning/Train Accuracy (epoch)', acc_meter.avg, epoch)
            writer.add_scalar('Finetuning/Train Loss (epoch)', loss_meter.avg, epoch)
            writer.add_scalar('Finetuning/Train AUC (epoch)', epoch_auc, epoch)
            writer.add_scalar('Finetuning/Train Weighted F1 (epoch)', epoch_f1, epoch)
    if prefix == 'val':
        print(PRE+"<VAL> Epoch %d Time [%3.2fs] Acc %2.5f AUC %2.5f weightedf1 %2.5f Loss %2.5f"
            % (epoch, duration, acc_meter.avg, epoch_auc, epoch_f1, loss_meter.avg))
        if writer is not None:
            writer.add_scalar('Finetuning/Validation Accuracy (epoch)', acc_meter.avg, epoch)
            writer.add_scalar('Finetuning/Validation Loss (epoch)', loss_meter.avg, epoch)
            writer.add_scalar('Finetuning/Validation AUC (epoch)', epoch_auc, epoch)
            writer.add_scalar('Finetuning/Validation Weighted F1 (epoch)', epoch_f1, epoch)
    elif prefix == 'test':
        print(PRE+"<TEST> Epoch %d Time [%3.2fs] Acc %2.5f AUC %2.5f weightedf1 %2.5f Loss %2.5f"
            % (epoch, duration, acc_meter.avg, epoch_auc, epoch_f1, loss_meter.avg))
        if writer is not None:
            writer.add_scalar('Finetuning/Test Accuracy', acc_meter.avg, epoch)
            writer.add_scalar('Finetuning/Test Loss', loss_meter.avg, epoch)
            writer.add_scalar('Finetuning/Test AUC', epoch_auc, epoch)
            writer.add_scalar('Finetuning/Test Weighted F1', epoch_f1, epoch)
    return {f'{prefix}_acc': acc_meter.avg, f'{prefix}_loss': loss_meter.avg, f'{prefix}_weightedf1':epoch_f1, f'{prefix}_auc': epoch_auc}
