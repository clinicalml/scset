import math
import time
import matplotlib.pyplot as plt

import torch
import numpy as np
from utils import AverageValueMeter

def train_one_epoch_supervised(epoch, model, optimizer, criterion, args, train_loader, avg_meters, logger):
    start_time = time.time()
    model.train()
    for bidx, data in enumerate(train_loader):
        step = bidx + len(train_loader) * epoch
        bsize = data['set'].size(0)
        preds = model(data['set'].cuda(), data['set_mask'].cuda())
        targets = data['target']
        loss = criterion(preds, torch.tensor(targets).cuda())
        acc = torch.sum(torch.argmax(preds, dim=1) ==  torch.tensor(targets).squeeze().cuda())/preds.shape[0]
        optimizer.zero_grad()
        loss.backward()
        # compute gradient norm
        total_norm = 0.
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        if logger is not None: #and total_norm > 1000:
            logger.add_scalar('grad_norm', total_norm, step)
        optimizer.step()

        avg_meters['acc_avg_meter'].update(acc.detach().item(), bsize)
        avg_meters['loss_avg_meter'].update(loss.detach().item(), bsize)

       
        # assert after logging and optimizing to sync subprocesses
        loss_finite = math.isfinite(loss.detach().item())
        assert loss_finite

    if logger is not None:
        logger.add_scalar('train acc (epoch)', avg_meters['acc_avg_meter'].avg, epoch)
        logger.add_scalar('train x-ent loss (epoch)', avg_meters['loss_avg_meter'].avg, epoch)
        avg_meters['acc_avg_meter'].reset()
        avg_meters['loss_avg_meter'].reset()

def validate_supervised(model, args, val_loader, epoch, criterion, logger):
    model.eval()
    start_time = time.time()
    acc_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    for bidx, data in enumerate(val_loader):
        bsize = data['set'].size(0)
        preds = model(data['set'].cuda(), data['set_mask'].cuda())
        targets = data['target']
        loss = criterion(preds, torch.tensor(targets).cuda())
        acc = torch.sum(torch.argmax(preds, dim=1) ==  torch.tensor(targets).squeeze().cuda())/len(torch.argmax(preds, dim=1) ==  torch.tensor(targets).squeeze().cuda())

        """
        # compute gradient norm
        total_norm = 0.
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        if logger is not None and total_norm > 1000:
            logger.add_scalar('grad_norm', total_norm, step)
        """

        # Only main process writes logs.
        acc_meter.update(acc.detach().item(), bsize)
        loss_meter.update(loss.detach().item(), bsize)

        # assert after logging and optimizing to sync subprocesses
        loss_finite = math.isfinite(loss.detach().item())
        assert loss_finite

    # log val set stats for this epoch
    duration = time.time() - start_time
    print("<VAL> Epoch %d Time [%3.2fs] Acc %2.5f Loss %2.5f"
            % (epoch, duration, acc_meter.avg, loss_meter.avg))
    if logger is not None:
        logger.add_scalar('val acc (epoch)', acc_meter.avg, epoch)
        logger.add_scalar('val x-ent loss (epoch)', loss_meter.avg, epoch)

    return {'val_acc': acc_meter.avg, 'val_loss': loss_meter.avg}

def test_supervised(model, args, test_loader, epoch, criterion, logger):
    model.eval()
    start_time = time.time()
    acc_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    for bidx, data in enumerate(test_loader):

        bsize = data['set'].size(0)
        preds = model(data['set'].cuda(), data['set_mask'].cuda())
        #preds = output['predictions']
        targets = data['target']
        loss = criterion(preds, torch.tensor(targets).cuda())
        acc = torch.sum(torch.argmax(preds, dim=1) ==  torch.tensor(targets).cuda())/len(torch.tensor(targets).cuda())

        # Only main process writes logs.
        acc_meter.update(acc.detach().item(), bsize)
        loss_meter.update(loss.detach().item(), bsize)

        # assert after logging and optimizing to sync subprocesses
        loss_finite = math.isfinite(loss.detach().item())
        assert loss_finite

    # log test set stats 
    print("<TEST> Epoch %d Acc %2.5f Loss %2.5f"
            % (epoch, acc_meter.avg, loss_meter.avg))
    if logger is not None:
        logger.add_scalar('test acc', acc_meter.avg, epoch)
        logger.add_scalar('test x-ent loss', loss_meter.avg, epoch)

    return {'test_acc': acc_meter.avg, 'test_loss': loss_meter.avg}