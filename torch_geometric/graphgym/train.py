import logging
import time
import pickle

import numpy as np
import torch
import os
import copy

from tqdm import tqdm
from torch_geometric.graphgym.checkpoint import (
    clean_ckpt,
    load_ckpt,
    save_ckpt,
)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss, compute_aux_loss, \
    compute_multi_stage_loss
from torch_geometric.graphgym.utils.epoch import (
    is_ckpt_epoch,
    is_eval_epoch,
    is_train_eval_epoch,
)


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in tqdm(loader):
        if isinstance(batch, list):
            for idx, b in enumerate(batch):
                if isinstance(b, torch.Tensor):
                    batch[idx] = b.to(torch.device(cfg.device))
                else:
                    b.to(torch.device(cfg.device))
        else:
            batch.split = 'train'
            batch.to(torch.device(cfg.device))
        optimizer.zero_grad()
        pred, true = model(batch)
        # 增加判断，如果有aux(暂时设定为只有一个)，那么pred需要拆分开计算
        if cfg.model.has_aux:
            assert isinstance(pred, tuple) and isinstance(true, tuple), \
                "pred and label should be tuples."
            label, aux_label = true[0], true[1]
            main_loss, pred_score = compute_loss(pred[0], label, model=model)
            aux_loss, aux_pred_score = compute_aux_loss(pred[1], true[1])
            loss = main_loss + cfg.model.aux_weight * aux_loss
            # loss = main_loss + model.model.weight_balance * aux_loss

        else:
            label, aux_label = true, None
            loss, pred_score = compute_loss(pred, label, model=model)
            aux_pred_score = None

        loss.backward()
        optimizer.step()

        logger.update_stats(
            true=label.detach().cpu(),
            pred=pred_score.detach().cpu(), loss=loss.item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - time_start,
            params=cfg.params, node_area=batch[1]['meta_data']['node_area'],
            aux_pred=None if aux_pred_score is None else [aux_pred_score.detach().cpu()],
            aux_loss=None if aux_label is None else [aux_loss.item() * cfg.model.aux_weight],
            aux_true=None if aux_label is None else [aux_label.detach().cpu()]
        )
        time_start = time.time()
    scheduler.step()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()

    fileOrders, extra_infos = [], []
    for batch in tqdm(loader):
        if isinstance(batch, list):
            for idx, b in enumerate(batch):
                if isinstance(b, torch.Tensor):
                    batch[idx] = b.to(torch.device(cfg.device))
                else:
                    b.to(torch.device(cfg.device))
        else:
            batch.split = split
            batch.to(torch.device(cfg.device))
        if cfg.val.extra_infos:
            temp_batch = copy.deepcopy(batch)
            extra_info = model.model.forward_test(temp_batch)
            edges = temp_batch[1].edge_index
            extra_infos.append((
                extra_info.cpu().numpy(), edges.cpu().numpy()
            ))

        pred, true = model(batch)
        if cfg.model.has_aux:
            assert isinstance(pred, tuple) and isinstance(true, tuple), \
                "pred and label should be tuples."
            label, aux_label = true[0], true[1]
            main_loss, pred_score = compute_loss(pred[0], label, model=model)
            aux_loss, aux_pred_score = compute_aux_loss(pred[1], true[1])
            loss = main_loss + cfg.model.aux_weight * aux_loss
            # loss = main_loss + model.model.weight_balance * aux_loss
        else:
            label, aux_label = true, None
            loss, pred_score = compute_loss(pred, label, model=model)
            aux_pred_score = None

        edges = batch[1].edge_dual
        # valid_edges = edges[batch[1].edge_type[:, 1] == 1, :]
        valid_edges = edges
        logger.update_stats(
            true=label.detach().cpu(),
            pred=pred_score.detach().cpu(), loss=loss.item(),
            lr=0, time_used=time.time() - time_start,
            params=cfg.params, node_area=batch[1]['meta_data']['node_area'],
            aux_pred=None if aux_pred_score is None else [aux_pred_score.detach().cpu()],
            aux_true=None if aux_label is None else [aux_label.detach().cpu()],
            aux_loss=None if aux_label is None else [aux_loss.item() * cfg.model.aux_weight],
            edge_set=None if aux_label is None else [valid_edges.cpu()]
        )

        fileOrders.append(batch[1].meta_data['filename'])
        time_start = time.time()

    if cfg.train.mode == 'eval':
        dataCates, dataExtras = [], []
        dataPreds = logger['_pred']
        dataReals = logger['_true']

        if cfg.model.has_aux:
            dataAuxPreds, dataAuxReals = logger._custom_stats['aux_pred'], \
                                        logger._custom_stats['aux_true']
            valid_edges = logger._custom_stats['edge_set']
            for fileOrder, dataPred, dataReal, dataAuxPred, dataAuxReal, valid_edge in \
                    zip(fileOrders, dataPreds,
                        dataReals, dataAuxPreds, dataAuxReals,
                        valid_edges):
                dataCate = logger._get_pred_int(dataPred).numpy()
                dataAuxCate = logger._get_pred_int(dataAuxPred).numpy()
                dataCates.append((
                    fileOrder,
                    dataCate, dataReal.numpy(),
                    dataAuxCate, dataAuxReal.numpy(), valid_edge.numpy()
                ))
        else:
            for fileOrder, dataPred, dataReal in zip(fileOrders, dataPreds, dataReals):
                dataCate = logger._get_pred_int(dataPred).numpy()
                dataCates.append((fileOrder, dataCate, dataReal.numpy()))

        if cfg.val.extra_infos:
            for fileOrder, extra_info in zip(fileOrders, extra_infos):
                dataExtras.append((
                    fileOrder, *extra_info
                ))
            with open(
                    os.path.join(cfg.run_dir, '{}_result_extra.pkl'.format(split)), 'wb'
            ) as f:
                pickle.dump(dataExtras, f)

        return dataCates
    else:
        return None


def train(loggers, loaders, model, optimizer, scheduler):
    """
    The core training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(
            model, optimizer, scheduler,
            cfg.train.epoch_resume,
            cfg.train.ckpt_prefix
        )
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        if is_train_eval_epoch(cur_epoch):
            loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                loggers[i].write_epoch(cur_epoch)
            if loggers[1].best_stat['updated']:
                save_ckpt(model, optimizer, scheduler, epoch=1, prefix='best')
                loggers[1].best_stat['updated'] = False

        if is_ckpt_epoch(cur_epoch) and cfg.train.enable_ckpt:
            save_ckpt(model, optimizer, scheduler, cur_epoch, prefix=cfg.train.ckpt_prefix)

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.run_dir))


def eval(loggers, loaders, model):
    """
    The core evaluation pipeline.
    """
    start_epoch = load_ckpt(
        model, None, None,
        cfg.train.epoch_resume,
        cfg.train.ckpt_prefix
    )
    num_splits = len(loggers)
    if num_splits == 3:
        loaders.pop(0)
        loggers.pop(0)
        num_splits -= 1
    split_names = ['val', 'test']
    assert num_splits == len(split_names), \
        'wrong match on the number of data loaders with splits.'

    for i, split in enumerate(split_names):
        predWrite = eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        loggers[i].write_epoch(0)
        # print(loggers[i])
        with open(
                os.path.join(cfg.run_dir, '{}_result.pkl'.format(split)), 'wb'
        ) as f:
            pickle.dump(predWrite, f)

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()