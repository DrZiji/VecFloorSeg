import logging
import os

import custom_graphgym  # noqa, register custom modules

import torch

from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
)
from torch_geometric.graphgym.loader import get_loader, load_dataset
from torch_geometric.graphgym.logger import create_logger, set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.train import train, eval
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device


def my_set_dataset_info(dataset):
    assert cfg.share.dim_node != 0 and cfg.share.dim_pos != 0,\
        "Node attribute must be specified."

    cfg.share.dim_node_in = cfg.share.dim_node * cfg.share.dim_pos
    cfg.share.dim_edge_in = cfg.share.dim_edge * cfg.share.dim_pos

    # get cfg.share.dim_out
    if 'edge' in cfg.dataset.task_type:
        assert cfg.share.dim_edge_out != 0, \
            "edge attribute must be specified because of edge task."
        cfg.share.dim_out = cfg.share.dim_edge_out

    else: #
        if cfg.share.dim_node_out == 0:
            try:
                cfg.share.dim_node_out = len(dataset[0].CLASSES)
                if 'classification' in cfg.dataset.task_type \
                        and cfg.share.dim_node_out == 2:
                    cfg.share.dim_node_out = 1

            except Exception:
                raise RuntimeError("error when specify model dim_out(in node related task.)")
        cfg.share.dim_out = cfg.share.dim_node_out

    # get cfg.share.aux_dim_out
    if cfg.model.has_aux:
        assert cfg.share.aux_dim_out != 0, \
            "auxiliary head will be initialized, plz specify the output."
    # count number of dataset splits
    cfg.share.num_splits = 0
    for _ in dataset:
        cfg.share.num_splits += 1


def create_dataset():
    r"""
    Create dataset object
    Returns: PyG dataset object
    """
    dataset = load_dataset()
    my_set_dataset_info(dataset)

    return dataset


def create_loader():
    dataset = create_dataset()
    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True)
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            get_loader(dataset[0], cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True)
        ]

    # val and test loaders
    for i in range(1, cfg.share.num_splits):
        if cfg.dataset.task == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(
                get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
                           shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(
                get_loader(dataset[i], cfg.val.sampler, cfg.val.batch_size,
                           shuffle=False))

    return loaders


if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    if not args.eval:
        dump_cfg(cfg)
    else:
        cfg.train.mode = 'eval'
        if 'train.epoch_resume' not in args.opts:
            raise ValueError('please give a specific checkpoint index')
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        set_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)
        auto_select_device()
        # Set machine learning pipeline
        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        optimizer = create_optimizer(model.parameters(), cfg.optim)
        scheduler = create_scheduler(optimizer, cfg.optim)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            train(loggers, loaders, model, optimizer, scheduler)
        elif cfg.train.mode == 'eval':
            eval(loggers, loaders, model)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)

    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
