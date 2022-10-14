import argparse
import ast
import datetime
import itertools
import os
from copy import deepcopy

import dateutil.tz
import numpy as np
import requests
import torch
import yaml
from easydict import EasyDict
from torch import nn
from tqdm import tqdm

import utils.cfg


def to_2d_coor(t: torch.tensor, n_last_dim):
    """Convert 1d flattened coordinates to 2d coordinates"""
    return torch.stack((t // n_last_dim, t % n_last_dim), -1)


def to_1d_coor(t: torch.Tensor, n_last_dim): 
    """Convert 2d coordinates to 1d flattened coordinates"""
    return t[..., 0]*n_last_dim + t[..., 1]


def construst_dual_edge_index(h=16, w=16):

    def h_edge_id2lr_nodes(edge_id):
        half_id = h * (w - 1)
        assert edge_id < half_id
        y = edge_id // (w - 1)
        x = edge_id % (w - 1)
        lnode, rnode = (y, x), (y, x + 1)
        return lnode, rnode

    def v_edge_id2ud_nodes(edge_id):
        half_id = (h - 1) * w
        assert edge_id >= half_id
        edge_id -= half_id
        y = edge_id // w
        x = edge_id % w
        unode, dnode = (y, x), (y + 1, x)
        return unode, dnode

    def node2edges(node_coor):
        half_id = (h - 1) * w
        edges = []
        y, x = node_coor
        if y != 0:
            edges.append(half_id + (y - 1) * w + x) # adding the up edge
        if x != 0:
            edges.append(y * (w - 1) + x - 1) # adding the left edge
        if y != h - 1:
            edges.append(half_id + y * w + x) # adding the down edge
        if x != w - 1:
            edges.append(y * (w - 1) + x) # adding the right edge
        return edges

    def get_neighbors(edge_id):
        half_id = (h - 1) * w
        if edge_id < half_id:
            node1, node2 = h_edge_id2lr_nodes(edge_id=edge_id)
        else:
            node1, node2 = v_edge_id2ud_nodes(edge_id=edge_id)
        neighbors = []
        for e in itertools.chain(node2edges(node1), node2edges(node2)):
            if e != edge_id:
                neighbors.append(e)
        return neighbors

    n_primal_edges = 2 * h * w - h - w

    edge_index = []
    for i in range(n_primal_edges):
        for n in get_neighbors(i):
            edge_index.append([i, n])

    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index


def load_cfg(base_path=None):
    if base_path is None:
        base_path = 'configs/base_config.yml'

    base_dir = os.path.dirname(base_path)

    with open(base_path, 'r') as f:
        cfg = EasyDict(yaml.full_load(f))


    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_path', help='config file', type=str, default=None)
    parser.add_argument('--gpu', dest='gpu', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--eval', action='store_true', help='eval')
    parser.add_argument('--eval_gen_train', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('--output_folder', '-o', type=str, default=None)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--opt', type=str, default=None, help='enable output_folder eval mode. cfg can be None in this case.')

    args = parser.parse_args()

    if args.cfg_path is not None:
        if args.cfg_path[-4:] != '.yml':
            args.cfg_path = args.cfg_path + '.yml'
        if not os.path.isfile(args.cfg_path):
            cfg_path = os.path.join(base_dir, args.cfg_path)
            if not os.path.isfile(cfg_path):
                raise FileNotFoundError('Cannot find %s or %s.' % (cfg_path, args.cfg_path))
            args.cfg_path = cfg_path

        rel_path = args.cfg_path    
        args.cfg_path = os.path.abspath(args.cfg_path)

        with open(args.cfg_path, 'r') as f:
            cfg_specified = EasyDict(yaml.full_load(f))
        
        for k, v in cfg_specified.items():
            cfg[k] = v

        for k, v in args.__dict__.items():
            if v is not None:
                cfg[k] = v

        cfg.cfg_short_name = os.path.basename(args.cfg_path)[:-4]

        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

        if torch.cuda.is_available():
            cfg.device = 'cuda'  
            torch.backends.cudnn.benchmark = cfg.benchmark
        else:
            cfg.device = 'cpu'

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        cfg_name = '/'.join(rel_path.split('/')[1:])
        cfg_name = cfg_name[:cfg_name.rfind('.')]
        if cfg.dataset is not None:
            assert isinstance(cfg.dataset, str)
            cfg_name = cfg_name.replace('anyds', cfg.dataset)

        suffix = cfg.suffix if cfg.suffix else timestamp

        if cfg.debug:
            cfg.output_folder = 'debug'
            cfg.n_workers = 0
            cfg.print_interval = 1

        if cfg.output_folder is None:
            output_dir = f'./output/{cfg_name}_{suffix}'
        else:
            output_dir = os.path.join('./output', cfg.output_folder, f'{cfg_name}_{suffix}')

        cfg.output_dir = output_dir
    
    else:
        assert args.opt and (args.eval or args.evalr)
        if torch.cuda.is_available():
            cfg.device = 'cuda'  
            torch.backends.cudnn.benchmark = cfg.benchmark
        else:
            cfg.device = 'cpu'

        yaml_path = os.path.join(args.opt, 'training.log')
        with open(yaml_path, 'r') as f:
            yaml_lines = f.readlines()

        cfg_specified = ast.literal_eval(get_yaml_content(yaml_lines))
        
        for k, v in cfg_specified.items():
            cfg[k] = v

        for k, v in args.__dict__.items():
            if v is not None:
                cfg[k] = v

        cfg.output_dir = cfg.opt
        cfg.suffix = args.opt.split('_')[-1] if not cfg.suffix else cfg.suffix

    # utils.cfg.init()
    utils.cfg.cfg_global = cfg
    return utils.cfg.cfg_global


def get_yaml_content(lines):
    for i, s in enumerate(lines):
        if s.endswith('start training...\n'):
            end = i
            break
    yaml_content = [x.strip() + '\n' for x in lines[1:end]]
    yaml_content = '{' + ''.join(yaml_content).strip('{}\n') + '}'
    return yaml_content


def search_partial_path(path: str):
    if not os.path.isfile(path):
        dirname = os.path.dirname(path)
        prefix = os.path.basename(path)
        candidate = [x for x in os.listdir(dirname) if x.startswith(prefix)]
        if len(candidate) == 1:
            path = os.path.join(dirname, candidate[0])
        else:
            raise FileNotFoundError(f'unknown model path: {path}')
    
    return path


def get_module(model):
    if type(model) is nn.DataParallel:
        model_module = model.module
    else:
        model_module = model
    return model_module


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def get_normalize_bounds():
    lo = float(utils.cfg.cfg_global.normalize_lo)
    hi = float(utils.cfg.cfg_global.normalize_hi)
    return lo, hi


def get_normalize_func():
    lo, hi = get_normalize_bounds()
    assert hi > lo
    half_span = (hi - lo) / 2
    m = (lo + hi) / 2
    normalize_func = lambda x: torch.clamp((x - m) / half_span, min=-1.0, max=1.0)
    return normalize_func

