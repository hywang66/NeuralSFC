from pprint import pprint

import torch
from torch.utils.data import DataLoader
from utils.functions import load_cfg

from neuralsfc.data import AnyDataset, UCMNIST
from neuralsfc.model import NeuralSFC


def run(cfg):
    print('Using config:')
    pprint(cfg)

    model = NeuralSFC(cfg)
    info = cfg.info if hasattr(cfg, 'info') else None
    if cfg.dataset in ['ucmnist', 'ucfmnist']:
        ds = UCMNIST(
            root=cfg.data_dir,
            dataset=cfg.dataset,
            train=not cfg.eval,
            ac_offset=cfg.ac_offset, 
            norm_type=cfg.weight_norm,
            out_nc=cfg.n_channels,
            n_select=cfg.n_select,
            e_class=cfg.e_class,
            info=info,
            normalize_e=cfg.normalize_e,
        )
    else:
        ds = AnyDataset(
            root=cfg.data_dir, 
            dataset=cfg.dataset,
            train=not cfg.eval, 
            ac_offset=cfg.ac_offset, 
            norm_type=cfg.weight_norm,
            out_nc=cfg.n_channels,
            e_class=cfg.e_class,
            info=info,
            normalize_e=cfg.normalize_e,
            )


    if not cfg.eval:
        len_val = int(cfg.val_ratio * len(ds))
        len_train = len(ds) - len_val
        train_dataset, val_dataset = torch.utils.data.random_split(ds, [len_train, len_val], generator=torch.Generator().manual_seed(42))

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.n_workers, shuffle=True), 
            'val': DataLoader(val_dataset, batch_size=cfg.val_batch_size, num_workers=cfg.n_workers, shuffle=True)
        }

        model.train(dataloaders)
    else:
        model.eval()


if __name__ == "__main__":
    run(load_cfg(base_path='configs/base_config.yml'))
