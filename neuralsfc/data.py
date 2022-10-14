import os
import random

import networkx as nx
import numpy as np
import torch
import basic.butils as butils
from basic.butils import img_to_arr
from basic.context import (apply_mst, build_circuits_graph, build_dual_graph,
                           iter_weights)
from basic.evaluation import (calc_total_variation,
                              fixed_offset_auto_correlation,
                              multi_offset_auto_correlation)
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from tqdm import tqdm
from utils.functions import get_normalize_func
from utils.lzw import get_lzw_length, get_centroids
from utils.misc import WeightsAssigner
from hashlib import md5
import subprocess



def notnan(tensor: torch.Tensor):
    assert not torch.any(torch.isnan(tensor)), tensor

def downlaod_ffhq32(path):    
    url = {
        'ffhq_32_train.npy': 'https://drive.google.com/uc?export=download&id=1RlLlEtd25XEb_LLzYdfA0GXItV6Zk9vJ&confirm=t&uuid=977a519e-4a35-4fc1-b189-b3577f0da65c',
        'ffhq_32_test.npy': 'https://drive.google.com/uc?export=download&id=1dIcZ-QLwk0eMrlxPYFg-FDOsjITjsDam'
    }
    file_md5= {
        'ffhq_32_train.npy': 'c82c8f33884869d61cb17d4fc15dabb2',
        'ffhq_32_test.npy': '0cff6d15ff92b5181ca8dc6b9b4a0f48'
    }
    directory, name = os.path.split(path)
    os.makedirs(directory, exist_ok=True)

    get_centroids()

    if os.path.exists(path) and md5(open(path, 'rb').read()).hexdigest() == file_md5[name]:
        print(f'File {path} already exists and md5 matches.')
        return

    print(f'Downloading {name} from {url[name]}')
    command = [
        'wget',
        '-O',
        path,
        url[name]
    ]

    n_trials = 10
    timeout = 20
    for i in range(n_trials):
        print(f'\n\nTrying to download {i + 1}/{n_trials}th try. {timeout}s to timeout. You can change the timeout in neuralsfc/data.py')
        try:
            subprocess.run(command, timeout=timeout)
        except subprocess.TimeoutExpired:
            print('\nDownloading timed out.')
            if os.path.exists(path):
                os.remove(path)
            continue
        else:
            if md5(open(path, 'rb').read()).hexdigest() == file_md5[name]:
                print(f'Downloaded {path} successfully.')
                return
            else:
                print(f'Downloaded file {path} has wrong md5. Trying again.')
                os.remove(path)
                continue


def get_dataset_np(dataset: str, dataset_path: str = '.', train=True, transform=None):
    # ds_np is a 4d tensor of the whole dataset
    # ds_np: [d_size, H, W, C]
    # ds_np dtype: uint8. range: [0-255]
    # pdb.set_trace()
    if dataset == 'mnist':
        mnist = MNIST(dataset_path, train=train, download=True, transform=transform)
        ds_np = np.stack([pad_mnist(item[0]).astype(np.uint8)[:, :, None] for item in mnist])
    elif len(dataset) == 6 and dataset[:5] == 'mnist' and int(dataset[5]) in range(10):
        mnist = MNIST(dataset_path, train=train, download=True, transform=transform)
        cmnist = ClassConditionalDataset(mnist, int(dataset[5]))
        pad_mnist = img_to_arr if transform is not None else butils.pad_mnist
        ds_np = np.stack([pad_mnist(item[0]).astype(np.uint8)[:, :, None] for item in cmnist])
    elif len(dataset) == 7 and dataset[:6] == 'fmnist' and int(dataset[6]) in range(10):
        mnist = FashionMNIST(dataset_path, train=train, download=True, transform=transform) 
        cmnist = ClassConditionalDataset(mnist, int(dataset[6]))
        pad_mnist = img_to_arr if transform is not None else butils.pad_mnist
        ds_np = np.stack([pad_mnist(item[0]).astype(np.uint8)[:, :, None] for item in cmnist])
    elif dataset in ['ffhq32', 'ffhq']:
        if dataset_path.endswith('.npy'):
            np_path = dataset_path
        else:
            traintest = 'train' if train else 'test'
            np_path = os.path.join(dataset_path, f'ffhq_32_{traintest}.npy')
            print(f'Loading ffhq32 images from {np_path}')

        if isinstance(transform, transforms.Resize):
            assert transform.size == 16 or transform.size == (16, 16)
            np_path = np_path.replace('_32_', '_16_')
            print(f'Stop using ffhq32. Loading ffhq16 images from {np_path}')
        else:
            assert transform is None

        ds_np = np.load(np_path)

    else:
        raise NotImplementedError

    assert ds_np.ndim == 4
    assert ds_np.shape[-1] in [1, 3]
    assert ds_np.dtype == np.uint8
    return ds_np


class BaseDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.root = None
        self.ac_offset = None
        self.multiple_ac_offset = None
        self.train = None
        self.norm_type = None
        self.data = None
        self._dataset_specific_transform = None

    def calc_dafner_ac_tv(self, img255):
        wa = WeightsAssigner(mode='predefined', example=img255, norm_type=self.norm_type)

        circuits = build_circuits_graph(img255)
        dual_g = build_dual_graph(img255, weight_func=wa.calc_weight)

        mst = nx.minimum_spanning_tree(dual_g, algorithm='prim')
        sfc_graph = apply_mst(circuits, mst)

        sfc_graph.remove_edge((0, 0), (0, 1))
        sfc = list(nx.all_simple_paths(sfc_graph, (0, 0), (0, 1)))[0]

        # ac = fixed_offset_auto_correlation(img255, sfc, self.ac_offset, True)
        if self.multiple_ac_offset:
            ac = multi_offset_auto_correlation(img255, sfc, self.ac_offset, True)
        else:
            ac = fixed_offset_auto_correlation(img255, sfc, self.ac_offset, True)
        ac = torch.tensor(ac).type(torch.FloatTensor)

        tv = calc_total_variation(img255, sfc)
        tv = torch.tensor(tv).type(torch.FloatTensor)

        return ac, tv


    def get_weights_sfc(self, img255, info=None):
        wa = WeightsAssigner(mode='mix', example=img255, norm_type=self.norm_type)
 
        circuits = build_circuits_graph(img255)

        iter_weights(img255, weight_func=wa.calc_weight)
        wa.normalize_weights()

        weights_dafner = wa.weights.copy()

        # pdb.set_trace()
        if info is not None and 'no_weight_neg' not in info:
            if np.random.rand() > 0.5:
                wa.weights = -wa.weights

        if info is not None and 'no_mix' not in info:
            if np.random.rand() > 0.5:
                wa.mix_with(np.random.randn(*wa.weights.shape))
        # else:
        #     wa.normalize_weights()

        dual_g = build_dual_graph(img255, weight_func=wa.predefined_weight)

        mst = nx.minimum_spanning_tree(dual_g, algorithm='prim')
        sfc_graph = apply_mst(circuits, mst)

        sfc_graph.remove_edge((0, 0), (0, 1))
        sfc = list(nx.all_simple_paths(sfc_graph, (0, 0), (0, 1)))[0]

        return wa.weights, weights_dafner, sfc

    def calc_nac_stv(self, img255, sfc, info=None):
        # ac = fixed_offset_auto_correlation(img255, sfc, self.ac_offset, True)
        if self.multiple_ac_offset:
            ac = multi_offset_auto_correlation(img255, sfc, self.ac_offset, True, info=info)
        else:
            ac = fixed_offset_auto_correlation(img255, sfc, self.ac_offset, True, info=info)
        nac = -torch.tensor(ac).type(torch.FloatTensor)

        notnan(nac)

        tv = calc_total_variation(img255, sfc)
        stv = tv / len(sfc)
        # assert len(sfc) == 1024 or len(sfc) == 4096 # Only for 32 x 32 image or 64 x 64
        stv = torch.tensor(stv).type(torch.FloatTensor)

        return nac, stv

    def get_weights_nac_stv(self, img255, info=None):
        weights, weights_dafner, sfc = self.get_weights_sfc(img255, info)
        weights = torch.tensor(weights).type(torch.FloatTensor)
        weights_dafner = torch.tensor(weights_dafner).type(torch.FloatTensor)
        nac, stv = self.calc_nac_stv(img255, sfc, info)
        return weights, weights_dafner, nac, stv

    def get_weights_nac_stv_lzwl(self, img255, info=None):
        weights, weights_dafner, sfc = self.get_weights_sfc(img255, info)
        weights = torch.tensor(weights).type(torch.FloatTensor)
        weights_dafner = torch.tensor(weights_dafner).type(torch.FloatTensor)
        nac, stv = self.calc_nac_stv(img255, sfc, info)
        lzw_len = get_lzw_length((img255, sfc))
        return weights, weights_dafner, nac, stv, torch.tensor(lzw_len).type(torch.FloatTensor)

    def get_weights_lzwl(self, img255, info=None):
        weights, weights_dafner, sfc = self.get_weights_sfc(img255, info)
        weights = torch.tensor(weights).type(torch.FloatTensor)
        weights_dafner = torch.tensor(weights_dafner).type(torch.FloatTensor)
        lzw_len = get_lzw_length((img255, sfc))
        return weights, weights_dafner, torch.tensor(lzw_len).type(torch.FloatTensor)


    def __len__(self) -> int:
        return len(self.data)


class ClassConditionalDataset(Dataset):
    def __init__(self, full_dataset: Dataset, label) -> None:
        super().__init__()
        self.dataset = full_dataset
        self.mapping = [i for i in range(len(full_dataset)) if full_dataset[i][1] == label]

    def __getitem__(self, index):
        return self.dataset[self.mapping[index]]

    def __len__(self):
        return len(self.mapping)


class AnyDataset(BaseDataset):
    def __init__(self, root, dataset, ac_offset=3, train=True, norm_type='in', out_nc=None, e_class='nac', **kargs) -> None:
        super().__init__()
        self.root = root
        self.ac_offset = ac_offset
        self.e_class=e_class
        if isinstance(self.ac_offset, list):
            self.multiple_ac_offset = True
        else:
            self.multiple_ac_offset = False
        self.train = train
        self.norm_type = norm_type
        self.dataset = dataset
        self.source_nc = None
        self.out_nc = out_nc
        t_list = []
        self.normalize_e = False
        self.pil_transform = None

        self.kargs = kargs
        if self.e_class == 'lzwl':
            assert 'normalize_e' in self.kargs
            self.normalize_e = kargs['normalize_e']
            assert isinstance(self.normalize_e, bool)
            if self.normalize_e:
                self.normalize_func = get_normalize_func()


        if dataset == 'mnist':
            self.data = MNIST(root, train, download=True)
            self.source_nc = 1
            t_list.append(transforms.Pad(2))
        elif len(dataset) == 6 and dataset[:5] == 'mnist' and int(dataset[5]) in range(10):
            mnist = MNIST(root, train, download=True)
            self.data = ClassConditionalDataset(mnist, int(dataset[5]))
            t_list.append(transforms.Pad(2))
        elif dataset == 'fashionmnist':
            self.data = FashionMNIST(root, train, download=True)
            self.source_nc = 1
            t_list.append(transforms.Pad(2))
        elif len(dataset) == 7 and dataset[:6] == 'fmnist' and int(dataset[6]) in range(10):
            fmnist = FashionMNIST(root, train, download=True)
            self.data = ClassConditionalDataset(fmnist, int(dataset[6]))
            t_list.append(transforms.Pad(2))
        elif dataset == 'ffhq32':
            self.source_nc = 3
            name = 'ffhq_32_train.npy' if train else 'ffhq_32_test.npy'
            datapath = os.path.join(root, name)
            # if not os.path.exists(datapath):
            downlaod_ffhq32(datapath)
            assert os.path.exists(datapath)
            self.data = np.load(datapath)
        else:
            print(f'Unknown dataset {dataset}.')
            return

        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self._dataset_specific_transform = transforms.Compose(t_list)


    def __getitem__(self, index: int):
        if self.dataset in ['celebamask', 'ffhq32']:
            img = self.data[index]
        else:
            img = self.data[index][0]
        
        if self.dataset.startswith('mnist') or self.dataset.startswith('fmnist') or self.dataset == 'fashionmnist':
            img255 = butils.pad_mnist(img)
        elif self.dataset == 'ffhq32':
            img255 = img_to_arr(img)
        else:
            img255 = img_to_arr(img)

        img = self._dataset_specific_transform(img) # torch.Tensor

        img_size = img255.shape[-2]

        if self.e_class == 'nac':
            weights, weights_dafner, nac, stv = self.get_weights_nac_stv(img255)
            if self.out_nc is not None and (self.source_nc != self.out_nc):
                if self.source_nc == 1 and self.out_nc == 3:
                    img = img.repeat(3, 1, 1)
                    img255 = img255.reshape(img_size, img_size, 1).repeat(3, axis=-1)

            batch = img, img255, weights, weights_dafner, nac, stv

        elif self.e_class == 'lzwl':
            info = self.kargs['info'] if 'info' in self.kargs else None
            # pdb.set_trace()
            weights, weights_dafner, lzwl = self.get_weights_lzwl(img255, info=info)
            if self.normalize_e:
                lzwl = self.normalize_func(lzwl)
            batch = img, img255, weights, weights_dafner, lzwl
        else:
            raise NotImplementedError
        
        return batch

  
class UCMNIST(BaseDataset):
    # unified conditional MNIST-like dataset
    def __init__(self, root, dataset, ac_offset=3, train=True, norm_type='in', out_nc=None, n_select=16, e_class='nac', **kargs) -> None:
        super().__init__()
        self.root = root 
        self.n_select = n_select
        if dataset == 'ucfmnist':
            self.mnist = FashionMNIST(self.root, train=train, download=True)
        elif dataset == 'ucmnist':
            self.mnist = MNIST(self.root, train=train, download=True)
        else: 
            raise NotImplementedError

        print(f'Using base dataset {type(self.mnist)}.')

        self.label_index_dict = self.get_label_index_dict()
#         self.remapped_index = sum([v[:len(v) // n_select * n_select] for v in self.label_index_dict.values()], start=[])
        self.reset_remapped_index() 
        self.ac_offset = ac_offset
        if isinstance(self.ac_offset, list):
            self.multiple_ac_offset = True
        else:
            self.multiple_ac_offset = False
        self.train = train
        self.norm_type = norm_type
        assert dataset in ['ucmnist', 'ucfmnist']
        self.dataset = dataset
        assert out_nc is None or out_nc == 1
        self.out_nc = 1

        self.source_nc = 1
        self.e_class = e_class
        self.kargs = kargs
        assert 'normalize_e' in self.kargs
        self.normalize_e = kargs['normalize_e']
        assert isinstance(self.normalize_e, bool)
        if self.normalize_e:
            self.normalize_func = get_normalize_func()

        t_list = [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]

        self._dataset_specific_transform = transforms.Compose(t_list)
        
    def get_label_index_dict(self,):
        assert self.mnist is not None
        label_index_dict = {}
        for index, (_, label) in enumerate(tqdm(self.mnist)):
            if label not in label_index_dict:
                label_index_dict[label] = []
            label_index_dict[label].append(index)
        return label_index_dict

    def reset_remapped_index(self,):
        self.remapped_index = []
        for v in self.label_index_dict.values():
            img_same_class = sum([v[:len(v) // self.n_select * self.n_select] ], start=[])
            random.shuffle(img_same_class)
            self.remapped_index = self.remapped_index + img_same_class
        print('dataset remapped_index reset')
    
    def __len__(self,):
        return len(self.remapped_index) // self.n_select
    
    def __getitem__(self, index: int):
        indices = [self.remapped_index[i] for i in range(self.n_select*index, self.n_select*(index + 1))]
        class_list = [self.mnist[i][0] for i in indices]
        assert self.mnist[indices[0]][1] == self.mnist[indices[-1]][1]
        imgs_tensor = torch.stack([self._dataset_specific_transform(x) for x in class_list])
        imgs255 = img_to_arr(np.stack([butils.pad_mnist(x) for x in class_list]))
        
        if self.e_class == 'nac':
            wwns = []
            info = self.kargs['info'] if 'info' in self.kargs else None
            for img255 in imgs255:
                wwns.append(self.get_weights_nac_stv(img255, info=info))
            weights_list, weights_dafner_list, nac_list, stv_list = list(zip(*wwns))
            weights, weights_dafner, nac, stv = torch.stack(weights_list), torch.stack(weights_dafner_list), torch.stack(nac_list), torch.stack(stv_list)
            batch = imgs_tensor, imgs255, weights, weights_dafner, nac, stv

        elif self.e_class == 'lzwl':
            wwl = []
            info = self.kargs['info'] if 'info' in self.kargs else None
            for img255 in imgs255:
                wwl.append(self.get_weights_lzwl(img255, info=info))
            weights_list, weights_dafner_list, lzwl_list = list(zip(*wwl))
            weights, weights_dafner, lzwl= torch.stack(weights_list), torch.stack(weights_dafner_list), torch.stack(lzwl_list)

            # pdb.set_trace()
            if self.normalize_e:
                lzwl = self.normalize_func(lzwl)
            batch = imgs_tensor, imgs255, weights, weights_dafner, lzwl
        else:
            raise NotImplementedError

        return batch


class NPDataset(Dataset):
    def __init__(self, array, transform=None) -> None:
        super().__init__()
        self.data = array
        self.transform = transform

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform is not None:
            x = self.transform(x)

        return x

