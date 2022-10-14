import networkx as nx
import numpy as np
import torch
from basic.butils import to_orig_index
from basic.context import apply_mst, build_circuits_graph, build_dual_graph
from basic.evaluation import (calc_total_variation,
                              fixed_offset_auto_correlation)
from sklearn.preprocessing import scale
from torch import nn


class WeightsAssigner:
    def __init__(self, mode, weights=None, example=None, norm_type='minmax') -> None:
        if example is None:
            h = w = 16
        else:
            assert isinstance(example, np.ndarray)
            h, w = example.shape[:2]
            h, w = h // 2, w // 2
        n = 2 * h * w - h - w
        self.half = n // 2
        self.w = w
        self.mode = mode
        self.mix_ratio = np.random.rand()
        self.norm_type = norm_type

        if mode == 'calc' or mode == 'calc_neg':
            self.weights = np.zeros(n)
            # self.weight_func = self.calc_weight

        elif mode == 'predefined' or mode == 'mix':
            if weights is None: # generate weights randomly for training
                self.weights = np.random.randn(n)
            else: # load weights
                if type(weights) is torch.Tensor:
                    weights = weights.cpu().numpy()
                else:
                    assert type(weights) is np.ndarray
                self.weights = weights 

        else:
            raise NotImplementedError


    def mix_with(self, new_weights: np.ndarray):
        assert self.weights.shape == new_weights.shape
        self.normalize_weights()
        new_weights = self.normalize(new_weights)

        # mix = np.random.rand(*new_weights.shape) > self.mix_ratio
        # self.weights[mix] = new_weights[mix]

        self.weights = self.mix_ratio * self.weights + (1 - self.mix_ratio) * new_weights

        self.normalize_weights()


    def normalize(self, array: np.ndarray):
        """
        normalize a tensor
        """
        if self.norm_type == 'minmax':
            return (array - array.min()) / (array.max() - array.min())
        elif self.norm_type == 'in':
            return scale(array.astype(np.float64))
        else:
            raise NotImplementedError


    def normalize_weights(self,):
        self.weights = self.normalize(self.weights)


    def predefined_weight(self, img_arr, y, x, right):
        if right:
            idx = y * (self.w - 1) + x
        else:
            idx = self.half + y * self.w + x

        return self.weights[idx]


    def calc_weight(self, img_arr: np.ndarray, y1, x1, right):
        def calc2d(img_arr: np.ndarray, y1, x1, right):

            # right case
            if right:
                y2, x2 = y1, x1 + 1
                p, a, d, k = to_orig_index(y1, x1)
                b, q, j, c = to_orig_index(y2, x2)

                
                u = abs(img_arr[a] - img_arr[b])
                w = abs(img_arr[c] - img_arr[d])
                e = abs(img_arr[a] - img_arr[d])
                f = abs(img_arr[b] - img_arr[c])

                idx =  y1 * (self.w - 1) + x1
            
            # down case
            else:
                y2, x2 = y1 + 1, x1
                p, q, b, a = to_orig_index(y1, x1)
                d, c, j, k = to_orig_index(y2, x2)
                
                u = abs(img_arr[a] - img_arr[d])
                w = abs(img_arr[b] - img_arr[c])
                e = abs(img_arr[a] - img_arr[b])
                f = abs(img_arr[d] - img_arr[c])

                idx = self.half + y1 * self.w + x1
            
            weight = u + w - e - f 

            return weight, idx

        if img_arr.ndim == 2:
            weight, idx = calc2d(img_arr, y1, x1, right) 

        elif img_arr.ndim == 3:
            oh, ow, nc = img_arr.shape
            w_id = [calc2d(img_arr[:, :, i], y1, x1, right) for i in range(nc)]
            weight = sum(x[0] for x in w_id)
            idx = w_id[0][1]

        else:
            raise NotImplementedError

        self.weights[idx] = weight
        return weight


def compute_weights_ac(img_arr, weights, ac_offset=4):
    if type(img_arr) is torch.Tensor:
        img_arr = img_arr.detach().cpu().numpy()
    else:
        assert type(img_arr) is np.ndarray

    if type(weights) is torch.Tensor:
        weights = weights.detach().cpu().numpy()
    else:
        assert type(weights) is np.ndarray

    wa = WeightsAssigner(mode='predefined', example=img_arr, weights=weights)
    circuits = build_circuits_graph(img_arr)
    dual_g = build_dual_graph(img_arr, weight_func=wa.predefined_weight)

    mst = nx.minimum_spanning_tree(dual_g, algorithm='prim')
    sfc_graph = apply_mst(circuits, mst)

    sfc_graph.remove_edge((0, 0), (0, 1))
    sfc = list(nx.all_simple_paths(sfc_graph, (0, 0), (0, 1)))[0]
    ac = fixed_offset_auto_correlation(img_arr, sfc, ac_offset, True)
    return ac


def batch_compute_weights_ac(batch_img_arr, batch_weights, ac_offset=4):
    if type(batch_img_arr) is torch.Tensor:
        batch_img_arr = batch_img_arr.detach().cpu().numpy()
    else:
        assert type(batch_img_arr) is np.ndarray

    if type(batch_weights) is torch.Tensor:
        batch_weights = batch_weights.detach().cpu().numpy()
    else:
        assert type(batch_weights) is np.ndarray

    acs = [compute_weights_ac(i, w, ac_offset=ac_offset) for i, w in zip(batch_img_arr, batch_weights)]
    return acs


def compute_weights_ac_tv(img_arr, weights, ac_offset=4):

    if type(img_arr) is torch.Tensor:
        img_arr = img_arr.detach().cpu().numpy()
    else:
        assert type(img_arr) is np.ndarray

    if type(weights) is torch.Tensor:
        weights = weights.detach().cpu().numpy()
    else:
        assert type(weights) is np.ndarray
    
    wa = WeightsAssigner(mode='predefined', example=img_arr, weights=weights)
    circuits = build_circuits_graph(img_arr)
    dual_g = build_dual_graph(img_arr, weight_func=wa.predefined_weight)

    mst = nx.minimum_spanning_tree(dual_g, algorithm='prim')
    sfc_graph = apply_mst(circuits, mst)

    sfc_graph.remove_edge((0, 0), (0, 1))
    sfc = list(nx.all_simple_paths(sfc_graph, (0, 0), (0, 1)))[0]

    ac = fixed_offset_auto_correlation(img_arr, sfc, ac_offset, True)

    tv = calc_total_variation(img_arr, sfc)
    tv = tv / len(sfc)
    return ac, tv


def batch_compute_weights_ac_tv(batch_img_arr, batch_weights, ac_offset=4):

    if type(batch_img_arr) is torch.Tensor:
        batch_img_arr = batch_img_arr.detach().cpu().numpy()
    else:
        assert type(batch_img_arr) is np.ndarray

    if type(batch_weights) is torch.Tensor:
        batch_weights = batch_weights.detach().cpu().numpy()
    else:
        assert type(batch_weights) is np.ndarray
    
    actvs = [compute_weights_ac_tv(i, w, ac_offset=ac_offset) for i, w in zip(batch_img_arr, batch_weights)]
    acs, tvs = list(zip(*actvs))
    return acs, tvs



class LoHiNormalizer(nn.Module):
    def __init__(self, lo, hi):
        super().__init__()
        assert hi > lo
        self.lo = lo
        self.hi = hi
        self.span = hi - lo
    
    def forward(self, x):
        r = (x - self.lo) / self.span
        return r


class TwoOneSequential(nn.Module):
    def __init__(self, module1, module2):
        super().__init__()
        self.m1 = module1
        self.m2 = module2

    def forward(self, x, y):
        r = self.m1(x, y)
        r = self.m2(r)
        return r

