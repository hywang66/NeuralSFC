import networkx as nx
import numpy as np
from PIL import Image
from numpy.core.numeric import normalize_axis_tuple
from torch_geometric.nn.inits import normal
from tqdm import tqdm
import pdb

from basic.butils import *

def notnannp(arr: np.array):
    assert not np.any(np.isnan(arr)), arr

def calc_total_variation(img, sfc, check=False):
    img_arr = img_to_arr(img)

    if img_arr.ndim == 2:
        h, w = img_arr.shape
    elif img_arr.ndim == 3:
        h, w, nc = img_arr.shape
    else:
        raise NotImplementedError

    if check:
        sfc_validity_check(sfc, filling_check=False, n_pixels=h*w)
    
    tv = 0
    for p1, p2 in zip(sfc[:-1], sfc[1:]):
        tv += np.mean(np.abs(img_arr[p1] - img_arr[p2]))
    return tv

# def calc_auto_correlation(img, sfc, max_dist=9, normalize=True):
#     img_arr = img_to_arr(img)

#     if img_arr.ndim == 2:
#         h, w = img_arr.shape
#     elif img_arr.ndim == 3:
#         h, w, nc = img_arr.shape
#     else:
#         raise NotImplementedError

#     sfc_validity_check(sfc, filling_check=False, n_pixels=h*w)
    
#     pixel_vals = [img_arr[p] for p in sfc]
    
#     results = []
    
#     for offset in range(max_dist + 1):
#         results.append(np.correlate(pixel_vals, np.roll(pixel_vals, -offset))[0])
    
#     if normalize:
#         max_val = results[0]
#         results = [x / max_val for x  in results]
    
#     return np.asarray(results)
    
def fixed_offset_auto_correlation(img, sfc, offset, normalize=True, info=None):
    img_arr = img_to_arr(img)
    notnannp(img_arr)

    def calc_fixed_ac(img_arr):
        pixel_vals = [img_arr[p] for p in sfc]
        
        ac = np.correlate(pixel_vals, np.roll(pixel_vals, -offset))[0]
        notnannp(ac)

        if normalize:
            max_ac = np.correlate(pixel_vals, pixel_vals)[0]
            notnannp(max_ac)
            # assert max_ac != 0.0
            # if max_ac == 0.0:
            #     print('pixel_vals')
            #     print(pixel_vals)
            #     print('img')
            #     print(img)
            #     print(f'illed gif path: {info}')
            #     raise NotImplementedError(info)
            # ac = ac / max_ac
            if max_ac != 0.0:
                ac = ac / max_ac
            else: # in case max_ac == 0.0, all black image
                ac = 0.5
            notnannp(ac)
        return ac   

    if img_arr.ndim == 2:
        h, w = img_arr.shape
        sfc_validity_check(sfc, filling_check=False, n_pixels=h*w)
        ac = calc_fixed_ac(img_arr)
    elif img_arr.ndim == 3:
        h, w, nc = img_arr.shape
        sfc_validity_check(sfc, filling_check=False, n_pixels=h*w)
        ac = np.mean([calc_fixed_ac(img_arr[:, :, i]) for i in range(nc)])
    else:
        raise NotImplementedError

    return ac


def multi_offset_auto_correlation(img, sfc, offset_list, normalize=True, info=None):
    return np.mean([fixed_offset_auto_correlation(img, sfc, offset, normalize, info=info) for offset in offset_list])


def calc_auto_correlation(img, sfc, max_dist=9, normalize=True):
    img_arr = img_to_arr(img)

    def calc_ac(img_arr):
        
        pixel_vals = [img_arr[p] for p in sfc]
        
        results = []
        
        for offset in range(max_dist + 1):
            results.append(np.correlate(pixel_vals, np.roll(pixel_vals, -offset))[0])
        
        if normalize:
            max_val = results[0]
            results = [x / max_val for x  in results]

        acs = np.asarray(results)
        return acs

    if img_arr.ndim == 2:
        h, w = img_arr.shape
        sfc_validity_check(sfc, filling_check=False, n_pixels=h*w)
        acs = calc_ac(img_arr)

    elif img_arr.ndim == 3:
        h, w, nc = img_arr.shape
        sfc_validity_check(sfc, filling_check=False, n_pixels=h*w)
        acs = np.mean([calc_ac(img_arr[:, :, i]) for i in range(nc)], axis=0)

    else:
        raise NotImplementedError


    
    return acs

