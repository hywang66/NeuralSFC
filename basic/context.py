import networkx as nx
import numpy as np
from PIL import Image
from tqdm import tqdm

from basic.butils import *
from basic.visualize import draw_grid_img

# assuming the 4 pixels in original image that we want to merge are
# a b
# d c

# order: a, b, c, d

# build the dual graph for the original grid graph.
def build_dual_graph(img_arr, weight_func):
    if img_arr.ndim == 2:
        oh, ow = img_arr.shape
    elif img_arr.ndim == 3:
        oh, ow, nc = img_arr.shape
    else:
        raise NotImplementedError

    assert ow % 2 == 0 and oh % 2 == 0
    dh, dw = oh // 2, ow // 2
    dual_g = nx.Graph()
    # dual_g = nx.grid_2d_graph(dh, dw)

    for y in range(dh):
        for x in range(dw):
            # create the right edge
            if x != dw - 1:
                w = weight_func(img_arr, y, x, True)
                dual_g.add_edge((y, x), (y, x + 1), weight=w)
            if y != dh - 1:        
                # create the down edge        
                w = weight_func(img_arr, y, x, False)
                dual_g.add_edge((y, x), (y + 1, x), weight=w)
    return dual_g


def build_dual_graph28(img_arr, weight_func):
    if img_arr.ndim == 2:
        oh, ow = img_arr.shape
    elif img_arr.ndim == 3:
        oh, ow, nc = img_arr.shape
    else:
        raise NotImplementedError

    assert ow % 2 == 0 and oh % 2 == 0
    dh, dw = oh // 2, ow // 2
    dual_g = nx.Graph()
    # dual_g = nx.grid_2d_graph(dh, dw)

    for y in range(dh):
        for x in range(dw):
            # create the right edge
            if x != dw - 1:
                w = weight_func(img_arr, y + 2, y + 2, True)
                dual_g.add_edge((y, x), (y, x + 1), weight=w)
            if y != dh - 1:        
                # create the down edge        
                w = weight_func(img_arr, y + 2, y + 2, False)
                dual_g.add_edge((y, x), (y + 1, x), weight=w)
    return dual_g

# build the dual graph for the original grid graph.
def iter_weights(img_arr, weight_func):
    if img_arr.ndim == 2:
        oh, ow = img_arr.shape
    elif img_arr.ndim == 3:
        oh, ow, nc = img_arr.shape
    else:
        raise NotImplementedError
    
    assert ow % 2 == 0 and oh % 2 == 0
    dh, dw = oh // 2, ow // 2
    # dual_g = nx.Graph()
    # dual_g = nx.grid_2d_graph(dh, dw)

    for y in range(dh):
        for x in range(dw):
            # create the right edge
            if x != dw - 1:
                w = weight_func(img_arr, y, x, True)
                # dual_g.add_edge((y, x), (y, x + 1), weight=w)
            if y != dh - 1:        
                # create the down edge        
                w = weight_func(img_arr, y, x, False)
                # dual_g.add_edge((y, x), (y + 1, x), weight=w)

# build initial circuits for the original grid graph
def build_circuits_graph(img_arr):
    if img_arr.ndim == 2:
        oh, ow = img_arr.shape
    elif img_arr.ndim == 3:
        oh, ow, nc = img_arr.shape
    else:
        raise NotImplementedError

    assert ow % 2 == 0 and oh % 2 == 0, f'ow: {ow}, oh: {oh}'
    dh, dw = oh // 2, ow // 2
    g = nx.Graph()
    # dual_g = nx.grid_2d_graph(dh, dw)

    for y in range(dh):
        for x in range(dw):
            a, b, c, d = to_orig_index(y, x)
            g.add_edges_from([(a, b), (b, c), (d, c), (a, d)])
            
    return g

# apply the minium spanning tree of the dual graph onto the circuits graph, resulting the (unbroken) sfc.
def apply_mst(circuits, mst):
    circuits = circuits.copy()
    for edge in mst.edges():
        (y1, x1), (y2, x2) = edge
        if x2 == x1 + 1 and y1 == y2:
            _, a, d, _ = to_orig_index(y1, x1)
            b, _, _, c = to_orig_index(y2, x2)
            circuits.add_edges_from([(a, b), (d, c)])
            circuits.remove_edges_from([(a, d), (b, c)])
        elif y2 == y1 + 1 and x1 == x2:
            _, _, b, a = to_orig_index(y1, x1)
            d, c, _, _ = to_orig_index(y2, x2)
            circuits.add_edges_from([(a, d), (b, c)])
            circuits.remove_edges_from([(a, b), (d, c)])
    return circuits


# assuming the 4 pixels in original image that we want to merge are
# a b
# d c
# advanced:
# p a b q
# k d c j
# or
# p q
# a b
# d c
# k j


# calcualte weights of an edge for running mst algorithm
def calc_weight(img_arr: np.ndarray, y1, x1, right, advanced=False):
    if img_arr.ndim == 2:
        # right case
        if right:
            y2, x2 = y1, x1 + 1
            p, a, d, k = to_orig_index(y1, x1)
            b, q, j, c = to_orig_index(y2, x2)

            
            u = abs(img_arr[a] - img_arr[b])
            w = abs(img_arr[c] - img_arr[d])
            e = abs(img_arr[a] - img_arr[d])
            f = abs(img_arr[b] - img_arr[c])

            if advanced:
                others = abs(img_arr[p] - img_arr[a]) + \
                        abs(img_arr[p] - img_arr[k]) + \
                        abs(img_arr[k] - img_arr[d]) + \
                        abs(img_arr[b] - img_arr[q]) + \
                        abs(img_arr[q] - img_arr[j]) + \
                        abs(img_arr[c] - img_arr[j])
            else:
                others = 0
        
        # down case
        else:
            y2, x2 = y1 + 1, x1
            p, q, b, a = to_orig_index(y1, x1)
            d, c, j, k = to_orig_index(y2, x2)
            
            u = abs(img_arr[a] - img_arr[d])
            w = abs(img_arr[b] - img_arr[c])
            e = abs(img_arr[a] - img_arr[b])
            f = abs(img_arr[d] - img_arr[c])
            
            if advanced:
                others = abs(img_arr[p] - img_arr[q]) + \
                        abs(img_arr[p] - img_arr[a]) + \
                        abs(img_arr[q] - img_arr[b]) + \
                        abs(img_arr[d] - img_arr[k]) + \
                        abs(img_arr[c] - img_arr[j]) + \
                        abs(img_arr[k] - img_arr[j])
            else:
                others = 0 
                
        return u + w - e - f + others

    elif img_arr.ndim == 3:
        oh, ow, nc = img_arr.shape
        weight = sum(calc_weight(img_arr[:, :, i], y1, x1, right, advanced) for i in range(nc))
        return weight

    else:
        raise NotImplementedError



# generate a context_based sfc for a gray-scale image
def gen_context_sfc(img, plot=False, advanced_weights=False, **kargs):
    img_arr = img_to_arr(img)
        
    circuits = build_circuits_graph(img_arr)
    dual_g = build_dual_graph(img_arr, weight_func=lambda arr, y1, x1, r : calc_weight(arr, y1, x1, r, advanced_weights))
    mst = nx.minimum_spanning_tree(dual_g, algorithm='prim')
    sfc_graph = apply_mst(circuits, mst)
    if plot:
#         h, w = img_arr.shape
#         draw_grid_img(img_arr, sfc_graph, node_size=h/28*50, no_img=False, edge_color='r', node_color='orange', **kargs)
        draw_grid_img(img_arr, sfc_graph, no_img=False, **kargs)

    
    sfc_graph.remove_edge((0, 0), (0, 1))
    sfc = list(nx.all_simple_paths(sfc_graph, (0, 0), (0, 1)))[0]
    
    return sfc
