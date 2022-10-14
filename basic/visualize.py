import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from basic.butils import *


# draw a grid graph with its original image for visualization 
def draw_grid_img(img_arr, g, h=None, w=None, save=False, save_path='plot.pdf', no_img=False, **kargs):
    if img_arr.ndim == 2:
        hh, ww = img_arr.shape
    elif img_arr.ndim == 3:
        hh, ww, nc = img_arr.shape
    else:
        raise NotImplementedError
    
    h = hh if h is None else h
    w = ww if w is None else w
        
    fig = plt.figure(figsize=(w/2.8, h/2.8), dpi=70)
    ax0 = fig.add_axes([0, 0, 1, 1])
    ax0.set_axis_off()
    
    if not no_img:
        tx = img_arr.repeat(7, 0).repeat(7, 1)
        ax0.imshow(tx / 255)

    pos = {}
    for n in g:
        y, x = n
        pos[n] = (x * 7 + 3, y * 7 + 3)

    # default settings
    if 'node_size' not in kargs:
        kargs['node_size'] = h/28*50
    if 'edge_color' not in kargs:
        kargs['edge_color'] = 'r'
    if 'node_color' not in kargs:
        kargs['node_color'] = 'orange'
    
    nx.draw(g, pos, ax0, **kargs)
    
    if save:
        plt.savefig(save_path)
        
def draw_sfc(img, sfc, **kargs):
    g = sfc_to_graph(sfc)
    img_arr = img_to_arr(img)
    draw_grid_img(img_arr, g, **kargs)
