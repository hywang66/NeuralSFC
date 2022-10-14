import networkx as nx
import numpy as np
from PIL import Image
from tqdm import tqdm

from basic.butils import *

# Generate scanning line space-filling curve 
def gen_line_curve_sfc(img):
    img_arr = img_to_arr(img)
    direction = +1

    if img_arr.ndim == 2:
        h, w = img_arr.shape
    elif img_arr.ndim == 3:
        h, w, nc = img_arr.shape
    else:
        raise NotImplementedError

    j, i = 0, 0
    sfc = [(j, i)]
    while not ((i == w - 1 and j == h - 1 and direction == 1) or (i == 0 and j == h - 1 and direction == -1)):
        if i == w - 1 and direction == 1:
            direction = -1
            j += 1
        elif i == 0 and direction == -1:
            direction = 1
            j += 1
        else:
            i += direction
        
        sfc.append((j, i))
 
    return sfc
    