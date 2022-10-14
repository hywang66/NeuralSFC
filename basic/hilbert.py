import math

import networkx as nx
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from PIL import Image
from tqdm import tqdm

from basic.butils import *
from basic.visualize import draw_grid_img


def gen_hilbert_sfc(img):
    img_arr = img_to_arr(img)
    
    if img_arr.ndim == 2:
        h, w = img_arr.shape
    elif img_arr.ndim == 3:
        h, w, nc = img_arr.shape
    else:
        raise NotImplementedError

    assert h == w
    p = math.log2(h)
    assert p.is_integer()
    p = int(p)
    hilbert_curve = HilbertCurve(p, 2)
    hilbert_sfc = [tuple(hilbert_curve.coordinates_from_distance(i)) for i in range(2**(2*p))]
    
    return hilbert_sfc
