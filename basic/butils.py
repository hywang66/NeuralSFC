import networkx as nx
import numpy as np
from PIL import Image

# assuming the 4 pixels in original image that we want to merge are
# a b
# d c

# order: a, b, c, d
def to_orig_index(y, x):
    return [(y * 2, x * 2), (y * 2, x * 2 + 1), (y * 2 + 1, x * 2 + 1), (y * 2 + 1, x * 2)]

def sfc_validity_check(sfc, filling_check=False, n_pixels=0):
    '''
    If filling_check is set as True, also check if this curve fill the entire space.
    n_pixels specifies the number of pixels in this space.
    
    '''
    # if isinstance(sfc, np.ndarray):
    #     sfc = [tuple(x) for x in sfc]

    # failed = False
    # if len(sfc) != len(set(sfc)) or (filling_check and len(sfc) != n_pixels):
    #     failed = True
    # else:
    #     for a, b in zip(sfc[:-1], sfc[1:]):
    #         (x1, y1), (x2, y2) = a, b
    #         if x1 == x2:
    #             if abs(y1 - y2) != 1:
    #                 failed = True
    #                 break
    #         elif y1 == y2:
    #             if abs(x1 - x2) != 1:
    #                 failed = True
    #                 break
    #         else:
    #             failed = True
    #             break
    # if failed:
    #     raise Exception('Not a valid space-filling curve')
    pass

def img_to_arr(img):
    if type(img) is Image.Image:
        img_arr = np.asarray(img).astype(np.float)
    elif type(img) is np.ndarray:
        img_arr = img.astype(np.float)
    else:
        raise NotImplementedError(type(img))
    return img_arr

def sfc_to_graph(sfc):
    if isinstance(sfc, np.ndarray):
        sfc = [tuple(x) for x in sfc]
    g = nx.Graph()
    for a, b in zip(sfc[:-1], sfc[1:]):
        g.add_edge(a, b)
    return g


def pad_mnist(img):
    '''
    pad 28x28 MNIST image to 32x32
    '''
    img = img_to_arr(img)
    return np.pad(img, ((2, 2), (2, 2)), 'constant', constant_values=0)

