import numpy as np
import os

from utils.quantize import quantize255
import subprocess
from hashlib import md5


# Modified from https://github.com/h-khalifa/python-LZW/

def LZW_encode(uncompressed):
 
    # Build the dictionary.
    # only big letters 
    # dict_size = 26
    # dictionary = {chr(i+ord('A')): i for i in range(dict_size)}

    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
 
    p = ""
    output = []
    for c in uncompressed:
        temp = p + c
        if temp in dictionary:
            p = temp
        else:
            output.append(dictionary[p])
            # Add temp to the dictionary.
            dictionary[temp] = dict_size
            dict_size += 1
            p = c
 
    # Output the code for w.
    if len(p):
        output.append(dictionary[p])
    return output


def LZW_decode(compressed):
    
    # Build the dictionary.
    # dict_size = 26
    # dictionary = {i: chr(i+ord('A')) for i in range(dict_size)}
    dict_size = 256
    dictionary = {i: chr(i) for i in range(dict_size)}
    
 
    result = ""
    p = ""
    bol = False     
    for k in compressed:
       
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = p + p[0]
        
        result += (entry)
        
        # Add p+entry[0] to the dictionary unless it's first element
        if bol:
            dictionary[dict_size] = p + entry[0]
            dict_size += 1
 
        p = entry
        bol = True
    return result 


def get_lzw_length(params):
    img255, sfc = params

    # pdb.set_trace()
    sfc = np.asarray(sfc)
    
    if img255.ndim == 3: # RGB image
        assert img255.shape[-1] == 3
        # pdb.set_trace()
        img: np.ndarray  = quantize255(img255, get_centroids())
    else: # greyscale image
        img: np.ndarray = img255.astype(np.uint8)

    seq = img[sfc[:, 0], sfc[:, 1]]
    seq = ''.join([chr(i) for i in seq])
    
    cipher = LZW_encode(seq)
    return len(cipher)


def get_centroids():
    from utils.cfg import cfg_global
    if hasattr(cfg_global, 'centroids') and cfg_global.centroids is not None:
        return cfg_global.centroids
    
    if cfg_global.dataset == 'ffhq32':
        url = 'https://drive.google.com/uc?export=download&id=1Nndy8ay3TUmZbB8TBv8gE3Qhmp2QBVdz'
        centroids_md5 = '99600171bd782471113c7f5aa8fa3583'
        path = os.path.join(cfg_global.data_dir, 'ffhq32_centroids_256.npy')
        if os.path.exists(path) and md5(open(path, 'rb').read()).hexdigest() != centroids_md5:
            os.remove(path)
        
        if not os.path.exists(path):
            n_trials = 5
            for i in range(n_trials):
                print(f'Trying to download {i + 1}/{n_trials}th try.')
                try:
                    subprocess.run(['wget', url, '-O', path], timeout=10)
                except subprocess.TimeoutExpired:
                    print('Downloading timed out.')
                    if os.path.exists(path):
                        os.remove(path)
                    continue
                else:
                    if md5(open(path, 'rb').read()).hexdigest() == centroids_md5:
                        print(f'Downloaded {path} successfully.')
                        break
                    else:
                        print(f'Downloaded file {path} has wrong md5. Trying again.')
                        os.remove(path)
                        continue
            
        assert os.path.exists(path)
        centroids = np.load(path)
        print('centroids loaded once.')
    else:
        raise NotImplementedError(f'Centroids for {cfg_global.dataset} not implemented. Please implement it yourself.')
    
    cfg_global.centroids = centroids
    return centroids

