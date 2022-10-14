import torch
import numpy as np


def squared_euclidean_distance(a, b):
    b = torch.transpose(b, 0, 1)
    a2 = torch.sum(torch.square(a), dim=1, keepdims=True)
    b2 = torch.sum(torch.square(b), dim=0, keepdims=True)
    ab = torch.matmul(a, b)
    d = a2 - 2 * ab + b2
    return d


def quantize(x, centroids):
    b, c, h, w = x.shape
    # [B, C, H, W] => [B, H, W, C]
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.view(-1, c)  # flatten to pixels
    d = squared_euclidean_distance(x, centroids)
    x = torch.argmin(d, 1)
    x = x.view(b, h, w)
    return x


def unquantize(x, centroids):
    return centroids[x]



def squared_euclidean_distance_np(a, b):
    b = np.transpose(b)
    a2 = np.sum(np.square(a), axis=1, keepdims=True)
    b2 = np.sum(np.square(b), axis=0, keepdims=True)
    ab = np.matmul(a, b)
    d = a2 - 2 * ab + b2
    return d

def quantize255(img255, centroids):
    img01 = img255.astype(np.float32) / 255.0
    assert centroids is not None and centroids.dtype == np.float32
    h, w, c = img01.shape
    img01 = img01.reshape(-1, c)
    d = squared_euclidean_distance_np(img01, centroids)
    imgq = np.argmin(d, 1)
    imgq = imgq.reshape(h, w)
    return imgq

def unquantize255(x, centroids):
    img01 =  centroids[x]
    img255 = img01*255.0
    return img255