import numpy as np
import cupy as cp

def relu(x):
    return cp.maximum(0, x)

def d_relu(x):
    cpx = cp.array(x)
    return cpx > 0

def soft_max(x):
    return cp.exp(x) / cp.sum(cp.exp(x))