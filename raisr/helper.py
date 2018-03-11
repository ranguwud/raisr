# -*- coding: utf-8 -*-
import numpy as np
import functools
import sys
import os

try:
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class SimpleProgressBar:
    def __init__(self, total = 100, desc = "", **kwargs):
        self._total = total
        self._desc = desc
        self._count = 0
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
    
    def __del__(self):
        self.close()
    
    def open(self):
        if len(self._desc) > 0:
            print(self._desc)
    
    def close(self):
        print('')
    
    def update(self, n = 1):
        old_count = self._count
        self._count += n
        new_count = old_count + n
        total = self._total
        if (new_count * 100) // total != (old_count * 100) // total:
            print('\r|', end='')
            print('#' * ((new_count * 100) // (2 * total)), end='')
            print(' ' * (50 - (new_count * 100) // (2 * total)), end='')
            print('|  {0}%'.format((new_count * 100) // total), end='')


def in_notebook():
    if 'ipykernel' in sys.modules:
        if any('SPYDER' in name for name in os.environ):
            return False
        if any('PYCHARM' in name for name in os.environ):
            return False
        return True
    else:
        return False


def select_pbar_cls():
    if TQDM_AVAILABLE:
        if in_notebook():
            return tqdm.tqdm_notebook
        else:
            return tqdm.tqdm
    else:
        return SimpleProgressBar


@functools.lru_cache(maxsize = 128)
def make_slice_list(start, stop, width):
    return [slice(i, i + width) for i in range(start, stop)]


def gaussian2d(size = 3):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    idx_max = size // 2
    y,x = np.ogrid[-idx_max:idx_max+1, -idx_max:idx_max+1]
    
    # Choose sigma such that Gaussian has value 0.01 at coordinate (0, idx_max)
    sigma = np.sqrt(idx_max**2 / (2 * np.log(100)))
    
    return np.exp( -(x**2 + y**2) / (2. * sigma**2) )
