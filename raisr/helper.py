# -*- coding: utf-8 -*-
import functools
import sys
import os

try:
    import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


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