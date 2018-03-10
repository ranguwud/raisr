# -*- coding: utf-8 -*-
import functools

@functools.lru_cache(maxsize = 128)
def make_slice_list(start, stop, width):
    return [slice(i, i + width) for i in range(start, stop)]