# -*- coding: utf-8 -*-
import cv2
import numpy as np
import weakref
from scipy.misc import imresize
from scipy import interpolate
from math import floor


class Pixel:
    def __init__(self, parent, row, col):
        self._parent = weakref.ref(parent)
        self._col = col
        self._row = row
        self._value = parent[row, col]
    
    @property
    def col(self):
        return self._col
    
    @property
    def row(self):
        return self._row
    
    @property
    def value(self):
        return self._value
    
    def patch(self, size):
        return self._parent().patch(self._row, self._col, size)

class Image:
    def __init__(self, data, mode = 'RGB'):
        self._data = data
        self._mode = mode
    
    @classmethod
    def from_file(cls, fname):
        return cls(cv2.imread(fname), mode = 'RGB')
        
    def __getitem__(self, indices):
        return self._data[indices]
    
    def patch(self, row, col, size):
        margin = size // 2
        return self._data[row-margin:row+margin+1, col-margin:col+margin+1]
    
    def pixels(self, *, margin = 0):
        height, width = self.shape
        for row in range(margin, height - margin):
            for col in range(margin, width - margin):
                yield Pixel(self, row, col)
    
    def number_of_pixels(self, *, margin = 0):
        height, width = self._data.shape
        return (height - 2*margin) * (width - 2*margin)
    
    @property
    def shape(self):
        return self._data.shape
    
    def to_grayscale(self):
        if self._mode == 'RGB':
            gray_data = cv2.cvtColor(self._data, cv2.COLOR_BGR2YCrCb)[:,:,0]
            gray_data_normalized = cv2.normalize(gray_data.astype('float'), None, 
                                                 gray_data.min() / 255.0, gray_data.max() / 255.0,
                                                 cv2.NORM_MINMAX)
            return Image(gray_data_normalized, mode = 'gray')
        elif self._mode == 'YCrCb':
            gray_data = self._data[:,:,0]
            gray_data_normalized = cv2.normalize(gray_data.astype('float'), None, 
                                                 gray_data.min() / 255.0, gray_data.max() / 255.0,
                                                 cv2.NORM_MINMAX)
            return Image(gray_data_normalized, mode = 'gray')
        else:
            raise ValueError('Expected RGB or YCrCb mode image.')
    
    def to_ycrcb(self):
        if self._mode == 'RGB':
            return Image(cv2.cvtColor(self._data, cv2.COLOR_BGR2YCrCb), mode = 'YCrCb')
        else:
            raise ValueError('Expected RGB mode image.')
    
    def to_rgb(self):
        if self._mode == 'YCrCb':
            return Image(cv2.cvtColor(np.uint8(self._data), cv2.COLOR_YCrCb2RGB), mode = 'RGB')
        else:
            raise ValueError('Expected YCrCb mode image.')

    def downscale(self, ratio):
        # TODO: Allow to choose downscaling algorithm
        height, width = self._data.shape
        downscaled_height = floor((height+1)/ratio)
        downscaled_width = floor((width+1)/ratio)
        #TODO: This method is deprecated.
        return Image(imresize(self._data, (downscaled_height, downscaled_width), interp='bicubic', mode='F'), mode = self._mode)
        #return cv2.resize(high_res, dsize = (floor((width+1)/self.ratio),floor((height+1)/self.ratio)), interpolation = cv2.INTER_CUBIC)
        #return skimage.transform.resize(high_res, (floor((height+1)/self.ratio),floor((width+1)/self.ratio)),
        #                                order = 3)
        
    def cheap_interpolate(self, ratio):
        # TODO: Allow to choose upscaling algorithm.
        if self._mode == 'gray':
            height, width = self._data.shape
            vert_grid = np.linspace(0, height - 1, height)
            horz_grid = np.linspace(0, width - 1, width)
            bilinear_interp = interpolate.interp2d(horz_grid, vert_grid, self._data, kind='linear')
            vert_grid = np.linspace(0, height - 1, height * ratio - 1)
            horz_grid = np.linspace(0, width - 1, width * ratio - 1)
            return Image(bilinear_interp(horz_grid, vert_grid), mode = self._mode)
        else:
            height, width, chan = self._data.shape
            vert_grid_LR = np.linspace(0, height - 1, height)
            horz_grid_LR = np.linspace(0, width - 1, width)
            vert_grid_HR = np.linspace(0, height - 1, height * ratio - 1)
            horz_grid_HR = np.linspace(0, width - 1, width * ratio - 1)
            result = np.zeros((len(vert_grid_HR), len(horz_grid_HR), chan))
            for idx in range(chan):
                channel = self._data[:,:,idx]
                bilinear_interp = interpolate.interp2d(horz_grid_LR, vert_grid_LR, channel, kind='linear')
                result[:,:,idx] = bilinear_interp(horz_grid_HR, vert_grid_HR)
            return Image(result, mode = self._mode)
    
    def export(self, fname):
        # TODO: Make this work also for other color modes
        cv2.imwrite(fname, cv2.cvtColor(self._data, cv2.COLOR_RGB2BGR))