# -*- coding: utf-8 -*-
import PIL
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
        self._value = parent.getpixel(row, col)
    
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

class Line:
    def __init__(self, parent, lineno, margin):
        self._parent = weakref.ref(parent)
        self._lineno = lineno
        self._margin = margin
        
        rect = (0, lineno - margin, parent.shape[0], lineno + margin + 1)
        self._image = parent._image.crop(rect)
    
    @property
    def parent(self):
        return self._parent()
    
    @property
    def lineno(self):
        return self._lineno
    
    @property
    def margin(self):
        return self._margin
    
    def to_array(self, margin = 0):
        self_margin = self.margin
        if margin > self_margin:
            raise ValueError("Margin {0} too large for line height {1}".format(margin, 2*self_margin + 1))
        rect = (self_margin - margin, self_margin - margin, 
                self._image.size[0] - (self_margin - margin), self_margin + margin + 1)
        return np.array(self._image.crop(rect))

class Image:
    def __init__(self, image):
        self._image = image
    
    @classmethod
    def from_file(cls, fname):
        # TODO: Error if file does not exist
        return cls(PIL.Image.open(fname))
    
    @classmethod
    def from_array(cls, arr):
        return cls(PIL.Image.fromarray(arr, mode = 'L'))
    
    @classmethod
    def from_channels(cls, mode, channels):
        channel_list = [ch._image for ch in channels]
        return cls(PIL.Image.merge(mode, channel_list))
        
    def patch(self, row, col, size):
        margin = size // 2
        box = (col - margin, row - margin, col + margin + 1, row + margin + 1)
        return np.array(self._image.crop(box))
    
    def census_transform(self, row, col, operator = np.greater, fuzzyness = 0.0):
        patch = np.array(self.patch(row, col, 3))
        comp_hi = operator(patch[1,1] + fuzzyness, patch)
        comp_lo = operator(patch[1,1] - fuzzyness, patch)
        bools = np.all((comp_hi, comp_lo), axis = 0).astype(int).ravel()
        return np.dot(bools, np.array((128, 16, 4, 64, 0, 2, 32, 8, 1)))
    
    def pixels(self, *, margin = 0):
        width, height = self.shape
        for row in range(margin, height - margin):
            for col in range(margin, width - margin):
                yield Pixel(self, row, col)
    
    def lines(self, *, margin = 0):
        for lineno in range(margin, self.shape[1] - margin):
            yield Line(self, lineno, margin)
    
    def getpixel(self, row, col):
        return self._image.getpixel((col, row))
    
    def getchannel(self, identifier):
        return self.__class__(self._image.getchannel(identifier))
    
    def number_of_pixels(self, *, margin = 0):
        width, height = self.shape
        return (height - 2*margin) * (width - 2*margin)
    
    @property
    def shape(self):
        return self._image.size
    
    @property
    def mode(self):
        return self._image.mode
    
    def to_grayscale(self):
        if self.mode == 'RGB':
            return self.to_ycbcr().to_grayscale()
        elif self.mode == 'YCbCr':
            return self.__class__(self._image.getchannel('Y'))
        else:
            raise ValueError('Expected RGB or YCbCr mode image.')
    
    def to_ycbcr(self):
        if self.mode == 'RGB':
            return self.__class__(self._image.convert('YCbCr'))
        else:
            raise ValueError('Expected RGB mode image.')
    
    def to_rgb(self):
        if self.mode == 'YCbCr':
            return self.__class__(self._image.convert('RGB'))
        else:
            raise ValueError('Expected YCbCr mode image.')
    
    def crop(self, box):
        return self.__class__(self._image.crop(box))

    def downscale(self, ratio, method = 'bicubic'):
        if method == 'bicubic':
            resample = PIL.Image.BICUBIC
        elif method == 'bilinear':
            resample = PIL.Image.BILINEAR
        elif method == 'lanczos':
            resample = PIL.Image.LANCZOS
        elif method == 'nearest':
            resample = PIL.Image.NEAREST
        else:
            raise ValueError('Unknown resampling method "{0}"'.format(method))

        width, height = self.shape
        downscaled_height = floor((height - 1) / ratio) + 1
        downscaled_width = floor((width - 1) / ratio) + 1
        return self.__class__(self._image.resize((downscaled_width, downscaled_height), resample = resample))
        
    def upscale(self, ratio, method = 'bilinear'):
        if method == 'bicubic':
            resample = PIL.Image.BICUBIC
        elif method == 'bilinear':
            resample = PIL.Image.BILINEAR
        elif method == 'lanczos':
            resample = PIL.Image.LANCZOS
        elif method == 'nearest':
            resample = PIL.Image.NEAREST
        else:
            raise ValueError('Unknown resampling method "{0}"'.format(method))

        width, height = self.shape
        upscaled_height = (height - 1) * ratio + 1
        upscaled_width = (width - 1) * ratio + 1
        return self.__class__(self._image.resize((upscaled_width, upscaled_height), resample = resample))
    
    def export(self, fname):
        # TODO: Make this work also for other color modes
        self._image.save(fname)