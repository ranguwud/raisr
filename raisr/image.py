# -*- coding: utf-8 -*-
import PIL
import numpy as np
import weakref
from math import floor
from .helper import make_slice_list


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
    
    def pixeltype(self, ratio):
        pixel_numbers = np.arange(self.margin, self._image.size[0] - self.margin)
        return ((self.lineno) % ratio) * ratio + ((pixel_numbers) % ratio)

    def census_transform(self, operator = np.greater, fuzzyness = 0.0):
        block = self.to_array(margin = 1)
        tl = block[0, 0:-2]
        tc = block[0, 1:-1]
        tr = block[0, 2:]
        cl = block[1, 0:-2]
        cc = block[1, 1:-1]
        cr = block[1, 2:]
        bl = block[2, 0:-2]
        bc = block[2, 1:-1]
        br = block[2, 2:]
        
        pixel_stack = np.vstack((tl, cl, bl, tc, bc, tr, cr, br))
        comp_hi = operator(cc[None,:] + fuzzyness, pixel_stack)
        comp_lo = operator(cc[None,:] - fuzzyness, pixel_stack)
        
        bools = np.all((comp_hi, comp_lo), axis = 0)
        byte_vector = np.array((128, 64, 32, 16, 8, 4, 2, 1))
        
        return np.sum(bools * byte_vector[:, None], axis = 0)
    
    def pixel_statistics(self, margin, gradient_weight):
        # Calculate gradient of input block
        block = self.to_array(margin = margin).astype('float')
        gy, gx = np.gradient(block)
        gradientsize = 2 * margin - 1
        
        # Decompose gradient into list of quadratic pieces
        start = 1
        stop = block.shape[1] - gradientsize
        # TODO: Do not compute this anew every time
        slice_list = make_slice_list(start, stop, gradientsize)
        gy_list = np.array([gy[..., 1:-1, sl] for sl in slice_list])
        gx_list = np.array([gx[..., 1:-1, sl] for sl in slice_list])
        gy_lines = gy_list.reshape((gy_list.shape[0], gy_list.shape[1] * gy_list.shape[2]))
        gx_lines = gx_list.reshape((gx_list.shape[0], gx_list.shape[1] * gx_list.shape[2]))
        
        # Get list of corresponding matrices G, G^T and W
        G_list = np.copy(np.array([gx_lines, gy_lines]).transpose((1,2,0)))
        GT_list = np.copy(G_list.transpose((0,2,1)))
        
        # Calculate list of G^T * W * G matrix products
        GTWG_list = np.einsum('ijk,ikl->ijl', GT_list,
                              gradient_weight[None, :, None] * G_list,
                              optimize = True)
        
        # Extract lists of individual matrix entries by writing
        #                / a  b \
        # G^T * W * G = |       |
        #               \ c  d /
        a_list = GTWG_list[:, 0, 0]
        b_list = GTWG_list[:, 0, 1]
        c_list = GTWG_list[:, 1, 0]
        d_list = GTWG_list[:, 1, 1]
        
        # Calculate lists of determinants and traces using general formula
        # for 2-by-2 matrices
        det_list = a_list * d_list - b_list * c_list
        tr_list = a_list + d_list
        
        # Calculate maximum and minimum eigenvalue using general formula
        # for 2-by-2 matrices
        sqrt_list = np.sqrt(tr_list**2 / 4 - det_list)
        sqrt_list[np.isnan(sqrt_list)] = 0
        eig_max_list = tr_list / 2 + sqrt_list
        eig_min_list = tr_list / 2 - sqrt_list
        
        # There exists no general closed form for the corresponding eigenvector.
        # Depending on whether c != 0 (case 1) or b != 0 (case 2) there are two
        # equivalent results
        v_list_1 = np.vstack((eig_max_list - d_list, c_list))
        v_list_2 = np.vstack((b_list, eig_max_list - a_list))
        
        # The results from the two cases are always correct, but it can happen
        # that the resulting vectors are zero, if c == 0 or b == 0, respectively.
        # Since G^T * W * G is symmetric, b == c holds true. So the two vectors
        # are of similar magnitude and adding them can help to rediuce numerical
        # noise. The following lines produce v_1 + v_2 or v_1 - v_2, respectively,
        # depending on which of the two sums has larger norm.
        # More importantly, this also resolves the not explicitly covered case
        # b == c == 0: If b*c is much smaller than a*d, then eig_max will be
        # approximately equal to max(a, d). This results in either v_1 or v_2
        # being approximately zero, while the respective other vector has length
        # of approximately abs(a - d). The only unhandled remaining case is
        # b == c == 0 and a == d, but then the corresponding eigenvector is
        # not well-defined anyway. Therefore, using the result v = v_1 ± v_2 is
        # sufficient.
        v_list_p = v_list_1 + v_list_2
        v_list_m = v_list_1 - v_list_2
        norm_list_p = v_list_p[0,:]**2 + v_list_p[1,:]**2
        norm_list_m = v_list_m[0,:]**2 + v_list_m[1,:]**2
        v_list = v_list_p * (norm_list_p > norm_list_m) + v_list_m * (norm_list_p <= norm_list_m)
        
        # Calculate theta
        theta_list = np.arctan2(v_list[1,:], v_list[0,:])
        theta_list[theta_list < 0] += np.pi
        
        # Calculate u
        sqrt_eig_max_list = np.sqrt(eig_max_list)
        sqrt_eig_min_list = np.sqrt(eig_min_list)
        u_list = (sqrt_eig_max_list - sqrt_eig_min_list) / (sqrt_eig_max_list + sqrt_eig_min_list)
        u_list[np.logical_not(np.isfinite(u_list))] = 0
        
        return theta_list, eig_max_list, u_list

    def hashkey(self, margin, gradient_weight, angle_bins, strength_thresholds, coherence_thresholds):
        theta_list, eig_max_list, u_list = self.pixel_statistics(margin, gradient_weight)
        
        # Quantize
        # TODO: Find optimal theshold values
        angle_list = (theta_list * angle_bins / np.pi).astype('uint')
        angle_list[angle_list == angle_bins] = 0
        
        strength_list = np.zeros(eig_max_list.shape, dtype = 'uint')
        for threshold in strength_thresholds:
            strength_list += (eig_max_list > threshold).astype('uint')
        
        coherence_list = np.zeros(eig_max_list.shape, dtype = 'uint')
        for threshold in coherence_thresholds:
            coherence_list += (u_list > threshold).astype('uint')
        
        return angle_list, strength_list, coherence_list


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
        if self.mode == 'RGB' or self.mode == 'RGBA':
            return self.to_ycbcr().to_grayscale()
        elif self.mode == 'YCbCr':
            return self.__class__(self._image.getchannel('Y'))
        else:
            raise ValueError('Expected RGB or YCbCr mode image.')
    
    def to_ycbcr(self):
        if self.mode == 'YCbCr':
            return self
        elif self.mode == 'RGB' or self.mode == 'RGBA':
            return self.__class__(self._image.convert('YCbCr'))
        else:
            raise ValueError('Expected RGB mode image.')
    
    def to_rgb(self):
        if self.mode == 'RGB':
            return self
        elif self.mode == 'YCbCr':
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