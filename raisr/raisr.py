# -*- coding: utf-8 -*-
import cv2
import numpy as np
import scipy.sparse.csgraph
import pickle
import matplotlib.pyplot as plt
import sys
import os
from math import atan2, floor, pi
from .image import Image

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


class RAISR:
    def __init__(self, *, ratio = 2, patchsize = 11, gradientsize = 9,
                 angle_bins = 24, strength_bins = 3, coherence_bins = 3,
                 pbar_cls = None):
        self._ratio = ratio
        self._patchsize = patchsize
        self._gradientsize = gradientsize
        self._angle_bins = angle_bins
        self._strength_bins = strength_bins
        self._coherence_bins = coherence_bins
        
        self._weighting = np.diag(RAISR.gaussian2d([gradientsize, gradientsize], 2).ravel())

        self._Q = np.zeros((angle_bins, strength_bins, coherence_bins, ratio * ratio,
                            patchsize * patchsize, patchsize * patchsize))
        self._V = np.zeros((angle_bins, strength_bins, coherence_bins, ratio * ratio,
                            patchsize * patchsize))
        self._h = np.zeros((angle_bins, strength_bins, coherence_bins, ratio * ratio,
                            patchsize * patchsize))
        
        if pbar_cls:
            self._pbar_cls = pbar_cls
        else:
            self._pbar_cls = select_pbar_cls()
    
    @staticmethod
    def gaussian2d(shape=(3,3),sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    
    @property
    def ratio(self):
        return self._ratio
    
    @property
    def patchsize(self):
        return self._patchsize
    
    @property
    def gradientsize(self):
        return self._gradientsize
    
    @property
    def margin(self):
        return floor(max(self.patchsize, self.gradientsize) / 2)
    
    @property
    def angle_bins(self):
        return self._angle_bins
    
    @property
    def strength_bins(self):
        return self._strength_bins
    
    @property
    def coherence_bins(self):
        return self._coherence_bins

    def learn_filters(self, file):
        img_original = Image.from_file(file).to_grayscale()
        img_low_res = img_original.downscale(self.ratio)
        img_high_res = img_low_res.cheap_interpolate(self.ratio)
        
        height, width = img_high_res.shape
        
        pbar_kwargs = self._make_pbar_kwargs(total = img_high_res.number_of_pixels(margin = self.margin),
                                             desc = "Learning")
        with self._pbar_cls(**pbar_kwargs) as pbar:
            for pixel in img_high_res.pixels(margin = self.margin):
                pbar.update()
                # Get patch
                patch = pixel.patch(self.patchsize).ravel().reshape(-1, self.patchsize**2)
                # Get gradient block
                gradientblock = pixel.patch(self.gradientsize)
                # Calculate hashkey
                angle, strength, coherence = self.hashkey(gradientblock)
                # Get pixel type
                pixeltype = self.pixeltype(pixel.row-self.margin, pixel.col-self.margin)
                # Compute A'A and A'b
                ATA, ATb = self.linear_regression_matrices(patch, img_original[pixel.row, pixel.col])
                # Compute Q and V
                self._Q[angle,strength,coherence,pixeltype] += ATA
                self._V[angle,strength,coherence,pixeltype] += ATb
                #mark[coherence*3+strength, angle, pixeltype] += 1
    
    def permute_bins(self):
        print('\r', end='')
        print(' ' * 60, end='')
        print('\rPreprocessing permutation matrices P for nearly-free 8x more learning examples ...')
        #TODO: What exactly is going on here?
        P = np.zeros((self.patchsize*self.patchsize, self.patchsize*self.patchsize, 7))
        rotate = np.zeros((self.patchsize*self.patchsize, self.patchsize*self.patchsize))
        flip = np.zeros((self.patchsize*self.patchsize, self.patchsize*self.patchsize))
        for i in range(0, self.patchsize*self.patchsize):
            i1 = i % self.patchsize
            i2 = floor(i / self.patchsize)
            j = self.patchsize * self.patchsize - self.patchsize + i2 - self.patchsize * i1
            rotate[j,i] = 1
            k = self.patchsize * (i2 + 1) - i1 - 1
            flip[k,i] = 1
        for i in range(1, 8):
            i1 = i % 4
            i2 = floor(i / 4)
            P[:,:,i-1] = np.linalg.matrix_power(flip,i2).dot(np.linalg.matrix_power(rotate,i1))
        Qextended = np.zeros((self.angle_bins, self.strength_bins, self.coherence_bins, self.ratio*self.ratio, self.patchsize*self.patchsize, self.patchsize*self.patchsize))
        Vextended = np.zeros((self.angle_bins, self.strength_bins, self.coherence_bins, self.ratio*self.ratio, self.patchsize*self.patchsize))
        for pixeltype in range(0, self.ratio*self.ratio):
            for angle in range(0, self.angle_bins):
                for strength in range(0, self.strength_bins):
                    for coherence in range(0, self.coherence_bins):
                        for m in range(1, 8):
                            m1 = m % 4
                            m2 = floor(m / 4)
                            newangleslot = angle
                            if m2 == 1:
                                newangleslot = self.angle_bins-angle-1
                            newangleslot = int(newangleslot-self.angle_bins/2*m1)
                            while newangleslot < 0:
                                newangleslot += self.angle_bins
                            newQ = P[:,:,m-1].T.dot(self._Q[angle,strength,coherence,pixeltype]).dot(P[:,:,m-1])
                            newV = P[:,:,m-1].T.dot(self._V[angle,strength,coherence,pixeltype])
                            Qextended[newangleslot,strength,coherence,pixeltype] += newQ
                            Vextended[newangleslot,strength,coherence,pixeltype] += newV
        self._Q += Qextended
        self._V += Vextended
                
    def pixeltype(self, row_index, col_index):
        return ((row_index) % self.ratio) * self.ratio + ((col_index) % self.ratio)
    
    def linear_regression_matrices(self, patch, pixel):
        ATA = np.matmul(patch.T, patch)
        ATb = (patch.T * pixel).ravel()
        return ATA, ATb
    
    def calculate_optimal_filter(self):
        pbar_kwargs = self._make_pbar_kwargs(total = self.ratio * self.ratio * self.angle_bins * self.strength_bins * self.coherence_bins,
                                             desc = "Computing h")
        with self._pbar_cls(**pbar_kwargs) as pbar:
            for pixeltype in range(0, self.ratio * self.ratio):
                for angle in range(0, self.angle_bins):
                    for strength in range(0, self.strength_bins):
                        for coherence in range(0, self.coherence_bins):
                            pbar.update()
                            h, _, _, _, = np.linalg.lstsq(self._Q[angle,strength,coherence,pixeltype], self._V[angle,strength,coherence,pixeltype], rcond = 1.e-6)
                            self._h[angle,strength,coherence,pixeltype] = h
    
    def upscale(self, file, show = False, blending = None, fuzzyness = 0.01):
        img_original_ycrcb = Image.from_file(file).to_ycrcb()
        img_original_grey = img_original_ycrcb.to_grayscale()
        
        img_cheap_upscaled_ycrcb = img_original_ycrcb.cheap_interpolate(self.ratio)
        img_cheap_upscaled_grey = img_original_grey.cheap_interpolate(self.ratio)
        
        height, width = img_cheap_upscaled_grey.shape
        sisr = np.zeros((height - 2*self.margin, width - 2*self.margin))
        
        pbar_kwargs = self._make_pbar_kwargs(total = img_cheap_upscaled_grey.number_of_pixels(margin = self.margin),
                                             desc = "Upscaling")
        with self._pbar_cls(**pbar_kwargs) as pbar:
            for pixel in img_cheap_upscaled_grey.pixels(margin = self.margin):
                pbar.update()
                patch = pixel.patch(self.patchsize).ravel()
                # Get gradient block
                gradientblock = pixel.patch(self.gradientsize)
                # Calculate hashkey
                angle, strength, coherence = self.hashkey(gradientblock)
                # Get pixel type
                pixeltype = self.pixeltype(pixel.row-self.margin, pixel.col-self.margin)
                
                sisr[pixel.row - self.margin, pixel.col - self.margin] = \
                    patch.dot(self._h[angle,strength,coherence,pixeltype])
        
        #TODO: Do not just cut off, but use cheap upscaled pixels instead
        sisr[sisr < 0] = 0.0
        sisr[sisr > 1] = 1.0

        img_filtered_grey = img_cheap_upscaled_ycrcb.to_grayscale()
        # TODO: Use patch or similar to perform this assignment
        img_filtered_grey._data[self.margin:height-self.margin,self.margin:width-self.margin] = sisr
        
        if blending == 'hamming':
            weight_table = np.zeros((256, 256))
            for ct_upscaled in range(256):
                for ct_filtered in range(256):
                    changed = bin(ct_upscaled ^ ct_filtered).count('1')
                    # TODO: Find best weights for blending
                    weight_table[ct_upscaled, ct_filtered] = np.sqrt(1. - (1. - 0.125*changed)**2)
            
            pbar_kwargs = self._make_pbar_kwargs(total = img_cheap_upscaled_grey.number_of_pixels(margin = self.margin),
                                                 desc = "Blending")
            with self._pbar_cls(**pbar_kwargs) as pbar:
                for pixel in img_cheap_upscaled_grey.pixels(margin = self.margin):
                    pbar.update()
    
                    ct_upscaled = img_cheap_upscaled_grey.census_transform(pixel.row, pixel.col, fuzzyness = fuzzyness)
                    ct_filtered = img_filtered_grey.census_transform(pixel.row, pixel.col, fuzzyness = fuzzyness)
                    weight = weight_table[ct_upscaled, ct_filtered]
                    img_filtered_grey._data[pixel.row, pixel.col] *= (1. - weight)
                    img_filtered_grey._data[pixel.row, pixel.col] += weight * pixel.value
        
        if blending == 'randomness':
            lcc_table = np.zeros(256)
            for ct in range(256):
                if ct == 0:
                    lcc = 0
                else:
                    a = np.array([[128, 16, 4], [64, 0, 2], [32, 8, 1]])
                    truths = (np.bitwise_and(a, ct) > 0).astype('float')
                    adjacency = np.diag(np.ones(9))
                    vert = (np.diff(truths, axis = 0) == 0)
                    for row in range(vert.shape[0]):
                        for col in range(vert.shape[1]):
                            if vert[row,col]:
                                adjacency[3*row+col,3*(row+1)+col] = 1
                                adjacency[3*(row+1)+col,3*row+col] = 1
                    horz = (np.diff(truths, axis = 1) == 0)
                    for row in range(horz.shape[0]):
                        for col in range(horz.shape[1]):
                            if horz[row,col]:
                                adjacency[3*row+col, 3*row+col+1] = 1
                                adjacency[3*row+col+1, 3*row+col] = 1
                    n, labels = scipy.sparse.csgraph.connected_components(adjacency, directed = False)
                    occurences = [list(labels).count(i) for i in range(n)]
                    occurences.sort()
                    lcc = occurences[0]
                lcc_table[ct] = lcc
                
            weight_table = np.zeros((256, 256))
            for ct_greater in range(256):
                for ct_less in range(256):
                    lcc = lcc_table[ct_greater | ct_less]
                    weight_table[ct_less, ct_greater] = np.sqrt(1. - (0.25*lcc)**2)
                    
            pbar_kwargs = self._make_pbar_kwargs(total = img_cheap_upscaled_grey.number_of_pixels(margin = self.margin),
                                                 desc = "Blending")
            with self._pbar_cls(**pbar_kwargs) as pbar:
                for pixel in img_cheap_upscaled_grey.pixels(margin = self.margin):
                    pbar.update()
    
                    ct_greater = img_cheap_upscaled_grey.census_transform(pixel.row, pixel.col, operator = np.greater, fuzzyness = fuzzyness)
                    ct_less = img_cheap_upscaled_grey.census_transform(pixel.row, pixel.col, operator = np.less, fuzzyness = fuzzyness)
                    weight = weight_table[ct_less, ct_greater]
                    img_filtered_grey._data[pixel.row, pixel.col] *= (1. - weight)
                    img_filtered_grey._data[pixel.row, pixel.col] += weight * pixel.value
                
        
#        plt.imshow(img_filtered_grey._data[self.margin:height-self.margin,self.margin:width-self.margin] - sisr, interpolation = 'none',
#                   vmin = -0.1, vmax = 0.1, cmap=plt.cm.seismic)
#        plt.show()
#
#        plt.imshow(img_filtered_grey._data - img_cheap_upscaled_grey._data, interpolation = 'none',
#                   vmin = -0.25, vmax = 0.25, cmap=plt.cm.seismic)
#        plt.show()

        # Scale back to [0,255]
        img_cheap_upscaled_ycrcb._data[:,:,0] = \
            cv2.normalize(img_filtered_grey._data.astype('float'), None, 0.0, 255.0, cv2.NORM_MINMAX)

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(1, 3, 1)
            ax.imshow(img_original_grey._data, cmap='gray', interpolation='none')
            ax = fig.add_subplot(1, 3, 2)
            ax.imshow(img_cheap_upscaled_grey._data, cmap='gray', interpolation='none')
            ax = fig.add_subplot(1, 3, 3)
            ax.imshow(img_cheap_upscaled_ycrcb._data[:,:,0], cmap='gray', interpolation='none')
            plt.show()
        
        return img_cheap_upscaled_ycrcb.to_rgb()

    def dump_filter(self, fname = "filter.pkl"):
        with open(fname, "wb") as f:
            pickle.dump(self._h, f)
    
    def load_filter(self, fname = "filter.pkl"):
        # TODO: Add check for dimensions of h
        with open(fname, "rb") as f:
            self._h = pickle.load(f)
            
    def hashkey(self, block):
        # Calculate gradient
        # TODO: This seems to be SLOW. Can it be replaced by differences in
        # x and y direction, respectively?
        gy, gx = np.gradient(block)
    
        # Transform 2D matrix into 1D array
        gx = gx.ravel()
        gy = gy.ravel()
    
        # SVD calculation
        G = np.vstack((gx,gy)).T
        # TODO: Is this the right way to do that product??
        GTWG = G.T.dot(self._weighting).dot(G)
        w, v = np.linalg.eigh(GTWG);
    
        # Sort w and v according to the descending order of w
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:,idx]
    
        # Calculate theta
        theta = atan2(v[1,0], v[0,0])
        if theta < 0:
            theta = theta + pi
    
        # Calculate lamda
        lamda = w[0]
    
        # Calculate u
        sqrtlamda1 = np.sqrt(w[0])
        sqrtlamda2 = np.sqrt(w[1])
        if sqrtlamda1 + sqrtlamda2 == 0:
            u = 0
        else:
            u = (sqrtlamda1 - sqrtlamda2)/(sqrtlamda1 + sqrtlamda2)
    
        # Quantize
        # TODO: Confirm these values for strength and coherence
        angle = floor(theta/pi*self._angle_bins)
        if lamda < 0.0001:
            strength = 0
        elif lamda > 0.001:
            strength = 2
        else:
            strength = 1
        if u < 0.25:
            coherence = 0
        elif u > 0.5:
            coherence = 2
        else:
            coherence = 1
    
        # Bound the output to the desired ranges
        if angle > 23:
            angle = 23
        elif angle < 0:
            angle = 0
    
        return angle, strength, coherence
    
    def filterplot(self):
        for pixeltype in range(0,self.ratio*self.ratio):
            maxvalue = self._h[:,:,:,pixeltype].max()
            minvalue = self._h[:,:,:,pixeltype].min()
            fig = plt.figure(pixeltype)
            plotcounter = 1
            for coherence in range(0, self.coherence_bins):
                for strength in range(0, self.strength_bins):
                    for angle in range(0, self.angle_bins):
                        filter1d = self._h[angle,strength,coherence,pixeltype]
                        filter2d = np.reshape(filter1d, (self.patchsize, self.patchsize))
                        ax = fig.add_subplot(self.strength_bins*self.coherence_bins, self.angle_bins, plotcounter)
                        ax.imshow(filter2d, interpolation='none', extent=[0,10,0,10], vmin=minvalue, vmax=maxvalue, cmap=plt.cm.seismic)
                        ax.axis('off')
                        plotcounter += 1
            plt.axis('off')
            plt.show()

    def _make_pbar_kwargs(self, total = 100, desc = ""):
        kwargs = {'total': total, 'desc': desc}
        if total > 1e6:
            kwargs['mininterval'] = 1
        else:
            kwargs['mininterval'] = 0.1
        if self._pbar_cls.__name__ == 'tqdm':
            kwargs['bar_format'] =  '{desc}: {percentage:3.0f}%|{bar}| [{elapsed} elapsed/{remaining} remaining]'
        if self._pbar_cls.__name__ == 'tqdm_notebook':
            kwargs['bar_format'] =  '{n}/|/ [{elapsed} elapsed/{remaining} remaining]'
        return kwargs