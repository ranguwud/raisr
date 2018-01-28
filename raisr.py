# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from cgls import cgls
from math import atan2, floor, pi
from image import Image


class RAISR:
    def __init__(self, *, ratio = 2, patchsize = 11, gradientsize = 9,
                 angle_bins = 24, strength_bins = 3, coherence_bins = 3):
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
        
        operationcount = 0
        totaloperations = img_high_res.number_of_pixels(margin = self.margin)
        for pixel in img_high_res.pixels(margin = self.margin):
            if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                print('\r|', end='')
                print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
            operationcount += 1
            # Get patch
            # TODO: This has size pathsize² * patchsize². Is that correct?
            patch = np.matrix(pixel.patch(self.patchsize).ravel())
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
        ATA = np.dot(patch.T, patch)
        ATb = np.dot(patch.T, pixel)
        ATb = np.array(ATb).ravel()
        return ATA, ATb
    
    def calculate_optimal_filter(self):
        print('Computing h ...')
        operationcount = 0
        totaloperations = self.ratio * self.ratio * self.angle_bins * self.strength_bins * self.coherence_bins
        for pixeltype in range(0, self.ratio * self.ratio):
            for angle in range(0, self.angle_bins):
                for strength in range(0, self.strength_bins):
                    for coherence in range(0, self.coherence_bins):
                        if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                            print('\r|', end='')
                            print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                            print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                            print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
                        operationcount += 1
                        # TODO: Check functionality of cgls function
                        self._h[angle,strength,coherence,pixeltype] = cgls(self._Q[angle,strength,coherence,pixeltype], self._V[angle,strength,coherence,pixeltype])
    
    def upscale(self, file, show = False):
        img_original_ycrcb = Image.from_file(file).to_ycrcb()
        img_original_grey = img_original_ycrcb.to_grayscale()
        
        img_cheap_upscaled_ycrcb = img_original_ycrcb.cheap_interpolate(self.ratio)
        img_cheap_upscaled_grey = img_original_grey.cheap_interpolate(self.ratio)
        
        height, width = img_cheap_upscaled_grey.shape
        sisr = np.zeros((height - 2*self.margin, width - 2*self.margin))
        operationcount = 0
        totaloperations = img_cheap_upscaled_grey.number_of_pixels(margin = self.margin)
        for pixel in img_cheap_upscaled_grey.pixels(margin = self.margin):
            if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                print('\r|', end='')
                print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
            operationcount += 1
            patch = pixel.patch(self.patchsize).ravel()
            # Get gradient block
            gradientblock = pixel.patch(self.gradientsize)
            # Calculate hashkey
            angle, strength, coherence = self.hashkey(gradientblock)
            # Get pixel type
            pixeltype = self.pixeltype(pixel.row-self.margin, pixel.col-self.margin)
            
            sisr[pixel.row - self.margin, pixel.col - self.margin] = \
                patch.dot(self._h[angle,strength,coherence,pixeltype])

        # Scale back to [0,255]
        sisr = cv2.normalize(sisr.astype('float'), None, 0, 255, cv2.NORM_MINMAX)
        # TODO: Use patch or similar to perform this assignment
        img_cheap_upscaled_ycrcb._data[self.margin:height-self.margin,self.margin:width-self.margin,0] = \
            sisr
        
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
        gy, gx = np.gradient(block)
    
        # Transform 2D matrix into 1D array
        gx = gx.ravel()
        gy = gy.ravel()
    
        # SVD calculation
        G = np.vstack((gx,gy)).T
        GTWG = G.T.dot(self._weighting).dot(G)
        # TODO: Use eigh instead of eig
        w, v = np.linalg.eig(GTWG);
    
        # Make sure V and D contain only real numbers
        # TODO: Remove check for reals, when using eigh above
        nonzerow = np.count_nonzero(np.isreal(w))
        nonzerov = np.count_nonzero(np.isreal(v))
        if nonzerow != 0:
            w = np.real(w)
        if nonzerov != 0:
            v = np.real(v)
    
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
                        ax.imshow(filter2d, interpolation='none', extent=[0,10,0,10], vmin=minvalue, vmax=maxvalue)
                        ax.axis('off')
                        plotcounter += 1
            plt.axis('off')
            plt.show()

