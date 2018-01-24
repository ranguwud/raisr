# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pickle
from scipy.misc import imresize
from cgls import cgls
from gaussian2d import gaussian2d
from scipy import interpolate
from math import atan2, floor, pi

class Image:
    def __init__(self, data):
        self._data = data
    
    def __getitem__(self, indices):
        return self._data[indices]
    
    @property
    def shape(self):
        return self._data.shape
    
    def to_grayscale(self):
        gray_data = cv2.cvtColor(self._data, cv2.COLOR_BGR2YCrCb)[:,:,0]
        gray_data_normalized = cv2.normalize(gray_data.astype('float'), None, 
                                             gray_data.min() / 255, gray_data.max() / 255,
                                             cv2.NORM_MINMAX)
        return Image(gray_data_normalized)

    def downscale(self, ratio):
        height, width = self._data.shape
        downscaled_height = floor((height+1)/ratio)
        downscaled_width = floor((width+1)/ratio)
        #TODO: This method is deprecated.
        return Image(imresize(self._data, (downscaled_height, downscaled_width), interp='bicubic', mode='F'))
        #return cv2.resize(high_res, dsize = (floor((width+1)/self.ratio),floor((height+1)/self.ratio)), interpolation = cv2.INTER_CUBIC)
        #return skimage.transform.resize(high_res, (floor((height+1)/self.ratio),floor((width+1)/self.ratio)),
        #                                order = 3)
        
    def cheap_interpolate(self, ratio):
        height, width = self._data.shape
        vert_grid = np.linspace(0, height - 1, height)
        horz_grid = np.linspace(0, width - 1, width)
        bilinear_interp = interpolate.interp2d(horz_grid, vert_grid, self._data, kind='linear')
        vert_grid = np.linspace(0, height - 1, height * ratio - 1)
        horz_grid = np.linspace(0, width - 1, width * ratio - 1)
        return Image(bilinear_interp(horz_grid, vert_grid))
    
class ImageFile(Image):
    def __init__(self, fname):
        super().__init__(cv2.imread(fname))

class RAISR:
    def __init__(self, *, ratio = 2, patchsize = 11, gradientsize = 9,
                 angle_bins = 24, strength_bins = 3, coherence_bins = 3):
        self._ratio = ratio
        self._patchsize = patchsize
        self._gradientsize = gradientsize
        self._angle_bins = angle_bins
        self._strength_bins = strength_bins
        self._coherence_bins = coherence_bins
        
        self._weighting = np.diag(gaussian2d([gradientsize, gradientsize], 2).ravel())

        self._Q = np.zeros((angle_bins, strength_bins, coherence_bins, ratio * ratio,
                            patchsize * patchsize, patchsize * patchsize))
        self._V = np.zeros((angle_bins, strength_bins, coherence_bins, ratio * ratio,
                            patchsize * patchsize))
        self._h = np.zeros((angle_bins, strength_bins, coherence_bins, ratio * ratio,
                            patchsize * patchsize))
    
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
        img_original = ImageFile(file).to_grayscale()
        img_low_res = img_original.downscale(self.ratio)
        img_high_res = img_low_res.cheap_interpolate(self.ratio)
        
        height, width = img_high_res.shape
        patchmargin = floor(self.patchsize / 2)
        gradientmargin = floor(self.gradientsize / 2)
        
        operationcount = 0
        totaloperations = (height-2*self.margin) * (width-2*self.margin)
        for row in range(self.margin, height - self.margin):
            for col in range(self.margin, width - self.margin):
                if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                    print('\r|', end='')
                    print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                    print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                    print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
                operationcount += 1
                # Get patch
                patch = img_high_res[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
                patch = np.matrix(patch.ravel())
                # Get gradient block
                gradientblock = img_high_res[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
                # Calculate hashkey
                angle, strength, coherence = self.hashkey(gradientblock)
                # Get pixel type
                pixeltype = self.pixeltype(row-self.margin, col-self.margin)
                # Compute A'A and A'b
                ATA, ATb = self.linear_regression_matrices(patch, img_original[row,col])
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
                        self._h[angle,strength,coherence,pixeltype] = cgls(Q[angle,strength,coherence,pixeltype], V[angle,strength,coherence,pixeltype])

    def dump_filter(self, fname = "filter.pkl"):
        with open(fname, "wb") as f:
            pickle.dump(self._h, f)

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
