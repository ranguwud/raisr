import cv2
import numpy as np
import os
import pickle
from scipy.misc import imresize
from cgls import cgls
from filterplot import filterplot
from gaussian2d import gaussian2d
from hashkey import hashkey
from math import floor
from matplotlib import pyplot as plt
from scipy import interpolate
import skimage.transform
from math import atan2, floor, pi

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

    def load_grayscale_image(self, file):
        rgb = cv2.imread(file)
        # Extract only the luminance in YCbCr
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCrCb)[:,:,0]
        # Normalize to [0,1]
        return cv2.normalize(gray.astype('float'), None, gray.min()/255, gray.max()/255, cv2.NORM_MINMAX)
    
    def downscale(self, high_res):
        height, width = high_res.shape
        #TODO: This method is deprecated.
        return imresize(high_res, (floor((height+1)/self.ratio),floor((width+1)/self.ratio)), interp='bicubic', mode='F')
        #return cv2.resize(high_res, dsize = (floor((width+1)/self.ratio),floor((height+1)/self.ratio)), interpolation = cv2.INTER_CUBIC)
        #return skimage.transform.resize(high_res, (floor((height+1)/self.ratio),floor((width+1)/self.ratio)),
        #                                order = 3)
        
    def cheap_interpolate(self, low_res):
        height, width = low_res.shape
        vert_grid = np.linspace(0, height - 1, height)
        horz_grid = np.linspace(0, width - 1, width)
        bilinear_interp = interpolate.interp2d(horz_grid, vert_grid, low_res, kind='linear')
        vert_grid = np.linspace(0, height - 1, height * self.ratio - 1)
        horz_grid = np.linspace(0, width - 1, width * self.ratio - 1)
        return bilinear_interp(horz_grid, vert_grid)
    
    def learn_filters(self, file):
        img_original = self.load_grayscale_image(file)
        img_low_res = self.downscale(img_original)
        img_high_res = self.cheap_interpolate(img_low_res)
        
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


# Define parameters
R = 2
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3
trainpath = 'train'

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)

Q = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize, patchsize*patchsize))
V = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize))
h = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize))
mark = np.zeros((Qstrength*Qcoherence, Qangle, R*R))

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

# Get image list
imagelist = []
for parent, dirnames, filenames in os.walk(trainpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

# Compute Q and V
imagecount = 1
for image in imagelist:
    print('\r', end='')
    print(' ' * 60, end='')
    print('\rProcessing image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + image + ')')
    origin = cv2.imread(image)
    # Extract only the luminance in YCbCr
    grayorigin = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)[:,:,0]
    # Normalized to [0,1]
    grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min()/255, grayorigin.max()/255, cv2.NORM_MINMAX)
    # Downscale (bicubic interpolation)
    height, width = grayorigin.shape
    LR = imresize(grayorigin, (floor((height+1)/2),floor((width+1)/2)), interp='bicubic', mode='F')
    # Upscale (bilinear interpolation)
    height, width = LR.shape
    heightgrid = np.linspace(0, height-1, height)
    widthgrid = np.linspace(0, width-1, width)
    bilinearinterp = interpolate.interp2d(widthgrid, heightgrid, LR, kind='linear')
    heightgrid = np.linspace(0, height-1, height*2-1)
    widthgrid = np.linspace(0, width-1, width*2-1)
    upscaledLR = bilinearinterp(widthgrid, heightgrid)
    # Calculate A'A, A'b and push them into Q, V
    height, width = upscaledLR.shape
    operationcount = 0
    totaloperations = (height-2*margin) * (width-2*margin)
    for row in range(margin, height-margin):
        for col in range(margin, width-margin):
            if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                print('\r|', end='')
                print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
            operationcount += 1
            # Get patch
            patch = upscaledLR[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
            patch = np.matrix(patch.ravel())
            # Get gradient block
            gradientblock = upscaledLR[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
            # Calculate hashkey
            angle, strength, coherence = hashkey(gradientblock, Qangle, weighting)
            # Get pixel type
            pixeltype = ((row-margin) % R) * R + ((col-margin) % R)
            # Get corresponding HR pixel
            pixelHR = grayorigin[row,col]
            # Compute A'A and A'b
            ATA = np.dot(patch.T, patch)
            ATb = np.dot(patch.T, pixelHR)
            ATb = np.array(ATb).ravel()
            # Compute Q and V
            Q[angle,strength,coherence,pixeltype] += ATA
            V[angle,strength,coherence,pixeltype] += ATb
            mark[coherence*3+strength, angle, pixeltype] += 1
    imagecount += 1

# Preprocessing permutation matrices P for nearly-free 8x more learning examples
print('\r', end='')
print(' ' * 60, end='')
print('\rPreprocessing permutation matrices P for nearly-free 8x more learning examples ...')
P = np.zeros((patchsize*patchsize, patchsize*patchsize, 7))
rotate = np.zeros((patchsize*patchsize, patchsize*patchsize))
flip = np.zeros((patchsize*patchsize, patchsize*patchsize))
for i in range(0, patchsize*patchsize):
    i1 = i % patchsize
    i2 = floor(i / patchsize)
    j = patchsize * patchsize - patchsize + i2 - patchsize * i1
    rotate[j,i] = 1
    k = patchsize * (i2 + 1) - i1 - 1
    flip[k,i] = 1
for i in range(1, 8):
    i1 = i % 4
    i2 = floor(i / 4)
    P[:,:,i-1] = np.linalg.matrix_power(flip,i2).dot(np.linalg.matrix_power(rotate,i1))
Qextended = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize, patchsize*patchsize))
Vextended = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize))
for pixeltype in range(0, R*R):
    for angle in range(0, Qangle):
        for strength in range(0, Qstrength):
            for coherence in range(0, Qcoherence):
                for m in range(1, 8):
                    m1 = m % 4
                    m2 = floor(m / 4)
                    newangleslot = angle
                    if m2 == 1:
                        newangleslot = Qangle-angle-1
                    newangleslot = int(newangleslot-Qangle/2*m1)
                    while newangleslot < 0:
                        newangleslot += Qangle
                    newQ = P[:,:,m-1].T.dot(Q[angle,strength,coherence,pixeltype]).dot(P[:,:,m-1])
                    newV = P[:,:,m-1].T.dot(V[angle,strength,coherence,pixeltype])
                    Qextended[newangleslot,strength,coherence,pixeltype] += newQ
                    Vextended[newangleslot,strength,coherence,pixeltype] += newV
Q += Qextended
V += Vextended

# Compute filter h
print('Computing h ...')
operationcount = 0
totaloperations = R * R * Qangle * Qstrength * Qcoherence
for pixeltype in range(0, R*R):
    for angle in range(0, Qangle):
        for strength in range(0, Qstrength):
            for coherence in range(0, Qcoherence):
                if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                    print('\r|', end='')
                    print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                    print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                    print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
                operationcount += 1
                h[angle,strength,coherence,pixeltype] = cgls(Q[angle,strength,coherence,pixeltype], V[angle,strength,coherence,pixeltype])

# Write filter to file
with open("filter", "wb") as fp:
    pickle.dump(h, fp)

# Uncomment the following line to show the learned filters
# filterplot(h, R, Qangle, Qstrength, Qcoherence, patchsize)

print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
