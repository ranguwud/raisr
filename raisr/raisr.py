# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse.csgraph, scipy.special
import pickle
import matplotlib.pyplot as plt
import PIL
from math import floor
from .image import Image
from .helper import select_pbar_cls, make_slice_list, gaussian2d

class RAISR:
    def __init__(self, *, ratio = 2, patchsize = 11, gradientsize = 9,
                 angle_bins = 24, strength_bins = 3, coherence_bins = 3,
                 pbar_cls = None):
        self._ratio = ratio
        self._patchsize = patchsize
        self._gradientsize = gradientsize
        self._angle_bins = angle_bins
        self._strength_thresholds = (10., 40.)
        self._coherence_thresholds = (0.25, 0.5)
        
        self._gradient_weight = gaussian2d(self.gradientsize).ravel()
        
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
        return max(self.patchsize // 2, self.gradientsize // 2 + 1)
    
    @property
    def angle_bins(self):
        return self._angle_bins
    
    @property
    def strength_thresholds(self):
        return self._strength_thresholds
    
    @property
    def strength_bins(self):
        return len(self._strength_thresholds) + 1
    
    def strength_from_log_normal(self, mu, sigma, bins = None):
        if bins is None:
            bins = self.strength_bins
            
        probability_thresholds = [p / bins for p in range(1, bins)]
        standard_normal_thresholds = \
            [scipy.special.ndtri(p) for p in probability_thresholds]
        self._strength_thresholds = \
            [np.exp(t * sigma + mu) for t in standard_normal_thresholds]
    
    @property
    def coherence_thresholds(self):
        return self._coherence_thresholds
    
    @property
    def coherence_bins(self):
        return len(self._coherence_thresholds) + 1
    
    def coherence_from_beta(self, alpha, beta, bins = None):
        if bins is None:
            bins = self.coherence_bins
        
        probability_thresholds = [p / bins for p in range(1, bins)]
        self._coherence_thresholds = \
            [scipy.special.betaincinv(alpha, beta, p) for p in probability_thresholds]

    
    def learn_filters(self, file, downscale_method = 'bicubic', upscale_method = 'bilinear'):
        img_original = Image.from_file(file).to_grayscale()
        shape = img_original.shape
        box = (0, 0, shape[0] - (shape[0]-1) % self.ratio, shape[1] - (shape[1]-1) % self.ratio)
        img_original = img_original.crop(box)
        img_low_res = img_original.downscale(self.ratio, method = downscale_method)
        img_high_res = img_low_res.upscale(self.ratio, method = upscale_method)
        assert img_original.shape == img_high_res.shape
        
        target_pixel_number = img_original.shape[0] * img_original.shape[1]
        target_pixel_number *= self.ratio ** 2
        
        if target_pixel_number > PIL.Image.MAX_IMAGE_PIXELS:
            pil_max_image_pixels = PIL.Image.MAX_IMAGE_PIXELS
            PIL.Image.MAX_IMAGE_PIXELS = None
        else:
            pil_max_image_pixels = None

        pbar_kwargs = self._make_pbar_kwargs(total = img_high_res.number_of_pixels(margin = self.margin),
                                             desc = "Learning")
        it = zip(img_high_res.lines(margin = self.margin), img_original.lines(margin = self.margin))
        with self._pbar_cls(**pbar_kwargs) as pbar:
            for img_high_res_line, img_original_line in it:
                patch_line = img_high_res_line.to_array(margin = self.patchsize // 2)
                original_line = img_original_line.to_array(margin = 0).ravel()
                # Calculate hashkey
                angle, strength, coherence = \
                    img_high_res_line.hashkey(self.gradientsize // 2 + 1,
                                              self._gradient_weight,
                                              self.angle_bins,
                                              self.strength_thresholds,
                                              self.coherence_thresholds)
                # Get pixel type
                pixeltype = img_high_res_line.pixeltype(self.ratio)

                # Compute A'A and A'b
                ATA, ATb = self.linear_regression_matrices(patch_line, original_line)
                # Compute Q and V
                for i in range(len(original_line)):
                    self._Q[angle[i],strength[i],coherence[i],pixeltype[i]] += ATA[i, ...]
                    self._V[angle[i],strength[i],coherence[i],pixeltype[i]] += ATb[i, ...]
                    #mark[coherence*3+strength, angle, pixeltype] += 1
                pbar.update(len(original_line))
    
        if not pil_max_image_pixels is None:
            PIL.Image.MAX_IMAGE_PIXELS = pil_max_image_pixels
        
    def get_image_statistics(self, file):
        img = Image.from_file(file).to_grayscale()
        
        angles = np.zeros((img.shape[1]-2*self.margin, img.shape[0]-2*self.margin))
        strengths = np.zeros((img.shape[1]-2*self.margin, img.shape[0]-2*self.margin))
        coherences = np.zeros((img.shape[1]-2*self.margin, img.shape[0]-2*self.margin))
        
        target_pixel_number = img.shape[0] * img.shape[1]
        target_pixel_number *= self.ratio ** 2
        
        if target_pixel_number > PIL.Image.MAX_IMAGE_PIXELS:
            pil_max_image_pixels = PIL.Image.MAX_IMAGE_PIXELS
            PIL.Image.MAX_IMAGE_PIXELS = None
        else:
            pil_max_image_pixels = None

        pbar_kwargs = self._make_pbar_kwargs(total = img.number_of_pixels(margin = self.margin),
                                             desc = "Analyzing")
        with self._pbar_cls(**pbar_kwargs) as pbar:
            for line in img.lines(margin = self.margin):
                # Calculate hashkey
                angle, strength, coherence = \
                    line.pixel_statistics(self.gradientsize // 2 + 1,
                                          self._gradient_weight)
                angles[line.lineno - self.margin, :] = angle
                strengths[line.lineno - self.margin, :] = strength
                coherences[line.lineno - self.margin, :] = coherence
                pbar.update(img.shape[0] - 2*self.margin)
                
        angles_permuted = np.concatenate((angles.ravel(),
                                          np.pi - angles.ravel(),
                                          (angles.ravel() + np.pi / 2) % np.pi,
                                          np.pi - (angles.ravel ()+ np.pi / 2) % np.pi))
                
        if not pil_max_image_pixels is None:
            PIL.Image.MAX_IMAGE_PIXELS = pil_max_image_pixels
        
        return angles_permuted, strengths.ravel(), coherences.ravel()
    
    def show_image_statistics(self, file, update_thresholds = False):
        angles, strengths, coherences = self.get_image_statistics(file)
        # TODO: This throws away all pixels in flat regions. Fix or document.
        strengths = strengths[strengths > 0]
        
        # MLE for strength
        log_strengths_mu = np.mean(np.log(strengths))
        log_strengths_sigma = np.std(np.log(strengths))
        log_strengths_range = np.linspace(np.min(np.log(strengths)),
                                          np.max(np.log(strengths)),
                                          num = 100)
        log_strengths_pdf = np.exp(-(log_strengths_range - log_strengths_mu)**2 / 
                                    (2 * log_strengths_sigma**2))
        log_strengths_pdf /= np.sqrt(2*np.pi) * log_strengths_sigma
        log_strengths_str = '$\\mu = {0}$\n$\\sigma = {1}$'.format(round(log_strengths_mu, 2),
                                                                   round(log_strengths_sigma, 2))
        
        # Method of moments for coherence
        coherence_mean = np.mean(coherences)
        coherence_var = np.var(coherences, ddof = 1)
        coherence_alpha = coherence_mean * (1 - coherence_mean)
        coherence_alpha /= coherence_var
        coherence_alpha -= 1
        coherence_alpha *= coherence_mean
        coherence_beta = coherence_mean * (1 - coherence_mean)
        coherence_beta /= coherence_var
        coherence_beta -= 1
        coherence_beta *= (1 - coherence_mean)
        
        coherence_range = np.linspace(0, 1, num = 100)
        coherence_pdf = np.power(coherence_range, coherence_alpha - 1)
        coherence_pdf *= np.power(1 - coherence_range, coherence_beta - 1)
        coherence_pdf /= scipy.special.beta(coherence_alpha, coherence_beta)
        coherence_str = '$\\alpha = {0}$\n$\\beta = {1}$'.format(round(coherence_alpha, 2),
                                                                 round(coherence_beta, 2))
        
        fig, axes = plt.subplots(nrows = 3, ncols = 1)
        axes[0].hist(angles / np.pi * 180, bins = 50, normed = True)
        axes[0].set_xlabel("Angle")
        axes[1].hist(np.log(strengths), bins = 50, normed = True)
        axes[1].plot(log_strengths_range, log_strengths_pdf, lw = 2)
        axes[1].set_xlabel("Log(strength)")
        axes[1].text(0.95, 0.9, log_strengths_str, 
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=axes[1].transAxes)
        axes[2].hist(coherences, bins = 50, normed = True)
        axes[2].plot(coherence_range, coherence_pdf, lw = 2)
        axes[2].set_xlabel("Coherence")
        axes[2].text(0.95, 0.9, coherence_str, 
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=axes[2].transAxes)
        fig.tight_layout()
        plt.show()
        
        if update_thresholds:
            self.strength_from_log_normal(log_strengths_mu, log_strengths_sigma)
            self.coherence_from_beta(coherence_alpha, coherence_beta)
            
    def permute_bins(self):
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
        pbar_kwargs = self._make_pbar_kwargs(total = self.ratio * self.ratio * self.angle_bins * self.strength_bins * self.coherence_bins * 7,
                                             desc = "Permuting")
        with self._pbar_cls(**pbar_kwargs) as pbar:
            for pixeltype in range(0, self.ratio*self.ratio):
                for angle in range(0, self.angle_bins):
                    for strength in range(0, self.strength_bins):
                        for coherence in range(0, self.coherence_bins):
                            for m in range(1, 8):
                                pbar.update()
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
                
    def linear_regression_matrices(self, patch_line_uint8, pixel_line_uint8):
        patch_line = patch_line_uint8.astype('float')
        pixel_line = pixel_line_uint8.astype('float')   
        patchsize = self.patchsize
        # Decompose patch into list of quadratic pieces
        start = 0
        stop = patch_line.shape[1] - 2 * self.margin
        slice_list = make_slice_list(start, stop, patchsize)
        patch_list = np.array([patch_line[..., sl] for sl in slice_list]).reshape(stop, (patchsize * patchsize))
        
        ATA_list = np.einsum('ij,ik->ijk', patch_list, patch_list)
        ATb_list = patch_list * pixel_line[:, None]
        return ATA_list, ATb_list
    
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
    
    def upscale(self, file, show = False, blending = None, fuzzyness = 0.01, method = 'bilinear'):
        img_original_ycbcr = Image.from_file(file).to_ycbcr()
        img_original_grey = img_original_ycbcr.to_grayscale()
        
        target_pixel_number = img_original_grey.shape[0] * img_original_grey.shape[1]
        target_pixel_number *= self.ratio ** 2
        
        if target_pixel_number > PIL.Image.MAX_IMAGE_PIXELS:
            pil_max_image_pixels = PIL.Image.MAX_IMAGE_PIXELS
            PIL.Image.MAX_IMAGE_PIXELS = None
        else:
            pil_max_image_pixels = None
        
        img_cheap_upscaled_ycbcr = img_original_ycbcr.upscale(self.ratio, method = method)
        img_cheap_upscaled_grey = img_cheap_upscaled_ycbcr.to_grayscale()
        
        width, height = img_cheap_upscaled_grey.shape
        patchsize = self.patchsize
        sisr = np.zeros((height - 2*self.margin, width - 2*self.margin), dtype = 'uint8')
        
        pbar_kwargs = self._make_pbar_kwargs(total = img_cheap_upscaled_grey.number_of_pixels(margin = self.margin),
                                             desc = "Upscaling")
        with self._pbar_cls(**pbar_kwargs) as pbar:
            for line in img_cheap_upscaled_grey.lines(margin = self.margin):
                patch_line = line.to_array(margin = self.patchsize // 2)
                # Calculate hashkey
                angle, strength, coherence = \
                    line.hashkey(self.gradientsize // 2 + 1,
                                 self._gradient_weight,
                                 self.angle_bins,
                                 self.strength_thresholds,
                                 self.coherence_thresholds)
                # Get pixel type
                pixeltype = line.pixeltype(self.ratio)

                # Decompose patch into list of quadratic pieces
                start = 0
                stop = patch_line.shape[1] - 2 * self.margin
                # TODO: Do not compute this anew every time
                slice_list = make_slice_list(start, stop, patchsize)
                patch_list = np.array([patch_line[..., sl] for sl in slice_list]).reshape(stop, (patchsize * patchsize))
                h_list = self._h[angle,strength,coherence,pixeltype,:]
                result = np.einsum('ij,ij->i', patch_list, h_list).round()
                
                #TODO: Do not just cut off, but use cheap upscaled pixels instead
                result[result <   0] =   0
                result[result > 255] = 255
                sisr[line.lineno - self.margin] = result.astype('uint8')
                pbar.update(stop)

        img_filtered_grey = img_cheap_upscaled_ycbcr.to_grayscale()
        # TODO: Use patch or similar to perform this assignment
        img_filtered_grey_data = np.array(img_filtered_grey._image)
        # TODO: Round to intergers before or after blending?
        img_filtered_grey_data[self.margin:height-self.margin,self.margin:width-self.margin] = np.round(sisr)
        img_filtered_grey = Image.from_array(img_filtered_grey_data)
        
        if blending == 'hamming':
            weight_table = np.zeros((256, 256))
            for ct_upscaled in range(256):
                for ct_filtered in range(256):
                    changed = bin(ct_upscaled ^ ct_filtered).count('1')
                    # TODO: Find best weights for blending
                    weight_table[ct_upscaled, ct_filtered] = np.sqrt(1. - (1. - 0.125*changed)**2)
            
            pbar_kwargs = self._make_pbar_kwargs(total = img_cheap_upscaled_grey.number_of_pixels(margin = self.margin),
                                                 desc = "Blending")
            it = zip(img_cheap_upscaled_grey.lines(margin = self.margin), img_filtered_grey.lines(margin = self.margin))
            with self._pbar_cls(**pbar_kwargs) as pbar:
                for img_cheap_upscaled_line, img_filtered_line in it:
                    ct_upscaled = img_cheap_upscaled_line.census_transform(fuzzyness = fuzzyness)
                    ct_filtered = img_filtered_line.census_transform(fuzzyness = fuzzyness)
                    weight = weight_table[ct_upscaled, ct_filtered]
                    
                    blended = img_filtered_line.to_array().astype('float') * (1 - weight)
                    blended += img_cheap_upscaled_line.to_array().astype('float') * weight
                    img_filtered_grey_data[img_cheap_upscaled_line.lineno,
                                           self.margin:-self.margin] = \
                        blended.round().astype('uint8')
                    pbar.update(len(ct_upscaled))
        
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
                for lines in img_cheap_upscaled_grey.lines(margin = self.margin):
                    ct_greater = line.census_transform(operator = np.greater, fuzzyness = fuzzyness)
                    ct_less    = line.census_transform(operator = np.less, fuzzyness = fuzzyness)
                    weight = weight_table[ct_less, ct_greater]

                    blended = line.to_array().astype('float') * (1 - weight)
                    blended += line.to_array().astype('float') * weight
                    # TODO: This causes rounding errors
                    img_filtered_grey_data[line.lineno,
                                           self.margin:-self.margin] = \
                        blended.round().astype('uint8')
                    pbar.update(len(ct_greater))
                
        
#        plt.imshow(img_filtered_grey_data[self.margin:height-self.margin,self.margin:width-self.margin] - sisr, interpolation = 'none',
#                   vmin = -25, vmax = 25, cmap=plt.cm.seismic)
#        plt.show()
#
#        plt.imshow(img_filtered_grey_data.astype('float') - np.array(img_cheap_upscaled_grey._image), interpolation = 'none',
#                   vmin = -25, vmax = 25, cmap=plt.cm.seismic)
#        plt.show()

        img_result_y = Image.from_array(img_filtered_grey_data)
        img_result_cb = img_cheap_upscaled_ycbcr.getchannel('Cb')
        img_result_cr = img_cheap_upscaled_ycbcr.getchannel('Cr')
        img_result = Image.from_channels('YCbCr', (img_result_y, img_result_cb, img_result_cr))

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(1, 3, 1)
            ax.imshow(img_original_grey._data, cmap='gray', interpolation='none')
            ax = fig.add_subplot(1, 3, 2)
            ax.imshow(img_cheap_upscaled_grey._data, cmap='gray', interpolation='none')
            ax = fig.add_subplot(1, 3, 3)
            ax.imshow(img_cheap_upscaled_ycbcr._data[:,:,0], cmap='gray', interpolation='none')
            plt.show()
            
        if not pil_max_image_pixels is None:
            PIL.Image.MAX_IMAGE_PIXELS = pil_max_image_pixels
        
        return img_result.to_rgb()

    def dump_filter(self, fname = "filter.pkl"):
        with open(fname, "wb") as f:
            pickle.dump(self._h, f)
    
    def load_filter(self, fname = "filter.pkl"):
        # TODO: Add check for dimensions of h
        with open(fname, "rb") as f:
            self._h = pickle.load(f)
                
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
        if self._pbar_cls.__name__ == 'tqdm':
            kwargs['bar_format'] =  '{desc}: {percentage:3.0f}%|{bar}| [{elapsed} elapsed/{remaining} remaining]'
        if self._pbar_cls.__name__ == 'tqdm_notebook':
            kwargs['bar_format'] =  '{n}/|/ [{elapsed} elapsed/{remaining} remaining]'
        return kwargs