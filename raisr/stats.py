# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class StatsBase:
    def __init__(self):
        self._samples = 0
    
    @property
    def samples(self):
        return self._samples
    
    def append(self, samples):
        self._samples += len(samples)
    
    
class StatsArray(StatsBase):
    def __init__(self, bins = 50):
        super().__init__()
        self.bins = 50
        self._list = list()
    
    @property
    def bins(self):
        return self._bins
    
    @bins.setter
    def bins(self, value):
        if not int(value) > 0:
            raise ValueError("Number of bins must be positive.")
        self._bins = int(value)
    
    @property
    def array(self):
        return np.array(self._list).ravel()
    
    @property
    def min(self):
        return np.min(self.array)
    
    @property
    def max(self):
        return np.max(self.array)
    
    @property
    def mean(self):
        return np.mean(self.array)
    
    @property
    def log_mean(self):
        return np.mean(np.ma.masked_invalid(np.log(self.array)))
    
    @property
    def var(self):
        return np.var(self.array)
    
    @property
    def log_var(self):
        return np.var(np.ma.masked_invalid(np.log(self.array)))
    
    @property
    def sample_var(self):
        return np.var(self.array, ddof = 1)
    
    @property
    def log_sample_var(self):
        return np.var(np.ma.masked_invalid(np.log(self.array)), ddof = 1)
    
    def append(self, samples):
        super().append(samples)
        self._list.append(samples)
    
    def hist(self, axis = None):
        if axis is None:
            plot = plt
        else:
            plot = axis
        plot.hist(self.array, bins = self.bins, normed = True)
    
    def log_hist(self, axis = None):
        if axis is None:
            plot = plt
        else:
            plot = axis
        plot.hist(np.log(self.array), bins = self.bins, normed = True)


class StatsBaseMoments(StatsBase):
    def __init__(self, binning_thresholds):
        super().__init__(self)
        self.binning_thresholds = binning_thresholds
        self._occurrences = np.zeros(len(binning_thresholds) + 1)
        self._sum = 0.0
        self._sum_sq = 0.0
    
    @property
    def binning_thresholds(self):
        return self._binning_thresholds
    
    @binning_thresholds.setter
    def binning_thresholds(self, value):
        if not np.all(np.diff(value) > 0):
            raise ValueError("Thresholds must be strictly monotonically increasing")
        self._binning_thresholds = value
    
    @property
    def mean(self):
        return self._sum / self._samples
    
    @property
    def var(self):
        return self._sum_sq / self._samples - self.mean**2
    
    @property
    def sample_var(self):
        return self.var * self._samples / (self._samples - 1)
    
    def append(self, samples):
        self._samples += len(samples)
        bins = np.digitize(samples, self._binning_thresholds)
        self._occurrences += np.bincount(bins, minlength = len(self._occurrences))
        self._sum += np.sum(samples)
        self._sum_sq += np.sum(samples * samples)

class StatsMoments(StatsBaseMoments):
    def hist(self, axis = None):
        if axis is None:
            plot = plt
        else:
            plot = axis
        x = np.concatenate(([0], self._binning_thresholds))
        width = np.diff(x)
        width = np.concatenate((width, [width[-1]]))
        height = self._occurrences / (width * np.sum(self._occurrences))
        plot.bar(x[height > 0], height[height > 0], width[height > 0], align = 'edge')
        if axis is None:
            plot.show()

class StatsLogMoments(StatsBaseMoments):
    def log_hist(self, axis = None):
        if axis is None:
            plot = plt
        else:
            plot = axis