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
    def log_min(self):
        return np.min(np.ma.masked_invalid(np.log(self.array)))
    
    @property
    def max(self):
        return np.max(self.array)
    
    @property
    def log_max(self):
        return np.max(np.ma.masked_invalid(np.log(self.array)))
    
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
    
    def hist(self, axis = None, color = "C0", lw = 0.5):
        if axis is None:
            plot = plt
        else:
            plot = axis
        plot.hist(self.array, bins = self.bins, density = True,
                  color = color, edgecolor = "black", lw = 0.5)
    
    def log_hist(self, axis = None, color = "C0", lw = 0.5):
        if axis is None:
            ax = plt.axes()
        else:
            ax = axis
        
        lower = np.min(np.ma.masked_invalid(np.log(self.array)))
        upper = np.max(np.ma.masked_invalid(np.log(self.array)))
        log_width = (upper - lower) / self.bins
        thresholds = np.exp(np.linspace(lower, upper, num = self.bins + 1))
        width = np.diff(thresholds)
        thresholds = thresholds[:-1]

        bins = np.digitize(self.array, thresholds)
        occurrences = np.bincount(bins, minlength = self.bins)
        
        if occurrences[0] > 0:
            width = np.concatenate((thresholds[0:1], width))
            thresholds = np.concatenate(([0], thresholds))
            height = occurrences / np.sum(log_width * occurrences)
            ax.bar(thresholds[height > 0], height[height > 0], width[height > 0],
                   align = 'edge', color = color, edgecolor = "black", lw = lw)
            ax.set_xscale('symlog', linthreshx = 5*thresholds[1])
        else:
            occurrences = occurrences[1:]
            height = occurrences / np.sum(log_width * occurrences)
            
            ax.bar(thresholds[height > 0], height[height > 0], width[height > 0],
                   align = 'edge', color = color, edgecolor = "black", lw = lw)
            ax.set_xscale('log')
        
        if axis is None:
            plt.show()
        


class StatsBaseMoments(StatsBase):
    def __init__(self, binning_thresholds):
        super().__init__()
        self.binning_thresholds = np.array(binning_thresholds)
    
    @property
    def binning_thresholds(self):
        return self._binning_thresholds
    
    @binning_thresholds.setter
    def binning_thresholds(self, value):
        if not np.all(np.diff(value) > 0):
            raise ValueError("Thresholds must be strictly monotonically increasing")
        self._binning_thresholds = value
        self._occurrences = np.zeros(len(value) + 1)
    
    def append(self, samples):
        super().append(samples)
        bins = np.digitize(samples, self._binning_thresholds)
        #TODO: Warn if data outside of range
        self._occurrences += np.bincount(bins, minlength = len(self._occurrences))
        

class StatsMoments(StatsBaseMoments):
    def __init__(self, binning_thresholds):
        super().__init__(binning_thresholds)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = np.inf
        self._max = -np.inf
    
    @property
    def min(self):
        return self._min
    
    @property
    def max(self):
        return self._max
        
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
        super().append(samples)
        self._sum += np.sum(samples)
        self._sum_sq += np.sum(samples * samples)
        self._min = min(self._min, np.min(samples))
        self._max = max(self._max, np.max(samples))
    
    def hist(self, axis = None, color = "C0", lw = 0.5):
        if axis is None:
            ax = plt.axes
        else:
            ax = axis
        thresholds = self._binning_thresholds[:-1]
        widths = np.diff(self._binning_thresholds)
        heights = self._occurrences[1:-1] / (widths * np.sum(self._occurrences))
        ax.bar(thresholds[heights > 0], heights[heights > 0], widths[heights > 0],
               align = 'edge', color = color, edgecolor = "black", lw = lw)
        if axis is None:
            plt.show()

class StatsLogMoments(StatsBaseMoments):
    def __init__(self, binning_thresholds):
        super().__init__(binning_thresholds)
        self._log_sum = 0.0
        self._log_sum_sq = 0.0
        self._log_min = np.inf
        self._log_max = -np.inf
    
    @StatsBaseMoments.binning_thresholds.setter
    def binning_thresholds(self, value):
        if np.any(value < 0):
            raise ValueError("Thresholds for log bins must not be negative.")
        if value[0] > 0:
            value = np.concatenate(([0], value))
        StatsBaseMoments.binning_thresholds.fset(self, value)
    
    @property
    def log_min(self):
        return self._log_min
    
    @property
    def log_max(self):
        return self._log_max
        
    @property
    def log_mean(self):
        return self._log_sum / self._samples
    
    @property
    def log_var(self):
        return self._log_sum_sq / self._samples - self.log_mean**2
    
    @property
    def log_sample_var(self):
        return self.log_var * self._samples / (self._samples - 1)
    
    def append(self, samples):
        super().append(samples)
        log_samples = np.ma.masked_invalid(np.log(samples))
        self._log_sum += np.sum(log_samples)
        self._log_sum_sq += np.sum(log_samples * log_samples)
        self._log_min = min(self._log_min, np.min(log_samples))
        self._log_max = max(self._log_max, np.max(log_samples))

    def log_hist(self, axis = None, color = "C0", lw = 0.5):
        if axis is None:
            ax = plt.axes()
        else:
            ax = axis
        
        log_widths = np.diff(np.log(self.binning_thresholds[1:]))
        
        if self._occurrences[1] > 0:
            occurrences = self._occurrences[1:-1]
            #TODO: Omit empty bins between 0 and first nonempty log-bin
            thresholds = self.binning_thresholds[:-1]
            widths = np.diff(self.binning_thresholds)
            log_widths = np.concatenate(([np.mean(log_widths)], log_widths))
            heights = occurrences / (log_widths * np.sum(occurrences))
            
            ax.bar(thresholds[heights > 0], heights[heights > 0], widths[heights > 0],
                   align = 'edge', color = color, edgecolor = "black", lw = lw)
            ax.set_xscale('symlog', linthreshx = 5*thresholds[1])
        else:
            occurrences = self._occurrences[2:-1]
            thresholds = self.binning_thresholds[1:-1]
            widths = np.diff(self.binning_thresholds[1:])
            heights = occurrences / (log_widths * np.sum(occurrences))
            
            ax.bar(thresholds[heights > 0], heights[heights > 0], widths[heights > 0],
                   align = 'edge', color = color, edgecolor = "black", lw = lw)
            ax.set_xscale('log')
        
        if axis is None:
            plt.show()