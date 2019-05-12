import h5py
import numpy as np
import tensorflow as tf

class batchsampler:
    """Pulls a random batch of ns sequences from nl unique labels
    
    batchdim = [nl, ns]
        nl - number of labels
        ns - number of sequences per label, mask value is 99
    nframes = seconds/0.01, defaults to 2 seconds
        0.01 - mfcc sampling stepsize
    
    ***fair sampling requires prob(segment) ~ segment length***
    """
    def __init__(self, hdfname, batchdim, nsec = 2.0, nfeatures = 36):
        self.nepoch = -1
        self.hdf = hdfname
        self.nl, self.ns = batchdim
        self.nframes = int(nsec/0.01)
        self.nfeatures = nfeatures
        self.psegs = {}
        
        with h5py.File(hdfname, 'r') as f:
            for ilabel in f.keys():
                self.psegs[ilabel] = list()
                g = f[ilabel]

                for jseg in g.keys():
                    self.psegs[ilabel].append(g[jseg].attrs['length'])
        f.close()
        
        for x in self.psegs.keys():
            lengtharray = np.array(self.psegs[x])
            self.psegs[x] = np.divide(lengtharray, np.sum(lengtharray))
            
        self.labeldict = {}
        
        classes = list(self.psegs.keys() )
        labels = np.arange(0, len(classes))
        
        for i, iclass in enumerate(classes):
            self.labeldict[labels[i]] = iclass
            
        self.epochlabels = list(self.labeldict.keys() )
        self._newepoch()
        self._i = 0
    
    def _newepoch(self):
        np.random.shuffle(self.epochlabels)
        self._i = 0
        self.nepoch += 1
            
    def getbatch(self):
        _i, _ii = self._i, self._i + self.nl
        xbatch = np.full((self.nl*self.ns, self.nframes, self.nfeatures), 99. )
        ybatch = list()
        ix = 0

        with h5py.File(self.hdf, 'r') as f:
            for ilabel in self.epochlabels[_i:_ii]:
                iclass = self.labeldict[ilabel]
                g = f[iclass]
                segments = np.random.choice(list(g.keys()), self.ns, \
                                            p = self.psegs[iclass], \
                                            replace = False )
                for i, iseg in enumerate(segments):
                    nframes = int(g[iseg].attrs['length']/0.01) - 1
                    
                    if nframes < self.nframes:
                        xbatch[ix, :nframes, :self.nfeatures] = \
                        g[iseg][:nframes, :self.nfeatures]
                    else:
                        xbatch[ix, :, :self.nfeatures] = \
                        g[iseg][:self.nframes, :self.nfeatures]
                    ix += 1
                    ybatch.append(ilabel)
        f.close()

        self._i = _ii
        if self._i + self.ns > len(self.epochlabels):
            self._newepoch()
            
        xbatchtf = tf.convert_to_tensor(xbatch, dtype=tf.float32)
        ybatchtf = tf.convert_to_tensor(np.array(ybatch), dtype=tf.float32)
            
        return (xbatchtf, ybatchtf)
