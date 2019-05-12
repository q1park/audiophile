import os
import json
import h5py
import random

import numpy as np
import pandas as pd
import soundfile as sf
import yaafelib
import librosa
from librosa.util import valid_audio
from librosa.util.exceptions import ParameterError
from tqdm import tqdm

class makefeatures:
    """MFCC feature extraction
    
    samplerate : int, Defaults to 16000.
    duration : float, Defaults to 0.025.
    step : float, Defaults to 0.010.
    e : bool, energy. Defaults to True.
    coefs : int, number of coefficients. Defaults to 11.
    De : bool, energy first derivative. Defaults to False.
    D : bool, first order derivatives. Defaults to False.
    DDe : bool, energy second derivative. Defaults to False.
    DD : bool, second order derivatives. Defaults to False.
    self.augment : NOT YET IMPLEMENTED. Defaults to None
    """

    def __init__(self, samplerate=16000, duration=0.025, step=0.010,
                 coefs=11, e=True, De=True, DDe=False, D=True, DD=False,
                 mono=True, channel = None, augment=None):
        
        self.samplerate = samplerate
        self.duration = duration
        self.step = step
        
        self.e = e
        self.coefs = coefs
        self.De = De
        self.DDe = DDe
        self.D = D
        self.DD = DD
        
        self.mono = mono
        self.channel = channel
        self.augment = augment

        n_features = 0
        n_features += self.e
        n_features += self.De
        n_features += self.DDe
        n_features += self.coefs
        n_features += self.coefs * self.D
        n_features += self.coefs * self.DD
        self.dimension = n_features

        blockSize = int(self.samplerate * self.duration)
        stepSize = int(self.samplerate * self.step)
        
        self.definition = list()
        # --- coefficients, (0,1) if energy is (kept,removed)
        self.definition.append(("mfcc",
                                "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
                                "blockSize=%d stepSize=%d" % (
                                    0 if self.e else 1,
                                    self.coefs + self.e * 1,
                                    blockSize, stepSize) ) )

        # --- 1st order derivatives
        if self.De or self.D:
            self.definition.append(("mfcc_d",
                                    "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
                                    "blockSize=%d stepSize=%d > Derivate DOrder=1" % (
                                        0 if self.De else 1,
                                        self.D * self.coefs + self.De * 1,
                                        blockSize, stepSize) ) )

        # --- 2nd order derivatives
        if self.DDe or self.DD:
            self.definition.append(("mfcc_dd",
                                    "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
                                    "blockSize=%d stepSize=%d > Derivate DOrder=2" % (
                                        0 if self.DDe else 1,
                                        self.DD * self.coefs + self.DDe * 1,
                                        blockSize, stepSize) ) )
            
        self.featurelist = [x[0] for x in self.definition]
        
        self.menufile = ''
        self.menu = {}
        self.features = {}
        self.featuretags = list()
            
    def loadmenu(self, menufile):
        self.menufile = menufile
        with open(menufile) as menu: 
            self.menu = json.load(menu)
    
    def featurize(self, xtype, idx = 0):
        assert len(self.menu['raw']) > 0
        assert len(self.features) == 0
        
        file = self.menu['raw'][idx]['filetag']
        self.featuretags = [file, xtype, idx]
        print('processing', file, 'as', xtype)

        filepath = self.menu['raw'][idx]['filepath']
        labelpath = self.menu['raw'][idx]['labelpath']
        
        dfwave, dflabels = self._file2wave(filepath, labelpath)

        with tqdm(total=len(list(dflabels.iterrows()))) as pbar:
            for i, irow in dflabels.iterrows():
                label = irow['id'].replace('Speaker', file)
                
                if irow['dt'] < 0.1 or np.abs(irow['dt']) > 100:
                    continue
                if label not in self.features:
                    self.features[label] = list()

                start, end = irow['t'], irow['t'] + irow['dt']
                segment = dfwave.loc[(dfwave['t'] >= start) & (dfwave['t'] <= end)]
                segment = segment['wave'].values

                featuredict = self._wave2features(segment)
                
                features = np.hstack([featuredict[name] for name, _ in self.definition])
                segmentlength = segment.shape[0]/self.samplerate
                
                self.features[label].append((segmentlength, features))
                
                pbar.update(1)
            pbar.close()

    def savefeatures(self, hdfname):
        assert len(self.features) > 0
        file, xtype, idx = self.featuretags
            
        with h5py.File(hdfname, 'a') as f:
            if os.path.exists(hdfname) and len(list(f.keys() ) ) > 0:
                assert f.attrs['type'] == xtype
                #assert f.attrs['features'] == [a.encode('utf8') for a in self.featurelist]
                print('file exists.. adding to file')
            else:
                f.attrs['type'] = xtype
                f.attrs['features'] = [a.encode('utf8') for a in self.featurelist]
                
            for ilabel in self.features.keys():
                g = f.create_group(ilabel) if ilabel not in f.keys() else f[ilabel]

                for i, isample in enumerate(self.features[ilabel]):
                    if str(i) not in g.keys():
                        d = g.create_dataset(str(i), data = isample[1])  
                        d.attrs['length'] = isample[0]
                    else:
                        pass
        f.close()
        
        self.menu['raw'][idx]['hdf5'] = hdfname
        processed = self.menu['raw'].pop(idx)
        self.menu[xtype].append(processed)
        with open(self.menufile, 'w') as outfile:
            json.dump(self.menu, outfile, indent=2)
            
        print('saved', file, 'as', xtype, 'in', hdfname)
        
        self.features.clear()
        self.featuretags.clear()
        
    def printhdf(self, hdfname):
        with h5py.File(hdfname, 'r') as f:
            print('data type:', f.attrs['type'])
            print('features:', [str(x, 'utf-8') for x in f.attrs['features']])
            print('labels:', list(f.keys()))
            
            g = f[list(f.keys())[0]]
            d = g[list(g.keys())[0]]

            print('   segment length:', d.attrs['length'])
            print('   data dimensions:', d[:].shape )
        
                    
    def _file2wave(self, filepath, labelpath):
        wavarr, filesamplerate = sf.read(filepath, dtype='float32', always_2d=True)
        filelength = wavarr.shape[0]/filesamplerate

        if self.channel is not None:
            wavarr = wavarr[:, self.channel - 1 : self.channel]

        if self.mono and wavarr.shape[1] > 1:
            wavarr = np.mean(wavarr, axis=1, keepdims=True)

        if filesamplerate != self.samplerate:
            wavarr = librosa.core.resample(wavarr.T, filesamplerate, self.samplerate).T

        timestamps = np.arange(0, filelength, 1/filesamplerate)

        assert wavarr.shape[1] == 1
        if len(timestamps) != wavarr.shape[0]:
            raise ValueError('sampling rate inconsistent with duration')

        wave = pd.DataFrame({'t': timestamps, 'wave': wavarr[:, 0]})
        labels = pd.read_csv(labelpath, delim_whitespace = True, \
                             usecols = [3, 4, 7], names = ['t', 'dt', 'id'])
        return wave, labels
                    
    def _wave2features(self, wavearray):
        engine = yaafelib.Engine()
        featureplan = yaafelib.FeaturePlan(sample_rate=self.samplerate)
        
        for name, recipe in self.definition:
            assert featureplan.addFeature("{name}: {recipe}".format(name=name, recipe=recipe));
            
        dataflow = featureplan.getDataFlow()
        engine.load(dataflow)
            
        wavearray = np.array(wavearray, dtype=np.float64, order='C').reshape((1, -1))
        features = engine.processAudio(wavearray)
        engine.reset()
        
        return features