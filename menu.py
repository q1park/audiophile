import os
import json
import soundfile as sf

import os
import json
import soundfile as sf

corrupt = ['EN2002a', 'EN2002c', 'EN2003a', 'EN2009d']

class makemenu:
    def __init__(self, basedir):
        self.basedir = basedir
        
        self.pathsdict = {}
        self.menudict = {}
    
    def loadpaths(self):
        labeldir = self.basedir + 'rttm/'
        filedir = self.basedir + 'amicorpus/'

        _, _, labelfiles = os.walk(labeldir).__next__()
        _, filetags, _ = os.walk(filedir).__next__()
        
        for x in labelfiles:
            filetag = x.split('.')[0]
            filepath = filedir + filetag + '/audio/' + filetag + '.Mix-Headset.wav'
            labelpath = labeldir + x
            self.pathsdict[filetag] = (filepath, labelpath)
            
    def createmenu(self):
        self.menudict['train'] = []
        self.menudict['test'] = []
        self.menudict['raw'] = []
        
        for x in self.pathsdict:
            if x in corrupt:
                continue
            
            filepath, labelpath = self.pathsdict[x]
            
            wavarr, samplerate = sf.read(filepath, dtype='float32', always_2d=True)
            length = wavarr.shape[0]/samplerate
            
            filedict = {}
            filedict['filetag'] = x
            filedict['filepath'] = filepath
            filedict['labelpath'] = labelpath
            filedict['samplerate'] = samplerate
            filedict['length'] = length
            filedict['hdf5'] = None
            
            self.menudict['raw'].append(filedict)

    def savemenu(self, menufile):
        with open(menufile, 'w') as outfile:
            json.dump(self.menudict, outfile, indent=2)