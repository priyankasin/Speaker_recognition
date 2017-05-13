from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from vqlbg import lbg
from mfcc import mfcc

def training(nfiltbank):
	nSpeaker = 8
	nCentroid = 16
	codebooks = np.empty((nSpeaker,nfiltbank,nCentroid))
	directory = 'train'
	fname = str()

	for i in range(nSpeaker):
		fname = '/s' + str(i+1) + '.wav'
		(fs,s) = read(directory + fname)
		mel_coeff = mfcc(s, fs, nfiltbank)
		codebooks[i,:,:] = lbg(mel_coeff, nCentroid)
			 
	print('Training complete')
	return (codebooks)
