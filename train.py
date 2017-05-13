from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from vqlbg import lbg
from mfcc import mfcc

def training(nfiltbank):
	nSpeaker=1
	nCentroid = 16
	codebooks_mfcc = np.empty((nSpeaker,nfiltbank,nCentroid))
	directory = 'train'
	fname = str()

	
	fname = '/s1.wav'
	(fs,s) = read(directory + fname)
	mel_coeff = mfcc(s, fs, nfiltbank)
	codebook = lbg(mel_coeff, nCentroid)
		
	 
	print('Training complete')
	return (codebook)
