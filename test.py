from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from vqlbg import EUDistance
from mfcc import mfcc
from train import training

nfiltbank = 12	
(codebook) = training(nfiltbank)
directory = 'test'
fname = str()
nCorrect_MFCC = 0


def minDistance(features, codebook):
	D = EUDistance(features, codebook)
	dist = np.sum(np.min(D, axis = 1))/(np.shape(D)[0])
	print(dist)
	return dist


dmin=20     # threshold
fname = '/s1.wav'
(fs,s) = read(directory + fname)
mel_coefs = mfcc(s,fs,nfiltbank)
distmin = minDistance(mel_coefs, codebook)
if distmin<=dmin:
	print('Match')
else:
	print('Not Match')

	
