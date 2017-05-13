from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from vqlbg import EUDistance
from mfcc import mfcc
from train_id import training

nSpeaker = 1
nfiltbank = 20
(codebooks) = training(nfiltbank)
directory = 'test'
fname = str()
nCorrect_MFCC = 0


def minDistance(features, codebooks):
	speaker = 0
	distmin = np.inf
	for k in range(np.shape(codebooks)[0]):
			D = EUDistance(features, codebooks[k,:,:])
			dist = np.sum(np.min(D, axis = 1))/(np.shape(D)[0])
			if dist < distmin:
				distmin = dist
				speaker = k
	return speaker

fname = '/s1.wav'
(fs,s) = read(directory + fname)
mel_coefs = mfcc(s,fs,nfiltbank)
sp_mfcc = minDistance(mel_coefs, codebooks)
print('Speaker',' is matches with speaker',(sp_mfcc+1))


