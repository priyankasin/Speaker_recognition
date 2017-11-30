#speaker specific Vector Quantization codebook using LBG algorithm
from __future__ import division
import numpy as np

#calculate Euclidean distance between two matrices
def EUDistance(d,c):

	n = np.shape(d)[1]
	p = np.shape(c)[1]
	distance = np.empty((n,p))

	if n<p:
		for i in range(n):
			copies = np.transpose(np.tile(d[:,i], (p,1)))
			distance[i,:] = np.sum((copies - c)**2,0)
	else:
		for i in range(p):
			copies = np.transpose(np.tile(c[:,i],(n,1)))
			distance[:,i] = np.transpose(np.sum((d - copies)**2,0))

	distance = np.sqrt(distance)
	return distance
def lbg(features, M):
	eps = 0.001
	codebook = np.mean(features, 1)
	distortion = 1
	nCentroid = 1
	while nCentroid < M:

	#double the size of codebook
		new_codebook = np.empty((len(codebook), nCentroid*2))
		if nCentroid == 1:
			new_codebook[:,0] = codebook*(1+eps)
			new_codebook[:,1] = codebook*(1-eps)
		else:
			for i in range(nCentroid):
				new_codebook[:,2*i] = codebook[:,i] * (1+eps)
				new_codebook[:,2*i+1] = codebook[:,i] * (1-eps)

		codebook = new_codebook
		nCentroid = np.shape(codebook)[1]
		D = EUDistance(features, codebook)
	
		while np.abs(distortion) > eps:
			#nearest neighbour search
			prev_distance = np.mean(D)
			nearest_codebook = np.argmin(D,axis = 1)

	#cluster vectors and find new centroid
			for i in range(nCentroid):
			#add along 3rd dimension
				codebook[:,i] = np.mean(features[:,np.where(nearest_codebook == i)], 2).T

	#replace all NaN values with 0
			codebook = np.nan_to_num(codebook)
			D = EUDistance(features, codebook)
			distortion = (prev_distance - np.mean(D))/prev_distance
	print(np.shape(codebook))
	return codebook