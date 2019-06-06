import sys
import pickle
from  gensim.models.keyedvectors import KeyedVectors

modelnow = KeyedVectors.load_word2vec_format('/home/cevbain/cc.en.300.vec')

testtestArray = pickle.load( open( "SuperNow2/500000.p", "rb" ) )

for i in range(0,52):
	print(modelnow.similar_by_vector(testtestArray[1][i], topn=1).encode('ascii', 'ignore'));

