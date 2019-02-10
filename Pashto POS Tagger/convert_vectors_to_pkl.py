import os
import pickle
import numpy as np
import chardet
'''
This script converts the specified Glove file into a pickled dict
'''

def find_encoding(fname):
    r_file = open(fname, 'rb').read()
    result = chardet.detect(r_file)
    charenc = result['encoding']
    return charenc

embeddings_index = {}
textencoding=find_encoding('vectors.txt')
with open('vectors.txt', encoding=textencoding) as glove_file:
    for line in glove_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

if not os.path.exists('PickledData/'):
    print('MAKING DIRECTORY PickledData/ to save pickled glove file')
    os.makedirs('PickledData/')

with open('PickledData/Glove.pkl', 'wb') as f:
    pickle.dump(embeddings_index, f)

print('SUCESSFULLY SAVED Glove data as a pickle file in PickledData/')
