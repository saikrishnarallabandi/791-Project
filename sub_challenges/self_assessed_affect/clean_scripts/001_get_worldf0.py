import os, sys
import numpy as np
import soundfile as sf
import pyworld as pw

# Locations
data_dir = '/home3/srallaba/data/ComParE2018_SelfAssessedAffect/wav/'
feats_dir = '../features/world_f0'

if not os.path.exists(feats_dir):
    os.makedirs(feats_dir)

files = sorted(os.listdir(data_dir))

for file in files:
    fname = file.split('.')[0]
    print fname
    x, fs = sf.read(data_dir + '/' + file)
    f0, sp, ap = pw.wav2world(x, fs) 
    np.savetxt(feats_dir + '/' + fname + '.f0_ascii', f0)

