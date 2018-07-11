""" Crop, resize and split data into training, validation and testing set
	
	All images from a folder will be resized to the same (square) size

	Data will be saved into memory maps to improve efficiency when loading into scripts

	Written by Anja Liljedahl Christensen, March 2018
"""

## Load packages
import numpy as np
import glob
import time

from PIL import Image
from datetime import datetime

import argparse

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--disease', nargs='?', type=str, default='Eczema', help='Name of disease')
parser.add_argument('-f', '--folder', nargs='?', type=str, default='Skin', help='Name of folder with disease images')
parser.add_argument('-ip', '--inpath', nargs='?', type=str, default='../Data', help='Path to disease folder')
parser.add_argument('-os', '--outsize', nargs='?', type=int, default=250, help='Size of output images')

args = parser.parse_args()
disease = args.disease
folder = args.folder
inpath = '{}/{}/{}/*.jpg'.format(args.inpath, disease, folder)
files = glob.glob(inpath)
outpath = '{}/{}/{}'.format(args.inpath, disease, 'No_crop')
output_size = args.outsize

## Resize each image in the files list to output_size x output_size
# 80% of the images will be saved in the X_train memory map, 10% in the X_valid memory map, 10% in the X_test memory map
def crop_imgs(files, output_name, output_size = 1200):
	n_imgs = len(files)
	n_train = int(0.8*n_imgs)
	n_valid = int(0.9*n_imgs)
	n_test = n_imgs

	X_train = np.memmap(output_name + "/X_train", dtype='float32', mode='w+', shape=(n_train,output_size,output_size,3))
	X_valid = np.memmap(output_name + "/X_valid", dtype='float32', mode='w+', shape=(n_valid-n_train,output_size,output_size,3))
	X_test = np.memmap(output_name + "/X_test", dtype='float32', mode='w+', shape=(n_test-n_valid,output_size,output_size,3))

	i_v = i_t = 0

	total_time = 0

	for i, f in enumerate(files):
		print("Processing file {}: {}".format(i, f))
		t0 = time.time()
		
		image = Image.open(f)
		image_res = image.resize((output_size,output_size))

		if i < n_train:
			X_train[i] = np.array(image_res)
		elif n_train <= i and i < n_valid:
			X_valid[i_v] = np.array(image_res)
			i_v += 1
		else:
			X_test[i_t] = np.array(image_res)
			i_t += 1

		t1 = time.time() - t0
		print("Took {} seconds".format(t1))

		total_time += t1

		image.close()

	print("Total time elapsed: {} seconds".format(total_time))

## FUNCTION CALL ## 
crop_imgs(files, output_name = outpath, output_size = output_size)
