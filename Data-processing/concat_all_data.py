"""	Concatenate training, validation and testing files from all diseases to 1 training, 1 validation and 1 testing set
	
	Written by Anja Liljedahl Christensen, May 2018
"""

## Load packages
import argparse
import numpy as np

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p0', '--path0', nargs='?', type=str, default='../Psoriasis', help='Path to input files of class 0')
parser.add_argument('-p1', '--path1', nargs='?', type=str, default='../Eczema', help='Path to input files of class 1')
parser.add_argument('-p2', '--path2', nargs='?', type=str, default='../Acne_vulgaris', help='Path to input files of class 2')
parser.add_argument('-p3', '--path3', nargs='?', type=str, default='../Rosacea', help='Path to input files of class 3')
parser.add_argument('-p4', '--path4', nargs='?', type=str, default='../Mycosis_fungoides', help='Path to input files of class 4')

parser.add_argument('-t0', '--train0', nargs='?', type=int, default=5236, help='Training size for class 0')
parser.add_argument('-t1', '--train1', nargs='?', type=int, default=4280, help='Training size for class 1')
parser.add_argument('-t2', '--train2', nargs='?', type=int, default=464, help='Training size for class 2')
parser.add_argument('-t3', '--train3', nargs='?', type=int, default=1284, help='Training size for class 3')
parser.add_argument('-t4', '--train4', nargs='?', type=int, default=1968, help='Training size for class 4')

parser.add_argument('-v0', '--valid0', nargs='?', type=int, default=654, help='Validation size for class 0')
parser.add_argument('-v1', '--valid1', nargs='?', type=int, default=535, help='Validation size for class 1')
parser.add_argument('-v2', '--valid2', nargs='?', type=int, default=58, help='Validation size for class 2')
parser.add_argument('-v3', '--valid3', nargs='?', type=int, default=161, help='Validation size for class 3')
parser.add_argument('-v4', '--valid4', nargs='?', type=int, default=246, help='Validation size for class 4')

parser.add_argument('-te0', '--test0', nargs='?', type=int, default=655, help='Test size for class 0')
parser.add_argument('-te1', '--test1', nargs='?', type=int, default=535, help='Test size for class 1')
parser.add_argument('-te2', '--test2', nargs='?', type=int, default=59, help='Test size for class 2')
parser.add_argument('-te3', '--test3', nargs='?', type=int, default=161, help='Test size for class 3')
parser.add_argument('-te4', '--test4', nargs='?', type=int, default=247, help='Test size for class 4')

parser.add_argument('-s', '--size', nargs='?', type=int, default=250, help='Size of images')
parser.add_argument('-c', '--chns', nargs='?', type=int, default=3, help='Number of color channels')

args = parser.parse_args()
path0 = args.path0
path1 = args.path1
path2 = args.path2
path3 = args.path3
path4 = args.path4

t0 = args.train0
t1 = args.train1
t2 = args.train2
t3 = args.train3
t4 = args.train4

v0 = args.valid0
v1 = args.valid1
v2 = args.valid2
v3 = args.valid3
v4 = args.valid4

te0 = args.test0
te1 = args.test1
te2 = args.test2
te3 = args.test3
te4 = args.test4

img_s = args.size
n_channels = args.chns

## The total training size, validation size and testing size
train_size = t0+t1+t2+t3+t4
val_size = v0+v1+v2+v3+v4
test_size = te0+te1+te2+te3+te4

## Open individual training sets
X_train_0 = np.memmap('{}/X_train'.format(path0), dtype='float32', mode='c', shape=(t0, img_s, img_s, n_channels))
print('Train 0 done')
X_train_1 = np.memmap('{}/X_train'.format(path1), dtype='float32', mode='c', shape=(t1, img_s, img_s, n_channels))
print('Train 1 done')
X_train_2 = np.memmap('{}/X_train'.format(path2), dtype='float32', mode='c', shape=(t2, img_s, img_s, n_channels))
print('Train 2 done')
X_train_3 = np.memmap('{}/X_train'.format(path3), dtype='float32', mode='c', shape=(t3, img_s, img_s, n_channels))
print('Train 3 done')
X_train_4 = np.memmap('{}/X_train'.format(path4), dtype='float32', mode='c', shape=(t4, img_s, img_s, n_channels))
print('Train 4 done')

## Open individual validation sets
X_valid_0 = np.memmap('{}/X_valid'.format(path0), dtype='float32', mode='c', shape=(v0, img_s, img_s, n_channels))
print('Valid 0 done')
X_valid_1 = np.memmap('{}/X_valid'.format(path1), dtype='float32', mode='c', shape=(v1, img_s, img_s, n_channels))
print('Valid 1 done')
X_valid_2 = np.memmap('{}/X_valid'.format(path2), dtype='float32', mode='c', shape=(v2, img_s, img_s, n_channels))
print('Valid 2 done')
X_valid_3 = np.memmap('{}/X_valid'.format(path3), dtype='float32', mode='c', shape=(v3, img_s, img_s, n_channels))
print('Valid 3 done')
X_valid_4 = np.memmap('{}/X_valid'.format(path4), dtype='float32', mode='c', shape=(v4, img_s, img_s, n_channels))
print('Valid 4 done')

## Open individual testing sets
X_test_0 = np.memmap('{}/X_test'.format(path0), dtype='float32', mode='c', shape=(te0, img_s, img_s, n_channels))
print('Test 0 done')
X_test_1 = np.memmap('{}/X_test'.format(path1), dtype='float32', mode='c', shape=(te1, img_s, img_s, n_channels))
print('Test 1 done')
X_test_2 = np.memmap('{}/X_test'.format(path2), dtype='float32', mode='c', shape=(te2, img_s, img_s, n_channels))
print('Test 2 done')
X_test_3 = np.memmap('{}/X_test'.format(path3), dtype='float32', mode='c', shape=(te3, img_s, img_s, n_channels))
print('Test 3 done')
X_test_4 = np.memmap('{}/X_test'.format(path4), dtype='float32', mode='c', shape=(te4, img_s, img_s, n_channels))
print('Test 4 done')

## Initiliaize memory maps for total training, validation and testing sets
X_train = np.memmap("../X_train", dtype='float32', mode='w+', shape=(train_size, img_s, img_s, n_channels))
X_valid = np.memmap("../X_valid", dtype='float32', mode='w+', shape=(val_size, img_s, img_s, n_channels))
X_test = np.memmap("../X_test", dtype='float32', mode='w+', shape=(test_size, img_s, img_s, n_channels))

## Initialize memory maps for total training, validation and testing labels 
Y_train = np.memmap("../Y_train", dtype='int', mode='w+', shape=(train_size, 1))
Y_valid = np.memmap("../Y_valid", dtype='int', mode='w+', shape=(val_size, 1))
Y_test = np.memmap("../Y_test", dtype='int', mode='w+', shape=(test_size, 1))

## Concatenate data
X_train[:train_size] = np.concatenate((X_train_0, X_train_1, X_train_2, X_train_3, X_train_4))
X_valid[:val_size] = np.concatenate((X_valid_0, X_valid_1, X_valid_2, X_valid_3, X_valid_4))
X_test[:test_size] = np.concatenate((X_test_0, X_test_1, X_test_2, X_test_3, X_test_4))

## Concatenate labels
Y_train[:train_size] = np.array([[0]]*t0 + [[1]]*t1 + [[2]]*t2 + [[3]]*t3 + [[4]]*t4, dtype = "int")
Y_valid[:val_size] = np.array([[0]]*v0 + [[1]]*v1 + [[2]]*v2 + [[3]]*v3 + [[4]]*v4, dtype = "int")
Y_test[:test_size] = np.array([[0]]*te0 + [[1]]*te1 + [[2]]*te2 + [[3]]*te3 + [[4]]*te4, dtype = "int")


