#coding:utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
#读入文件，把文件读到buf里面 
file_train_data = 'data_set/train-images.idx3-ubyte'
file_train_label = 'data_set/train-labels.idx1-ubyte'
file_test_data = 'data_set/t10k-images.idx3-ubyte'
file_test_label = 'data_set/t10k-labels.idx1-ubyte'

filelist =[file_train_data,file_train_label,file_test_data,file_test_label]
def getData(name):
	if name =='train':
		binfile_data = open(filelist[0] , 'rb')
	elif name =='test':
		binfile_data = open(filelist[2] , 'rb')
	else:
		return -1
	buf = binfile_data.read()
	#index是偏移量，表示读文件的起始位置，‘>’表示大端存储，‘I’表示32位整数。这里以大端模式读取四个整数，所以有四个返回值（其实是返回一个包含四个整数的list）。
	index = 0
	magic , numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
	index += struct.calcsize('>IIII')
	numPixel = numRows*numColumns
	data = []
	while index < numPixel*numImages+struct.calcsize('>IIII'):
		pix = struct.unpack_from('>'+str(numPixel)+'B' ,buf, index)
		index += struct.calcsize('>'+str(numPixel)+'B')
		data.append(pix)
	data = np.array(data)
	return data

def getLabel(name):
	if name =='train':
		binfile_label = open(filelist[1] , 'rb')
	elif name =='test':
		binfile_label = open(filelist[3] , 'rb')
	else:
		return -1
	buf = binfile_label.read()
	index = 0
	magic , numItems = struct.unpack_from('>II' , buf , index)
	index += struct.calcsize('>II')
	label = []
	while index < numItems+struct.calcsize('>II'):
		lb = struct.unpack_from('>B',buf,index)
		index += struct.calcsize('>B')
		label.append(lb)
	label = np.array(label)
	return label


