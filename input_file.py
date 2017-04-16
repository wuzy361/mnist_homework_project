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


# draw_by_pixel(getData("test")[0])
# print getLabel("test")[0]

# draw_by_pixel(getData("test")[102])
# print getLabel("test")[102]

# draw_by_pixel(getData("test")[103])
# print getLabel("test")[103]
'''
binfile = open(filelist[1] , 'rb')
buf = binfile.read()
#index是偏移量，表示读文件的起始位置，‘>’表示大端存储，‘I’表示32位整数。这里以大端模式读取四个整数，所以有四个返回值（其实是返回一个包含四个整数的list）。
index = 0
magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
index += struct.calcsize('>IIII')
#一下读入784个unsigned byte
train_data = []
while index < 784*6000+32*4:
	im = struct.unpack_from('>784B' ,buf, index)
	index += struct.calcsize('>784B')
	train_data.append(im)
train_data = np.array(train_data)


'''

#把tuple类型的im转化成numpy.array，之后把784*1的变成28*28
# im_row = np.array(train_data[1])
# im = im_row.reshape(28,28)

#train_data = train_data.reshape(6000,784)


# fig = plt.figure()
# plt.imshow(im , cmap='gray')
# plt.show()