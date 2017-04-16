from input_file import getData ,getLabel
import matplotlib.pyplot as plt

def draw_by_pixel(ndarray):
	im = ndarray.reshape(28,28)
	fig = plt.figure()
	plt.imshow(im,cmap = 'gray')
	plt.show()
	return 'draw success!'

def just_test(s,n1,n2):
	data = getData(s)
	label = getLabel(s)	
	for x in range(n1,n2):
		draw_by_pixel(data[x])
		print label[x]
print "train group:"
just_test('train',5,10)
print "test group:"
just_test('test',105,110)
