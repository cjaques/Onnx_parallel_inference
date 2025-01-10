from sklearn import datasets
import cv2

data = datasets.load_digits(n_class=2)

cnt =0
for i in range(data.data.shape[0]):
	dir_to_save = data.target[i]
	cv2.imwrite(str(dir_to_save) + "//" + str(i) + ".bmp", data.data[i].reshape((8, 8)))
	
