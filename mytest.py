from src.drawing_utils import draw_label, draw_losangle, write2img
from src.label import lread, Label, readShapes
import cv2
from os.path import splitext, basename, isfile
import numpy as np
from src.utils import show
import random

def labelImgShow():

	YELLOW = (  0,255,255)
	RED    = (  0,  0,255)

	img_file = '/home/yaokun/code/alpr-unconstrained/samples/train-detector/00011.jpg'
	lp_label = '/home/yaokun/code/alpr-unconstrained/samples/train-detector/00011.txt'

	I = cv2.imread(img_file)

	if isfile(lp_label):
		#pt1 = tuple(pts[:,i].astype(int).tolist())
		#pt2 = tuple(pts[:,(i+1)%4].astype(int).tolist())
		#cv2.line(I,pt1,pt2,color,thickness)
		Llp_shapes = readShapes(lp_label)
		print(np.array(I.shape[1::-1]).reshape(2,1))
		print(Llp_shapes[0].pts)
		pts = Llp_shapes[0].pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
		print(pts)
		#ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
		draw_losangle(I,pts,RED,3)
		show(I)


def test():
	a = np.zeros((2,0))
	values = [1,2,3,4,5,6,7,8]
	a = np.array(values).reshape((2,-1))
	print(a)
	whratio = random.uniform(2.,4.)
	print(whratio)
	rand = np.random.rand(3)
	print(rand)



if __name__ == '__main__':
	test()