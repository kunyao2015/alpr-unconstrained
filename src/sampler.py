
import cv2
import numpy as np
import random
import pysnooper

from src.utils 	import im2single, getWH, hsv_transform, IOU_centre_and_dims
from src.label	import Label
from src.projection_utils import perspective_transform, find_T_matrix, getRectPts

#@pysnooper.snoop('./log/file.log')
def labels2output_map(label,lppts,dim,stride):

	side = ((float(dim) + 40.)/2.)/stride # 7.75 when dim = 208 and stride = 16

	outsize = int(dim/stride) # python2和3除法的区别
	Y  = np.zeros((outsize,outsize,2*4+1),dtype='float32')
	MN = np.array([outsize,outsize])
	WH = np.array([dim,dim],dtype=float)

	tlx,tly = np.floor(np.maximum(label.tl(),0.)*MN).astype(int).tolist()
	brx,bry = np.ceil (np.minimum(label.br(),1.)*MN).astype(int).tolist()

	for x in range(tlx,brx):
		for y in range(tly,bry):

			mn = np.array([float(x) + .5, float(y) + .5])
			iou = IOU_centre_and_dims(mn/MN,label.wh(),label.cc(),label.wh())

			if iou > .5:

				p_WH = lppts*WH.reshape((2,1))
				p_MN = p_WH/stride

				p_MN_center_mn = p_MN - mn.reshape((2,1))

				p_side = p_MN_center_mn/side

				Y[y,x,0] = 1.
				Y[y,x,1:] = p_side.T.flatten()

	return Y

def pts2ptsh(pts):
	return np.matrix(np.concatenate((pts,np.ones((1,pts.shape[1]))),0))

def project(I,T,pts,dim):
	ptsh 	= np.matrix(np.concatenate((pts,np.ones((1,4))),0))
	ptsh 	= np.matmul(T,ptsh)
	ptsh 	= ptsh/ptsh[2]
	ptsret  = ptsh[:2]
	ptsret  = ptsret/dim
	Iroi = cv2.warpPerspective(I,T,(dim,dim),borderValue=.0,flags=cv2.INTER_LINEAR)
	return Iroi,ptsret

def flip_image_and_pts(I,pts):
	I = cv2.flip(I,1)
	pts[0] = 1. - pts[0]
	idx = [1,0,3,2]
	pts = pts[...,idx]
	return I,pts

#@pysnooper.snoop('./log/file.log')
def augment_sample(I,pts,dim):

	maxsum,maxangle = 120,np.array([80.,80.,45.])
	angles = np.random.rand(3)*maxangle
	if angles.sum() > maxsum:
		angles = (angles/angles.sum())*(maxangle/maxangle.sum())

	I = im2single(I)  # /255.0 归一化
	iwh = getWH(I.shape) #得到宽高

	whratio = random.uniform(2.,4.) # 随即取一个2-4的值作为宽高比
	wsiz = random.uniform(dim*.2,dim*1.) # 宽取0.2*208 到 208之间
	
	hsiz = wsiz/whratio

	dx = random.uniform(0.,dim - wsiz)
	dy = random.uniform(0.,dim - hsiz)

    #下面涉及到整个变换
    # In the first 3 lines, the original corner points are transformed into a rectangular bounding box with aspect ratio 
    # varying between 2:1 and 4:1. In other words, T matrix rectifies the LP with a random aspect ratio. Then, 
    
	pph = getRectPts(dx,dy,dx+wsiz,dy+hsiz)
	pts = pts*iwh.reshape((2,1))  #将点恢复到真实坐标值
	T = find_T_matrix(pts2ptsh(pts),pph)
    #in the next two lines, a perspective transformation with random rotation (H) is combined with T to 
    #generate the final transformation.
	H = perspective_transform((dim,dim),angles=angles)
	H = np.matmul(H,T)

	Iroi,pts = project(I,H,pts,dim)
	
	hsv_mod = np.random.rand(3).astype('float32')
	hsv_mod = (hsv_mod - .5)*.3
	hsv_mod[0] *= 360
	Iroi = hsv_transform(Iroi,hsv_mod)
	Iroi = np.clip(Iroi,0.,1.)

	pts = np.array(pts)

	if random.random() > .5:
		Iroi,pts = flip_image_and_pts(Iroi,pts)

	tl,br = pts.min(1),pts.max(1)
	llp = Label(0,tl,br)

	return Iroi,llp,pts
