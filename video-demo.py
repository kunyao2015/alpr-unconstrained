import sys
import cv2
import numpy as np
import traceback
# keras 放在这里太尼玛重要了，否则会导致load_model加载不成功，报h5py Invalid high library version bound的错误，
# 此错误与下面 darknet.python.darknet 的引入有关
import keras   

import darknet.python.darknet as dn

from src.label 				import Label, lwrite, dknet_label_conversion
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder, im2single, nms
from darknet.python.darknet import detect, load_image, nparray_to_image

from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes
from src.drawing_utils			import draw_label, draw_losangle, write2img

YELLOW = (  0,255,255)
RED    = (  0,  0,255)

if __name__ == '__main__':

	try:
	
		input_video_path  = '/home/yaokun/data/03016.jpg'#sys.argv[1]
		input_video = input_video_path.encode('ascii')
		output_dir = '/home/yaokun/tmp'#sys.argv[2]

		bname = basename(splitext(input_video_path)[0])


		# vehicle detect 
		vehicle_threshold = .5

		vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'  #.encode('ascii')
		vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'#.encode('ascii').encode('ascii')
		vehicle_dataset = 'data/vehicle-detector/voc.data'#.encode('ascii')
		vehicle_weights = vehicle_weights.encode('ascii')
		vehicle_netcfg = vehicle_netcfg.encode('ascii')
		vehicle_dataset = vehicle_dataset.encode('ascii')

		vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
		vehicle_meta = dn.load_meta(vehicle_dataset)

        # lp detect
		lp_threshold = .5

		wpod_net_path = 'data/lp-detector/bak/wpod-net_update1.h5'
		print(wpod_net_path)
		#wpod_net_path = wpod_net_path.encode('ascii')
		wpod_net = load_model(wpod_net_path)

        # ocr
		ocr_threshold = .4

		ocr_weights = 'data/ocr/ocr-net.weights'
		ocr_netcfg  = 'data/ocr/ocr-net.cfg'
		ocr_dataset = 'data/ocr/ocr-net.data'
		ocr_weights = ocr_weights.encode('ascii')
		ocr_netcfg = ocr_netcfg.encode('ascii')
		ocr_dataset = ocr_dataset.encode('ascii')

		ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
		ocr_meta = dn.load_meta(ocr_dataset)



		if not isdir(output_dir):
			makedirs(output_dir)

		print('Searching for vehicles using YOLO...')

		cap = cv2.VideoCapture(input_video)

		fps = cap.get(cv2.CAP_PROP_FPS)
		print("fps:", fps)
		totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		print("总帧数:%s" % totalFrameNumber)

		widht = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		print("宽%s , 高%s:" % (width, height))

		frame_index = 0
        #frame_count = 0
		if cap.isOpened():
			success = True
		else:
			success = False
			print("Open vedio file failed!")

		while(success):
			success, frame = cap.read()
			frame_index += 1
			# 跳10帧处理一次
			if frame_index % 10 != 1:
				continue

			R,_ = detect(vehicle_net, vehicle_meta, frame ,thresh=vehicle_threshold, is_imgpath=False)
			#print(R)

			R = [r for r in R if r[0] in [b'car',b'bus']]

			print('\t\t%d cars found' % len(R))

			if len(R):
				Iorig = frame  # 原始帧图片
				WH = np.array(Iorig.shape[1::-1],dtype=float)
				Lcars = []

				for i,r in enumerate(R):

					cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
					tl = np.array([cx - w/2., cy - h/2.])
					br = np.array([cx + w/2., cy + h/2.])
					label = Label(0,tl,br)
					print(label.wh())
					print(label.tl())
					Icar = crop_region(Iorig,label) # 截取车辆区域 Icar 车辆 label 车辆坐标信息
					Icar = Icar.astype('uint8')

					Lcars.append(label)
					draw_label(Iorig,label,color=YELLOW,thickness=3)

					# lp detector
					print('Searching for license plates using WPOD-NET')

					ratio = float(max(Icar.shape[:2]))/min(Icar.shape[:2])
					side  = int(ratio*288.)
					bound_dim = min(side + (side%(2**4)),608)
					print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

	 
					Llp,LlpImgs,elapse = detect_lp(wpod_net,im2single(Icar),bound_dim,2**4,(240,80),lp_threshold)
					print('\tld detection used %d s time' % elapse)

					if len(LlpImgs):
						# 检测的车牌图像信息
						Ilp = LlpImgs[0]
						Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
						Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

						s = Shape(Llp[0].pts) # 车牌坐标信息

						#cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
						#writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
						lpts = Llp[0].pts*label.wh().reshape(2,1) + label.tl().reshape(2,1)
						lptspx = lpts*np.array(Iorig.shape[1::-1],dtype=float).reshape(2,1)
						draw_losangle(Iorig,lptspx,RED,3)

						Ilp = Ilp*255.
						Ilp = Ilp.astype('uint8') # 十分重要，不做这个转换，就无法识别
						'''
						cv2.imshow("img", Ilp)
						cv2.waitKey(3000)
						cv2.imwrite('%s/%s_new_lp.png' % (output_dir,bname),Ilp)	
						'''
						dk_image = nparray_to_image(Ilp)
						
						R2,(width,height) = detect(ocr_net, ocr_meta, dk_image ,thresh=ocr_threshold, nms=None, is_imgpath=False)
						print(width,height)

						if len(R2):

							L = dknet_label_conversion(R2,width,height)
							L = nms(L,.45)

							L.sort(key=lambda x: x.tl()[0])
							lp_str = ''.join([chr(l.cl()) for l in L]) # ocr识别出的车牌字符串

							#with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
							#	f.write(lp_str + '\n')

							print('\t\tLP: %s' % lp_str)
							label_lp = Label(0,tl=lpts.min(1),br=lpts.max(1))
							write2img(Iorig,label_lp,lp_str)

						else:

							print('No characters found')

			cv2.imshow("test.vedio", Iorig)
			cv2.waitKey(10)
			#lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)