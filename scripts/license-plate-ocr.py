import sys
import cv2
import numpy as np
import traceback
import time

import darknet.build.darknet.x64.darknet as dn

from os.path 				import splitext, basename
from glob					import glob
from darknet.build.darknet.x64.darknet import detect
from src.label				import dknet_label_conversion
from src.utils 				import nms

(width,height) = (240,80)

if __name__ == '__main__':

	try:

		input_dir  = sys.argv[1]
		output_dir = input_dir

		ocr_threshold = .4

		ocr_weights = 'data/ocr/ocr-net.weights'
		ocr_netcfg  = 'data/ocr/ocr-net.cfg'
		ocr_dataset = 'data/ocr/ocr-net.data'

		ocr_net  = dn.load_net(ocr_netcfg.encode("ascii"), ocr_weights.encode("ascii"), 0)
		ocr_meta = dn.load_meta(ocr_dataset.encode("ascii"))

		imgs_paths = sorted(glob('%s/*.jpg' % output_dir))

		print('Performing OCR...')

		for i,img_path in enumerate(imgs_paths):

			print('\tScanning %s' % img_path)
			bname = basename(splitext(img_path)[0])
			'''imj = cv2.imread(img_path)
			imj = cv2.cvtColor(imj,cv2.COLOR_BGR2GRAY)
			imj = cv2.resize(imj,(width,height))
			cv2.imwrite("tmp/f/{}.jpg".format(bname),imj)'''
			start_time=time.time()
			R = detect(ocr_net, ocr_meta, img_path.encode("ascii") ,thresh=ocr_threshold, nms=None)
			print(1/(time.time()-start_time))
			if len(R):

				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)

				L.sort(key=lambda x: x.tl()[0])
				lp_str = ''.join([chr(l.cl()) for l in L])

				with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
					f.write(lp_str + '\n')

				#print('\t\tLP: %s' % lp_str)

			else:
				continue
				#print('No characters found')

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
