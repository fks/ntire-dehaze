# coding=utf-8
import os
import cv2
import keras.backend as K
import numpy as np
import skimage
import tensorflow as tf
import time
from generator_dehaze import get_data, rgbf2bgr, flip_axis
from kf_util import mkdir_p
from predict_depth import Depth4Predictor
from keras_custom import init_seeds
from models_dehaze import create_simple_dehaze10
import sys

batch_size=10
seed=123
#scale_down_to=None
scale_down_to=(1200,1972)
input_size=(512,512)
tile_overlap=64    # higher overlap will get better results (tiles less visible) at higher CPU cost
tile_core=(input_size[0]-tile_overlap,input_size[1]-tile_overlap)


weights_file = "../pretrained/dehaze10_weights_128_0.0987226869911.hdf5"
#weights_file = "../pretrained/dehaze10_weights_130_0.102881066501.hdf5"
#weights_file = "../pretrained/dehaze10_weights_197_22.1342916489.hdf5"

if len(sys.argv) == 2 and sys.argv[1] == 'outdoor':
	track="outdoor"
elif len(sys.argv) == 2 and sys.argv[1] == 'indoor':
	track="indoor"
elif len(sys.argv) == 1:
	track = 'indoor'
	print ("Use 'python submission.py outdoor' to generate submission for outdoor track.")
else:
	raise ValueError("Unknown arguments. Usage: submission.py <indoor | outdoor>")

test_image_path='../data/'+track+'/test'
out_dir= "../submission/submit_"+track

init_seeds(seed)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.allow_soft_placement=True
sess = tf.Session(config=config)
K.set_session(sess)

def save_prediced_samples(seq, model, ouput_dir):
	mkdir_p(ouput_dir)
	i=0
	for xb,yb in seq:
		ypb = model.predict_on_batch(xb)
		for x,y,yp in zip(xb,yb,ypb):
			img_all=np.hstack((x[:,:,0:3],y[:,:,0:3],yp[:,:,0:3]))
			bgr = rgbf2bgr(img_all)
			cv2.imwrite('%s/%d.jpg'%(ouput_dir,i,),bgr)
			i += 1

df = get_data([test_image_path])
if len(df) == 0:
	raise ValueError("No test images found at " + test_image_path)

depth_predictor = Depth4Predictor(input_shape=input_size + (3,))

model = create_simple_dehaze10(input_shape=input_size + (4,))
model.summary(line_length=150)
print("loading best weights %s" % (weights_file))
model.load_weights(weights_file)

mkdir_p(out_dir)
print("Output directory: %s"%out_dir)




def crop_center(img1, size):
	h,w = size
	dy = (img1.shape[0] - h) // 2
	dx = (img1.shape[1] - w) // 2
	return img1[dy:dy + h, dx:dx + w, :]

tile_mask = None

def get_tile_mask():
	global tile_mask
	if tile_mask is not None:
		return tile_mask
	a = np.zeros(input_size[0:2],dtype=np.float32)
	for y in range(a.shape[0]):
		yd = y if y < a.shape[0]//2 else a.shape[0] - y
		for x in range(a.shape[1]):
			xd = x if x < a.shape[1]//2 else a.shape[1] - x
			d = min(yd,xd)
			v = 1.0 if d > tile_overlap else float(d) / tile_overlap
			v = 0.0001 if v == 0. else v
			a[y,x] = v
	tile_mask = np.expand_dims(a, axis=2)
	return tile_mask


def predict(img):
	# tile input image (rgbf)
	#n_tiles=(3,4)
	n_tiles = ((img.shape[0] + tile_core[0] - 1) / tile_core[0], (img.shape[1] + tile_core[1] - 1) / tile_core[1])
	sum = np.zeros(img.shape[0:2] + (3,),dtype=np.float32)
	div = np.zeros(img.shape[0:2] + (1,),dtype=np.float32)
	for k in range(n_tiles[0]):
		y0 = k * img.shape[0] / n_tiles[0]
		y0 = y0 if y0 + input_size[0] < img.shape[0] else img.shape[0] - input_size[0]
		for i in range(n_tiles[1]):
			x0 = i * img.shape[1] / n_tiles[1]
			x0 = x0 if x0 + input_size[1] < img.shape[1] else img.shape[1] - input_size[1]
			crop = img[y0:y0 + input_size[0], x0:x0 + input_size[1], :]
			crop = np.array(crop, np.float32)

			batch = []
			batch.append(crop)
			batch.append(flip_axis(crop, axis=0))
			batch.append(flip_axis(crop, axis=1))
			batch.append(flip_axis(flip_axis(crop, axis=1), axis=0))
			batch.append(np.rot90(crop))
			batch = np.array(batch)

			predb = model.predict_on_batch(batch)

			add_tile(x0, y0, predb[0], sum, div)
			add_tile(x0, y0, flip_axis(predb[1], axis=0), sum, div)
			add_tile(x0, y0, flip_axis(predb[2], axis=1), sum, div)
			add_tile(x0, y0, flip_axis(flip_axis(predb[3], axis=1), axis=0), sum, div)
			pred = predb[4]
			pred = np.rot90(pred)
			pred = np.rot90(pred)
			pred = np.rot90(pred)
			add_tile(x0, y0, pred, sum, div)

	# mittelwert
	avg = sum / div
	return avg


def add_tile(x0, y0, pred, sum, div):
	tile = pred[0:,0:,:]
	mask = get_tile_mask()
	sum[y0:y0 + input_size[0], x0:x0 + input_size[1], :] += tile*mask
	div[y0:y0 + input_size[0], x0:x0 + input_size[1], :] += mask


def predict_tile(x):
	predb = model.predict_on_batch(np.expand_dims(x, axis=0))
	pred = predb[0]
	return pred


start = time.time()
for row in df.iterrows():
	fname = row[1].path_in
	bgr_in = cv2.imread(fname)
	bgr = skimage.transform.resize(bgr_in, scale_down_to + (3,), order=3, preserve_range=True, mode='reflect')
	rgbf =  bgr[..., ::-1] / 255.0
	depth = depth_predictor.predict(rgbf)
	x = np.append(rgbf, np.expand_dims(depth,2), axis=2)
	y = predict(x)
	# restore input scale
	y = skimage.transform.resize(y, bgr_in.shape[0:2], order=3, preserve_range=True, mode='reflect')
	dehazed = rgbf2bgr(y)
	outfile = '%s/%s.png' % (out_dir, os.path.basename(fname)[0:-4],)
	print("Writing " + outfile)
	cv2.imwrite(outfile, dehazed)

end = time.time()
s_per_img=(end-start)/len(df)
# write readme
with open(out_dir + "/readme.txt", "w") as text_file:
	text_file.write("runtime per image [s] : %f\n" % (s_per_img,))
	text_file.write("CPU[1] / GPU[0] : 0\n")
	text_file.write("Extra Data [1] / No Extra Data [0] : 1\n")
	text_file.write("A depth estimation CNN and an UNET style CNN working together.\n")
	text_file.write(weights_file)
	text_file.write("\n")

print("done")
