# coding=utf-8
import glob
import os

import cv2
import numpy as np
import skimage
import skimage.transform
from kf_util import mkdir_p, flip_axis
from models_depth import create_unetd_4


class DepthPredictor():
	def __init__(self, depth_model, weights_file):
		self.model = depth_model
		self.model_input_shape = depth_model.input.shape[1:4].as_list()
		self.weights_file = weights_file
		print("loading weights %s" % (weights_file))
		self.model.load_weights(weights_file)
		self.model._make_predict_function()



	def predict(self, rgbf_img):
		"""Predict depth of image.
			Parameters
			----------
			rgbf_img : ndarray
				Input image (channels last, rgb with component range 0...1). Will be scaled down to model input size.
			Returns
			-------
			resized : ndarray
				Depth prediction, same size als input image dtype=float32
		"""
		x_batch = []
		x_in = skimage.transform.resize(rgbf_img, self.model_input_shape, order=3, preserve_range=True, mode='reflect')
		# add the unmodified image
		x_batch.append(x_in)
		# add several augmentations
		x_batch.append(flip_axis(x_in, axis=0))
		x_batch.append(flip_axis(x_in, axis=1))
		x_batch.append(flip_axis(flip_axis(x_in, axis=1), axis=0))

		x_batch = np.array(x_batch, np.float32)
		y_batch = self.model.predict_on_batch(x_batch)

		# undo augmentation
		y_batch[1] = flip_axis(y_batch[1], axis=0)
		y_batch[2] = flip_axis(y_batch[2], axis=1)
		y_batch[3] = flip_axis(flip_axis(y_batch[3], axis=1), axis=0)

		y_batch = np.mean(y_batch, axis=0)[:,:,0]
		y_img = skimage.transform.resize(y_batch, rgbf_img.shape[0:2], order=3, preserve_range=True, mode='reflect')
		return y_img


class Depth4Predictor(DepthPredictor):
	"""Using depth model 4 with pretrained weights.
	"""
	def __init__(self, input_shape, weights_file='../pretrained/depth4_weights_085_0.00295022252195.hdf5'):
		depth_model = create_unetd_4(input_shape=input_shape)
		DepthPredictor.__init__(self, depth_model, weights_file)


def predict_files(filenames, model, weights_file, output_dir, max_depth=100., gray_scale = 5.):
	mkdir_p(output_dir)
	predictor = DepthPredictor(model, weights_file)
	for file in filenames:
		bgr = cv2.imread(file)
		rgb = bgr[..., ::-1]
		rgbf = np.array(rgb,dtype=np.float32) / 255.0
		depth = predictor.predict(rgbf)
		depth *= max_depth # usually now in meter
		img_gray = np.clip(depth*gray_scale, 0.,255.0).astype(np.uint8)
		out_name = os.path.join(output_dir, os.path.basename(file))
		print("writing " + out_name)
		cv2.imwrite(out_name,img_gray)

		
# if __name__ == '__main__':
# 	input_size=(384,384)
# 	run_name='depth4_nyu_smsle_ep1_s123_384x384_lr0.001000_20'
# 	#run_name='depth5_nyu_smsle_ep1_s123_384x384_lr0.000500_16'
# 	log_dir='../log/' + run_name
# 	out_dir=os.path.join(log_dir, "depth085")
# 	#weights_file = log_dir + "/best_weights_048_0.00319335149425.hdf5"
# 	weights_file = log_dir + "/best_weights_085_0.00295022252195.hdf5"
#
# 	init_seeds(123)
#
# 	config = tf.ConfigProto()
# 	config.gpu_options.allow_growth=True
# 	sess = tf.Session(config=config)
# 	K.set_session(sess)
#
# 	model = create_unetd_4(input_shape=input_size + (3,))
# 	files = glob.glob('../data/outdoor/test/*.png')
# 	predict_files(files, model, weights_file, out_dir, max_depth=10., gray_scale=25.)