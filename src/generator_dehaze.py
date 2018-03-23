# coding=utf-8
import skimage.transform
from keras.utils import Sequence
import numpy as np
import cv2
import glob
import pandas as pd
import os

from kf_util import mkdir_p


def rgbf2bgr(rgbf):
	t = rgbf*255.0
	t = np.clip(t, 0.,255.0)
	bgr = t.astype(np.uint8)[..., ::-1]
	return bgr

def rgbf2rgb(rgbf):
	t = rgbf*255.0
	t = np.clip(t, 0.,255.0)
	rgb = t.astype(np.uint8)
	return rgb

def flip_axis(x, axis):
	x = np.asarray(x).swapaxes(axis, 0)
	x = x[::-1, ...]
	x = x.swapaxes(0, axis)
	return x

def get_data(paths=None):
	haze=[]
	if paths is None:
		paths = ['../data/indoor/train/haze']
	for path in paths:
		for file in glob.glob(path+"/*.jpg"):
			haze.append(file)
		for file in glob.glob(path+"/*.JPG"):
			haze.append(file)
		for file in glob.glob(path+"/*.png"):
			if not ".cache" in file:
				haze.append(file)
	df = pd.DataFrame(haze,columns=['path_in'])
	df['path_out'] = df['path_in'].apply(lambda f: os.path.dirname(f) + "/../nohaze/" + os.path.basename(f))
	return df


class ImageSequence(Sequence):
	def __init__(self, df, depth_predictor, batch_size=8, input_size=(512, 512),
				 random_crop=True,
				 random_flip=True,
				 random_rot90=True,
				 depth_cache=None,
				 scale_down_to=None,
				 randomize_epoch=False):
		self.depth_predictor = depth_predictor
		self.depth_cache= depth_cache
		if depth_cache is not None:
			mkdir_p(depth_cache)
		self.random_rot90 = random_rot90
		self.random_flip = random_flip
		self.scale_down_to = scale_down_to

		self.randomize_epoch = randomize_epoch
		self.input_shape = input_size + (3,)
		if depth_predictor is not None:
			self.input_shape = input_size + (self.input_shape[-1]+1,)
		self.batch_size = batch_size
		self.random_crop = random_crop
		self.df = df
		self.epoch = 0

	def crop_rand(self, img1,img2):
		assert img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]
		if img1.shape[0] == self.input_shape[0] and img1.shape[1] == self.input_shape[1]:
			return img1, img2
		w = self.input_shape[0]
		h = self.input_shape[1]
		dx = img1.shape[0] - w
		dy = img1.shape[1] - h
		if dx > 0:
			dx = np.random.randint(0, dx)
		if dy > 0:
			dy = np.random.randint(0, dy)
		return img1[dx:dx + w, dy:dy + h, :],img2[dx:dx + w, dy:dy + h, :]

	def crop_center(self, img1,img2):
		if img1.shape[0] == self.input_shape[0] and img1.shape[1] == self.input_shape[1]:
			return img1, img2
		assert img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1]
		w = self.input_shape[0]
		h = self.input_shape[1]
		dx = (img1.shape[0] - w) // 2
		dy = (img1.shape[1] - h) // 2
		return img1[dx:dx + w, dy:dy + h, :],img2[dx:dx + w, dy:dy + h, :]

	# number of steps
	def __len__(self):
		return (len(self.df) + self.batch_size - 1) // self.batch_size

	def read_bgr_scaled(self,f):
		if not os.path.exists(f):
			return None
		cache_name = os.path.dirname(f) + "/" + os.path.basename(f)[0:-4] + ".cache_%dx%d.png"%self.scale_down_to
		if os.path.exists(cache_name):
			bgr = cv2.imread(cache_name)
			return bgr
		bgr = cv2.imread(f)
		bgr = skimage.transform.resize(bgr, self.scale_down_to, order=3, preserve_range=True, mode='reflect')
		cv2.imwrite(cache_name, bgr.astype(np.uint8))
		return bgr

	def read_scaled_images_bgr(self, row):
		bgr = self.read_bgr_scaled(row.path_in)
		obgr = self.read_bgr_scaled(row.path_out)
		return bgr, obgr

	def read_scaled_images_rgb(self, row):
		bgr,obgr = self.read_scaled_images_bgr(row)
		return bgr[..., ::-1], obgr[..., ::-1] if obgr is not None else None

	def read_images_bgr(self, row):
		bgr = cv2.imread(row.path_in)
		obgr = cv2.imread(row.path_out)
		return bgr, obgr

	def read_images_rgb(self, row):
		bgr = cv2.imread(row.path_in)
		if self.out_16bit_gray:    # assume 16bit grayscale
			obgr = cv2.imread(row.path_out, -1 )
			return bgr[..., ::-1], np.expand_dims(obgr,axis=2)
		else:
			obgr = cv2.imread(row.path_out)
			return bgr[..., ::-1], obgr[..., ::-1]

	def read_images_hsv(self, row):
		bgr, obgr = self.read_images_bgr(row)
		ximg = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) #bgr[..., ::-1]
		yimg = cv2.cvtColor(obgr, cv2.COLOR_BGR2HSV) #obgr[..., ::-1]
		return ximg,yimg


	# called every step
	def __getitem__(self, idx):
		if idx >= self.__len__():
			raise IndexError
		x_batch = []
		y_batch = []
		start = idx * self.batch_size
		end = min(start + self.batch_size, len(self.df))
		df_batch = self.df.iloc[start:end]

		for i, d in df_batch.iterrows():
			if self.scale_down_to is None:
				ximgo, yimgo = self.read_images_rgb(d)
			else:
				ximgo, yimgo = self.read_scaled_images_rgb(d)

			ximgf = ximgo.astype(np.float32) / 255.0
			yimgf = yimgo.astype(np.float32) / 255.0
			if self.depth_predictor is not None:
				if self.depth_cache is not None:
					cache_file=os.path.join(self.depth_cache, os.path.basename(d.path_in)[0:-4]+"_depth.npy")
					if os.path.exists(cache_file):
						xdepthf = np.load(cache_file)
					else:
						xdepthf = self.depth_predictor.predict(ximgf)
						np.save(cache_file, xdepthf)
				else:
					xdepthf = self.depth_predictor.predict(ximgf)
				ximgf = np.append(ximgf, np.expand_dims(xdepthf,2), axis=2)


			if self.random_crop:
				ximg,yimg = self.crop_rand(ximgf,yimgf)
			else:
				ximg,yimg = self.crop_center(ximgf,yimgf)

			if self.random_flip:
				if np.random.random() < 0.5:
					ximg = flip_axis(ximg, 0)
					yimg = flip_axis(yimg, 0)
				if np.random.random() < 0.5:
					ximg = flip_axis(ximg, 1)
					yimg = flip_axis(yimg, 1)
			if self.random_rot90:
				if np.random.random() < 0.5:
					ximg = np.rot90(ximg)
					yimg = np.rot90(yimg)

			x_batch.append(ximg)
			y_batch.append(yimg)

		x_batch = np.array(x_batch, np.float32)
		y_batch = np.array(y_batch, np.float32)
		return x_batch,y_batch

	def on_epoch_end(self):
		self.epoch += 1
		if self.randomize_epoch:
			self.df = self.df.sample(frac=1)  # shuffle



# def show_samples():
#   from predict_depth import DepthPredictor
# 	df = get_data()
# 	input_size=(384,384)
# 	run_name='depth4_nyu_smsle_ep1_s123_384x384_lr0.001000_20'
# 	log_dir='/data/ssd2/learn/ntire-haze/log/' + run_name
# 	weights_file = log_dir + "/best_weights_052_0.00326197986727.hdf5"
# 	from models_depth import create_unetd_4
# 	depth_model = create_unetd_4(input_shape=input_size + (3,))
# 	depth_predictor = DepthPredictor(depth_model, weights_file)
#
# 	train_seq = ImageSequence(df, depth_predictor, depth_cache=log_dir + "/tmp", input_size=(512, 512), scale_down_to=(1200,1972))
#
# 	import matplotlib.pylab as plt
# 	plt.figure(figsize=(20, 20))
# 	for x_batch, y_batch in train_seq:
# 		for x, y in zip(x_batch, y_batch):
# 			plt.imshow(rgbf2rgb(x[:, :, 0:3]))
# 			plt.show(block=False)
#
# 			plt.imshow(x[:,:,3])
# 			plt.show(block=False)
#
# 			plt.imshow(rgbf2rgb(y[:,:,0:3]))
# 			plt.show(block=False)
#
#
# if __name__ == '__main__':
# 	show_samples()
#
