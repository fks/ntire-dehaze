# coding=utf-8
import traceback

import h5py
import skimage.transform
from keras.utils import Sequence
import numpy as np
import cv2
import glob
import pandas as pd
import os

from kf_util import flip_axis

def rgbf2bgr(rgbf):
	t = rgbf*255.0
	t = np.clip(t, 0.,255.0)
	bgr = t.astype(np.uint8)[..., ::-1]
	return bgr

def get_depth_data(paths=None):
	path_in=[]
	if paths is None:
		paths = ['../data/nyudepthv2/train']
	for path in paths:
		for file in glob.glob(path+"/*/*.h5"):
			path_in.append(file)
	df = pd.DataFrame(path_in,columns=['path_in'])
	return df

def read_rgbd(filename):
	f = h5py.File(filename, 'r')
	rgb = np.array(f['rgb'], dtype=np.float32) #channels_first
	rgb = np.moveaxis(rgb, 0, 2) #channels_last
	depth = np.array(f['depth'], dtype=np.float32)
	depth = np.expand_dims(depth,axis=2)
	f.close()
	return rgb, depth


class DepthImageSequence(Sequence):
	def __init__(self, df, batch_size=8, input_size=(128, 128),
				 crop_mode='random',
				 random_flip=False,
				 random_rot90=False,
				 scale_down_to=None,
				 random_color=False,
				 augment_bright_contrast=False,
				 max_depth = 10.0, #max depth in m
				 randomize_epoch=False):
		# self.random_squeeze = random_squeeze
		self.max_depth = max_depth
		self.random_color = random_color
		self.augment_bright_contrast = augment_bright_contrast
		self.random_rot90 = random_rot90
		self.random_flip = random_flip
		self.scale_down_to = scale_down_to
		self.randomize_epoch = randomize_epoch
		self.input_shape = input_size + (3,)
		self.batch_size = batch_size
		self.crop_mode = crop_mode
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

	def random_color_change(self, ximg):
		random_color = np.random.uniform(0.6, 1.2, size=3)
		return ximg * random_color

	def random_hsv_change(self, ximg):
		imghsv = cv2.cvtColor(ximg, cv2.COLOR_RGB2HSV).astype("float32")
		(h, s, v) = cv2.split(imghsv)
		ss = np.random.random() * 0.7
		s = s * (ss + 0.3)
		s = np.clip(s,0,255)
		vs = np.random.random() * 0.9
		v = v * (vs + 0.8)
		v = np.clip(v,0,255)
		imghsv = cv2.merge([h,s,v])
		imgrgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2RGB)
		return imgrgb

	def random_bright_contrast(self, ximg):
		alpha = (np.random.random()*0.8 + 0.2)
		beta = np.random.random()*150.0
		xf = ximg.astype("float32")*alpha + beta
		xf = np.clip(xf,0.0,255.0)
		ximg = xf.astype("uint8")
		return ximg

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
			ximgo, yimgo = read_rgbd(d.path_in)
			if self.scale_down_to == 'random':
				if np.random.random() < 0.5:
					h = int(np.random.uniform(self.input_shape[0]+8, ximgo.shape[0]))
					w = int(np.random.uniform(self.input_shape[1]+8, ximgo.shape[1]))
					ximgo = skimage.transform.resize(ximgo, (h,w), order=3, preserve_range=True, mode='reflect')
					yimgo = skimage.transform.resize(yimgo, (h,w), order=0, preserve_range=True, mode='reflect')
			elif self.scale_down_to is not None:
				ximgo = skimage.transform.resize(ximgo, self.scale_down_to, order=3, preserve_range=True, mode='reflect')
				yimgo = skimage.transform.resize(yimgo, self.scale_down_to, order=0, preserve_range=True, mode='reflect')

			if self.crop_mode == 'random':
				ximg,yimg = self.crop_rand(ximgo,yimgo)
			elif self.crop_mode == 'center':
				ximg,yimg = self.crop_center(ximgo,yimgo)
			else:
				raise ValueError("Unknown crop mode " + self.crop_mode)

			if self.augment_bright_contrast:
				if np.random.random() < 0.5:
					ximg = self.random_bright_contrast(ximg)
			if self.random_color:
				if np.random.random() < 0.5:
					ximg = self.random_color_change(ximg)
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

		x_batch = np.array(x_batch, np.float32) / 255.0
		y_batch = np.array(y_batch, np.float32) / self.max_depth
		return x_batch,y_batch

	def on_epoch_end(self):
		self.epoch += 1
		if self.randomize_epoch:
			self.df = self.df.sample(frac=1)  # shuffle


def show_samples():
	df = get_depth_data(['../data/nyudepthv2/train'])
	df = df.sample(frac=1, random_state=42)  # shuffle
	train_seq = DepthImageSequence(df, input_size=(384, 384), scale_down_to='random', random_flip=True, random_rot90=True, random_color=False, augment_bright_contrast=True, crop_mode='random')

	import matplotlib.pylab as plt
	plt.figure(figsize=(20, 20))
	for x_batch, y_batch in train_seq:
		for x, y in zip(x_batch, y_batch):
			plt.imshow(rgbf2bgr(x[:, :, 0:3]))
			plt.show(block=False)
			plt.imshow(y[:,:,0])
			plt.show(block=False)


#if __name__ == '__main__':
	#show_samples()

