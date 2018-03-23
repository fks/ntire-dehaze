# coding=utf-8
import keras.backend as K
from keras_contrib.losses import DSSIMObjective
import tensorflow as tf

def PSNRLoss(y_true, y_pred):
	return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def sparse_mse(y_true, y_pred):
	nonzero = K.sign(y_true)
	n = K.sum(nonzero, axis=None)
	diff = y_pred - y_true
	diff = diff * nonzero
	loss = K.sum(K.square(diff), axis=None) / n
	return loss

def sparse_msle(y_true,y_pred):
	nonzero = K.sign(y_true)
	n = K.sum(nonzero, axis=None)
	pred_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
	true_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
	diff = pred_log - true_log
	diff = diff * nonzero
	loss = K.sum(K.square(diff), axis=None) / n
	return loss


def sparse_PSNR(y_true, y_pred):
	return -10. * K.log(sparse_mse(y_true, y_pred) + K.epsilon()) /  K.log(10.)

def sparse_SSIM(y_true, y_pred):
	y_pred_masked = K.switch(y_true == 0., y_true, y_pred)
	return DSSIMObjective(kernel_size=23)(y_true, y_pred_masked)

def init_seeds(seed=42):
	import random
	import numpy as np
	if seed is not None:
		print("Using seed %d"%seed)
	else:
		random.seed(None)
		seed = int(random.random()*1234567)
		random.seed(seed)
		print("Using random seed %d"%seed)

	random.seed(seed)
	np.random.seed(seed)
	tf.set_random_seed(seed)
