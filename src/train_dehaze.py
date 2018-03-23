# coding=utf-8
import os
import sys
import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from generator_dehaze import ImageSequence, get_data, rgbf2bgr
from keras_custom import init_seeds, PSNRLoss
from kf_util import Logger, mkdir_p, find_best_weights
from predict_depth import Depth4Predictor
from models_dehaze import create_simple_dehaze5, create_simple_dehaze10
from keras_contrib.losses import DSSIMObjective


initial_epoch=0
epochs=800
lr_initial=0.0005
batch_size=7
seed=123
scale_down_to=(1200,1972)
input_size=(512,512)

run_name='dehaze10_inout2_dssim23_depth4nyu%dx%d_s%d_%dx%d_lr%f_%d' % (scale_down_to + (seed,) + input_size + (lr_initial, batch_size,))
log_dir='../log/' + run_name
weights_file = log_dir + "/dehaze10_weights_{epoch:03d}_{val_loss}.hdf5"

init_seeds(seed)

best_weights = find_best_weights(log_dir)
while initial_epoch == 0 and os.path.exists(log_dir) and best_weights is None:
	s = log_dir.split('#')
	nr = 0 if len(s) == 1 else s[1]
	log_dir = '%s#%d'%(s[0], int(nr) +1)

mkdir_p(log_dir)
logger = Logger()
logger.set_logfile(log_dir+"/out.log")
sys.stdout = logger

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.allow_soft_placement=True
sess = tf.Session(config=config)
K.set_session(sess)



def get_callbacks(log_dir):
	mkdir_p(log_dir)
	return [
		EarlyStopping(min_delta=1e-4,
										  monitor='PSNRLoss',
										  mode='max',
										  patience=26,
										  verbose=1),
		ReduceLROnPlateau(factor=0.2,
												monitor='PSNRLoss',
												mode='max',
												patience=12,
												verbose=1,
												epsilon=1e-4),
		ModelCheckpoint(
			filepath=weights_file,
			monitor='val_loss',
			mode='min',
			verbose=1,
			save_best_only=True,
			# filepath=log_dir+'/best_weights_{epoch:03d}_{val_loss}.hdf5',
			# save_best_only=False,
			save_weights_only=True,
		),
		ModelCheckpoint(
			filepath=log_dir+'/best_psnr_{epoch:03d}_{val_PSNRLoss}.hdf5',
			monitor='val_PSNRLoss',
			mode='max',
			verbose=1,
			save_best_only=True,
			# save_best_only=False,
			save_weights_only=True,
		),
		TensorBoard(log_dir=log_dir)]

def save_prediced_samples(df, model, depth_predictor, ouput_dir, n_samples=10):
	seq = ImageSequence(df, depth_predictor, scale_down_to=scale_down_to, input_size=input_size, batch_size=batch_size,
						depth_cache=depth_cache,
						random_flip=False,
						random_rot90=False,
						random_crop=False)
	mkdir_p(ouput_dir)
	i=0
	for xb,yb in seq:
		ypb = model.predict_on_batch(xb)
		for x,y,yp in zip(xb,yb,ypb):
			img_all=np.hstack((x[:,:,0:3],y[:,:,0:3],yp[:,:,0:3]))
			bgr = rgbf2bgr(img_all)
			cv2.imwrite('%s/%d.jpg'%(ouput_dir,i,),bgr)
			i += 1
			if i >= n_samples:
				return


df = get_data(['../data/indoor/train/haze','../data/outdoor/train/haze'])
df = df.sample(frac=1) # shuffle

df_train, df_valid = train_test_split(df, test_size=0.2, random_state=seed)

depth_predictor = Depth4Predictor(input_shape=input_size + (3,))
depth_cache = log_dir + "/tmp"
train_seq = ImageSequence(df_train, depth_predictor, scale_down_to=scale_down_to, input_size=input_size, depth_cache=depth_cache, batch_size=batch_size, randomize_epoch=True)
valid_seq = ImageSequence(df_valid, depth_predictor, scale_down_to=scale_down_to, input_size=input_size, depth_cache=depth_cache, batch_size=batch_size)

model = create_simple_dehaze10(input_shape=train_seq.input_shape)
model.summary(line_length=150)
print("Output directory: %s"%log_dir)

model.compile(loss=DSSIMObjective(kernel_size=23), optimizer=RMSprop(lr=lr_initial), metrics=[PSNRLoss,DSSIMObjective(kernel_size=23)])

if best_weights is not None:
	print("loading weights %s" % (best_weights))
	model.load_weights(best_weights)

model.fit_generator(
	generator=train_seq,
	verbose=1,
	initial_epoch=initial_epoch,
	epochs=epochs,
	workers=1, # add more workers depending on CPU / GPU performance
	max_queue_size=10,
	steps_per_epoch=len(train_seq)*40,
	validation_data=valid_seq,
	validation_steps=len(valid_seq)*2,
	callbacks=get_callbacks(log_dir=log_dir)
)

best_weights = find_best_weights(log_dir)
print("loading best weights %s" % (best_weights))
model.load_weights(best_weights)
save_prediced_samples(df_valid, model, depth_predictor, log_dir + "/val")
save_prediced_samples(df_train, model, depth_predictor, log_dir + "/train")

print("done")
