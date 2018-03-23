# coding=utf-8
import os
import sys

import keras.backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop

from generator_depth import DepthImageSequence, get_depth_data
from keras_custom import init_seeds,sparse_PSNR, sparse_msle
from kf_util import Logger, mkdir_p, find_best_weights
from models_depth import create_unetd_4


initial_epoch=0
epochs=800
lr_initial=0.001
batch_size=20
seed=123
scale_down_to=None
input_size=(384,384)


run_name='depth4_nyu_smsle_ep1_s%d_%dx%d_lr%f_%d' % ((seed,) + input_size + (lr_initial, batch_size,))
log_dir='../log/' + run_name
weights_file = log_dir + "/depth4_weights_{epoch:03d}_{val_loss}.hdf5"

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
										  monitor='val_loss',
										  mode='min',
										  patience=29,
										  verbose=1),
		ReduceLROnPlateau(factor=0.2,
												monitor='val_loss',
												mode='min',
												patience=9,
												verbose=1,
												epsilon=1e-5),
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
		TensorBoard(log_dir=log_dir)]


model = create_unetd_4(input_shape=input_size + (3,))
model.summary(line_length=150)
print("Output directory: %s"%log_dir)

df_train = get_depth_data(['../data/nyudepthv2/train'])
df_train = df_train.sample(frac=1) # shuffle

df_valid = get_depth_data(['../data/nyudepthv2/val'])
df_valid.sample(frac=1) # shuffle

train_seq = DepthImageSequence(df_train, scale_down_to=scale_down_to, max_depth=10.0, input_size=input_size, batch_size=batch_size, random_flip=True, random_rot90=True, random_color=True, augment_bright_contrast=True, randomize_epoch=True)
valid_seq = DepthImageSequence(df_valid, scale_down_to=scale_down_to, max_depth=10.0, input_size=input_size, batch_size=batch_size, random_flip=True, random_rot90=True, random_color=True, augment_bright_contrast=True, )


model.compile(loss=sparse_msle, optimizer=RMSprop(lr=lr_initial), metrics=[sparse_PSNR])


if best_weights is not None:
	print("loading weights %s" % (best_weights))
	model.load_weights(best_weights)

model.fit_generator(
	generator=train_seq,
	verbose=1,
	initial_epoch=initial_epoch,
	epochs=epochs,
	workers=1,  # add more workers depending on CPU / GPU performance
	max_queue_size=10,
	steps_per_epoch=len(train_seq)*1,
	validation_data=valid_seq,
	validation_steps=len(valid_seq)*1,
	callbacks=get_callbacks(log_dir=log_dir)
)


print("done")