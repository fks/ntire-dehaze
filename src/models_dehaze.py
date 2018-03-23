# coding=utf-8
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Add, Dropout, concatenate

def unet_down_1(filter_count, inputs, activation='relu', pool=(2, 2), n_layers=3):
	down = inputs
	for i in range(n_layers):
		down = Conv2D(filter_count, (3, 3), padding='same', activation=activation)(down)
		down = BatchNormalization()(down)

	if pool is not None:
		x = MaxPooling2D(pool, strides=pool)(down)
	else:
		x = down
	return (x, down)

def unet_up_1(filter_count, inputs, down_link, activation='relu', n_layers=3):
	reduced = Conv2D(filter_count, (1, 1), padding='same', activation=activation)(inputs)
	up = UpSampling2D((2, 2))(reduced)
	up = BatchNormalization()(up)
	link = Conv2D(filter_count, (1, 1), padding='same', activation=activation)(down_link)
	link = BatchNormalization()(link)
	up = Add()([up,link])
	for i in range(n_layers):
		up = Conv2D(filter_count, (3, 3), padding='same', activation=activation)(up)
		up = BatchNormalization()(up)
	return up


def create_simple_dehaze5(input_shape=(512,512,4)):

	n_layers_down = [2, 3, 3, 3, 3, 3]
	n_layers_up = [2, 3, 3, 3, 3, 3]
	n_filters_down = [16,32,64, 96, 144, 192]
	n_filters_up = [16,32,64, 96, 144, 192]
	n_filters_center=256
	n_layers_center=4
	print('n_filters_down:%s  n_layers_down:%s'%(str(n_filters_down),str(n_layers_down)))
	print('n_filters_center:%d  n_layers_center:%d'%(n_filters_center, n_layers_center))
	print('n_filters_up:%s  n_layers_up:%s'%(str(n_filters_up),str(n_layers_up)))
	activation='relu'
	inputs = Input(shape=input_shape)
	x = inputs
	x = BatchNormalization()(x)
	xbn = x
	depth = 0
	back_links = []
	for n_filters in n_filters_down:
		n_layers = n_layers_down[depth]
		x, down_link = unet_down_1(n_filters, x, activation=activation, n_layers=n_layers)
		back_links.append(down_link)
		depth += 1

	center, _ = unet_down_1(n_filters_center, x, activation=activation, pool=None, n_layers=n_layers_center)


	# center
	x1 = center
	while depth > 0:
		depth -= 1
		link = back_links.pop()
		n_filters = n_filters_up[depth]
		n_layers = n_layers_up[depth]
		x1 = unet_up_1(n_filters, x1, link, activation=activation, n_layers=n_layers)
		if depth <= 1:
			x1 = Dropout(0.25)(x1)

	x1 = concatenate([x1,xbn])
	x1 = Conv2D(16, (3, 3), padding='same', activation=activation)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Conv2D(16, (3, 3), padding='same', activation=activation)(x1)
	x1 = BatchNormalization()(x1)

	x1 = Conv2D(3, (1, 1), activation='sigmoid')(x1)
	model = Model(inputs=inputs, outputs=x1)
	return model

def create_simple_dehaze10(input_shape=(512,512,4)):

	n_layers_down = [2, 2, 2, 2, 2, 2]
	n_layers_up = [2, 2, 2, 2, 2, 2]
	n_filters_down = [16,32,64, 96, 144, 192]
	n_filters_up = [16,32,64, 96, 144, 192]
	n_filters_center=256
	n_layers_center=4
	print('n_filters_down:%s  n_layers_down:%s'%(str(n_filters_down),str(n_layers_down)))
	print('n_filters_center:%d  n_layers_center:%d'%(n_filters_center, n_layers_center))
	print('n_filters_up:%s  n_layers_up:%s'%(str(n_filters_up),str(n_layers_up)))
	activation='relu'
	inputs = Input(shape=input_shape)
	x = inputs
	x = BatchNormalization()(x)
	xbn = x
	depth = 0
	back_links = []
	for n_filters in n_filters_down:
		n_layers = n_layers_down[depth]
		x, down_link = unet_down_1(n_filters, x, activation=activation, n_layers=n_layers)
		back_links.append(down_link)
		depth += 1

	center, _ = unet_down_1(n_filters_center, x, activation=activation, pool=None, n_layers=n_layers_center)


	# center
	x1 = center
	while depth > 0:
		depth -= 1
		link = back_links.pop()
		n_filters = n_filters_up[depth]
		n_layers = n_layers_up[depth]
		x1 = unet_up_1(n_filters, x1, link, activation=activation, n_layers=n_layers)
		if depth <= 1:
			x1 = Dropout(0.25)(x1)

	x1 = concatenate([x1,xbn])
	x1 = Conv2D(16, (3, 3), padding='same', activation=activation)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Conv2D(16, (3, 3), padding='same', activation=activation)(x1)
	x1 = BatchNormalization()(x1)

	x1 = Conv2D(3, (1, 1), activation='sigmoid')(x1)
	model = Model(inputs=inputs, outputs=x1)
	return model
