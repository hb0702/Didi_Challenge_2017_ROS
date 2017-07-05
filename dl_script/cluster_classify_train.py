import numpy as np
import tensorflow as tf
import keras
import os
import time

from keras.optimizers import Adam
from keras.models import load_model

from keras.callbacks import ModelCheckpoint, CSVLogger

from cluster_classify_model import cluster_classify_model
from cluster_classify_util import *


def data_generator(list_of_cars, list_of_not_cars, list_of_gtboxes):
    '''
    input: list_of_cars, list_of_not_cars, list_of_gtbox
    output: generator of lidar and gtbox
    '''
    nb_cars = len(list_of_cars)
    nb_notcars = len(list_of_not_cars)
    p = 1.*nb_cars/(nb_cars + nb_notcars)


    next_epoch_car = True
    next_epoch_notcar = True
    
    while True:
        coin = np.random.binomial(1, p, size=1)[0]
        if coin == 1:

            if next_epoch_notcar:
                ind_notcar = 0
                indices_notcar = np.arange(nb_notcars)
                np.random.shuffle(indices_notcar)
                
                yield list_of_not_cars[indices_notcar[ind_notcar]], 0, 0
                
                ind_notcar = 1
                next_epoch_notcar = False
            else:
                yield list_of_not_cars[indices_notcar[ind_notcar]], 0, 0
                ind_notcar += 1
                if ind_notcar >= nb_notcars:
                    next_epoch_notcar = True

        else:
            if next_epoch_car:
                ind_car = 0
                indices_car = np.arange(nb_cars)
                np.random.shuffle(indices_car)
                
                yield list_of_cars[indices_car[ind_car]], list_of_gtboxes[indices_car[ind_car]], 1
                
                ind_car = 1
                next_epoch_car = False
            else:
                yield list_of_cars[indices_car[ind_car]], list_of_gtboxes[indices_car[ind_car]], 1
                ind_car += 1
                if ind_car >= nb_cars:
                    next_epoch_car = True


def train_batch_generator(list_of_cars, list_of_not_cars, list_of_gtboxes, batch_size = 32, data_augmentation = True, width = 64, height = 64, nb_channels = 2, nb_features = 7):

    ind = 0
    for lidar_file, gtbox_file, is_car in data_generator(list_of_cars, list_of_not_cars, list_of_gtboxes):
        
        if ind == 0:
            batch_sample = np.zeros((batch_size, height, width, nb_channels))
            batch_label = np.zeros((batch_size, nb_features))

        
        if is_car == 1:
            lidar = np.load(lidar_file)
            gtbox = np.load(gtbox_file)[0]
            # need to implement rotate function and maybe flip also
            if data_augmentation:
                rotate_angle = np.random.rand()*np.pi*2
                flip = np.random.randint(2)

                lidar = rotation_cluster(rotate_angle, lidar, flip)
                gtbox = rotation_cluster(rotate_angle, gtbox, flip)

            img, center = discretize(lidar)
            encode = gt_box_encode(gtbox, center)

            batch_sample[ind] = img
            batch_label[ind] = encode
            #batch_label[ind] = 1

        else:
            lidar = np.load(lidar_file)
            # need to implement rotate function
            if data_augmentation:
                rotate_angle = np.random.rand()*np.pi*2
                flip = np.random.randint(2)

                lidar = rotation_cluster(rotate_angle, lidar, flip)

            img, _ = discretize(lidar)

            batch_sample[ind] = img

        ind += 1
        if ind == batch_size:
            yield batch_sample, batch_label
            ind = 0


def my_loss(y_true, y_pred):


	cls_true,reg_true = tf.split(y_true, [1, 6], 1)
	cls_pred,reg_pred = tf.split(y_pred, [1, 6], 1)

	#cls_loss = -tf.reduce_mean(tf.multiply(y_true,tf.log(y_pred+1e-8)) + tf.multiply(1-y_true,tf.log(1-y_pred+1e-8)))	
	cls_loss = -tf.reduce_mean(tf.multiply(cls_true,tf.log(cls_pred+1e-8)) + tf.multiply(1-cls_true,tf.log(1-cls_pred+1e-8)))
	#seg_loss = -tf.reduce_mean(
	#    tf.multiply(tf.multiply(seg_true,tf.log(seg_pred)) + tf.multiply(1-seg_true,tf.log(1-seg_pred)), weight1))

	diff = tf.sqrt(tf.reduce_mean(tf.squared_difference(reg_true, reg_pred), axis=1, keep_dims=True))
	reg_loss = tf.reduce_mean(tf.multiply(cls_true,diff))

	#total_loss = reg_loss
	#total_loss = cls_loss
	total_loss = cls_loss + reg_loss
	return total_loss 
	#return 1

if __name__ == '__main__':



	car_dir = './data/training_didi_data/car_cluster/'
	not_car_dir =  './data/training_didi_data/not_car_cluster/'
	gtbox_dir = './data/training_didi_data/car_train_gt_box_edited/'

	list_of_cars, list_of_not_cars, list_of_gtboxes = list_of_data(car_dir, not_car_dir, gtbox_dir)

	# list_of_cars =  list_of_cars[:2]
	# list_of_not_cars = list_of_not_cars[:20]
	# list_of_gtboxes = list_of_gtboxes[:2]
	# print(list_of_cars)
	# print(list_of_not_cars)
	# print(list_of_gtboxes)


	#test on just two sample
	#list_of_view = ['./data/training_didi_data/car_train_edited/suburu_leading_front_left/view/view_281.npy',
	#				'./data/training_didi_data/car_train_edited/cmax_following_long/view/view_6631.npy']


	batch_size = 64
	epochs = 100
	augmentation = False
	
	num_frame = 2*len(list_of_cars)
	steps_per_epoch = int(num_frame/batch_size)
	
	continue_training = True
	#saved_model = 'saved_model/last_model.h5'
	saved_model = 'saved_model/model_for_car_classifier_30_June_10_199.h5'

	if not continue_training:
		print('Initiate training')
		model = cluster_classify_model(summary = True)
		opt = Adam(lr=1e-4)
		#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.compile(optimizer=opt, loss=my_loss)
		
	else:
		print('Continue training')
		from keras.utils.generic_utils import get_custom_objects
		get_custom_objects().update({"my_loss": my_loss})
		
		model = load_model(saved_model)
		opt = Adam(lr=1e-4)
	# #keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		model.compile(optimizer=opt, loss=my_loss)
	

	checkpointer = ModelCheckpoint('saved_model/model_for_car_classifier_200_{epoch:02d}.h5')
	#logger = CSVLogger(filename='saved_model/model_May_29_450.csv')

	print('Start training - batch_size : {0} - num_frame : {1} - steps_per_epoch : {2}'.format(batch_size,num_frame,steps_per_epoch))
	start = time.time()

	model.fit_generator(generator=train_batch_generator(list_of_cars, list_of_not_cars, list_of_gtboxes, batch_size = batch_size, data_augmentation = augmentation),
                       steps_per_epoch=steps_per_epoch,
                       epochs=epochs,
                       callbacks=[checkpointer])#, logger])

	print('End training - during time: {0} minutes'.format( int((time.time() - start)/60) ))
	model.save("saved_model/last_model.h5")

