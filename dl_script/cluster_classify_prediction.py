import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import os
import time

from sklearn.cluster import DBSCAN

from keras.optimizers import Adam

from cluster_classify_model import cluster_classify_model
from cluster_classify_util import *
from cluster_classify_train import *
from test_on_udacity_data import *

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"my_loss": my_loss})


def predict(model,lidar, thresh=0.5):
    lidar, labels = cluster(lidar)
    list_clusters = list(set(labels))
    nb_clusters = len(list_clusters)
    if nb_clusters == 0:
    	return np.array([])
    
    list_of_cluster = np.array([lidar[labels == list_clusters[i]] for i in range(nb_clusters)] )
    
    centers = np.zeros((nb_clusters,2))
    imgs = np.zeros((nb_clusters, 64,64,2))
    
    for i in range(nb_clusters):
        img, center = discretize(list_of_cluster[i])
        imgs[i] = img
        centers[i] = center

    features = model.predict(imgs)
    #print('features.shape: ', features.shape)
    features_thresh = features[features[:,0] >= thresh]
    centers_thresh = centers[features[:,0] >= thresh]
    
    boxes = np.array([gt_box_decode(features_thresh[i], centers_thresh[i], z_min = -1.5) for i in range(len(features_thresh)) ])
    
    return boxes

def predict_test_set(model, test_dir, pred_dir):

	if not os.path.exists(pred_dir):
		os.mkdir(pred_dir)
	
	print('start predicting ....')
	start =  time.time()
	nb = 0
	for bag in os.listdir(test_dir):
		bag_dir = os.path.join(test_dir, bag)
		
		pred_bag_dir = os.path.join(pred_dir, bag)
		if not os.path.exists(pred_bag_dir):
			os.mkdir(pred_bag_dir)

		for f in os.listdir(bag_dir):
			nb+=1
			lidar_file = os.path.join(bag_dir, f)
			lidar = np.load(lidar_file)

			boxes = predict(model, lidar)
			box_file = os.path.join(pred_bag_dir, f.replace('lidar', 'boxes') )

			np.save(box_file, boxes)
	print('End prediction. Number of frame: {0}. Total time: {1}. Time per frame {2}'.format(nb, int(time.time()-start), (time.time()-start)/nb))


if __name__ == "__main__":
	
	#lidar = np.load('./data/training_didi_data/car_train_edited/bmw_sitting_still/lidar/lidar_100.npy')
	#gtbox = np.load('./data/training_didi_data/car_train_gt_box_edited/bmw_sitting_still/gt_boxes3d/gt_boxes3d_100.npy')
	#viz_mayavi_with_labels(lidar, gtbox)


	model = load_model('./saved_model/last_model.h5') 
	#boxes = predict(model, lidar)

	#viz_mayavi_with_labels(lidar, boxes)
	test_dir = './data/test_cars/'
	pred_dir = './data/pred_box_cars/'

	predict_test_set(model, test_dir, pred_dir)
