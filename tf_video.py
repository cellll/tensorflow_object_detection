import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util


class TF_DEMO:
	
	def __init__(self):
		self.MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
		#self.MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
		#self.MODEL_NAME = 'rfcn_resnet101_coco_11_06_2017'
		#self.MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
		#self.MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
		self.MODEL_FILE = self.MODEL_NAME + '.tar.gz'
		self.DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

		# Path to frozen detection graph. This is the actual model that is used for the object detection.
		self.PATH_TO_CKPT = self.MODEL_NAME + '/frozen_inference_graph.pb'

		# List of the strings that is used to add correct label for each box.
		self.PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
		self.NUM_CLASSES = 90

		#self.download_models()
		self.tf_init()
		print ("Init complete")
		print ("Model : {}".format(self.MODEL_NAME))

	def download_models(self):
		print ("Start download models")
		opener = urllib.request.URLopener()
		opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, self.MODEL_FILE)
		tar_file = tarfile.open(self.MODEL_FILE)
		for file in tar_file.getmembers():
		    file_name = os.path.basename(file.name)
		    if 'frozen_inference_graph.pb' in file_name:
		        tar_file.extract(file, os.getcwd())

	def tf_init(self):
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
		    od_graph_def = tf.GraphDef()
		    with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
		        serialized_graph = fid.read()
		        od_graph_def.ParseFromString(serialized_graph)
		        tf.import_graph_def(od_graph_def, name='')

		label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
		self.category_index = label_map_util.create_category_index(categories)

		self.PATH_TO_TEST_IMAGES_DIR = 'test_images'
		self.TEST_IMAGE_PATHS = [ os.path.join(self.PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 6) ]

		# Size, in inches, of the output images.
		self.IMAGE_SIZE = (15, 10)

		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
	    	self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        	self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

		self.sess = tf.Session(graph=self.detection_graph, config=tf.ConfigProto(gpu_options=gpu_options))



	def detect(self, frame):

    		s=time.time()

    		img_exp = np.expand_dims(frame, axis=0)

    		#print ("Image open : {}".format(time.time()-s))


	    		#image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
			#boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
			#scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        		#classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
			#num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

		(boxes, scores, classes, num_detections) = self.sess.run(
		    [self.boxes, self.scores, self.classes, self.num_detections],
		    feed_dict={self.image_tensor: img_exp})
            

		print ("Elapsed time : {}".format(time.time()-s))

		vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
		

		return frame











