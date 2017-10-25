import numpy as np
import mxnet as mx
from dms.utils.mxnet import image
from dms.model_service.mxnet_model_service import MXNetBaseService

class SSDService(MXNetBaseService):
    '''
    	SSD Service to perform real time multi-object detection using pre-trained
        MXNet SSD model.
    	This class extends Deep Model Service to add custom preprocessing of input
        and preparing the output.
    '''
    def __init__(self, path, ctx=mx.cpu()):
        super(SSDService, self).__init__(path, ctx)
	    self.threshold = 0.2
	    self.input_width = None
        self.input_height = None

    def _preprocess(self, data):
	'''
	    Input image buffer from data is read into NDArray. Then, resized to
        expected shape. Swaps axes to convert image from BGR format to RGB.
	    Returns the preprocessed NDArray as a list for next step, Inference.
	'''
    	# Read input
    	input_image = image.read(data[0])

    	# Save original input image shape.
    	# This is required for preparing the bounding box of the detected object relative to
    	# original input
    	self.input_height = input_image.shape[0]
    	self.input_width = input_image.shape[1]

    	# Resize the input as per requirement (512 x 512)
    	input_shape = self.signature['inputs'][0]['data_shape']
            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]
    	input_image = image.resize(input_image, h, w)

    	# swap BGR to RGB
    	input_image = np.swapaxes(input_image.asnumpy(), 0, 2)
    	input_image = np.swapaxes(input_image, 1, 2)

    	# Prepare MXNet NDArray for doing prediction
    	input_image = mx.nd.array([input_image])
    	return [input_image]

    def _postprocess(self, data):
	'''
	    From the detections, prepares the output in the format of list of
        [(object_class, xmin, ymin, xmax, ymax)]
        object_class is name of the object detected. xmin, ymin, xmax, ymax
        provides the bounding box coordinates.

        Example: [(person, 555, 175, 581, 242), (dog, 306, 446, 468, 530)]
	'''
    	# Read the detections output after forward pass (inference)
    	detections = data[0].asnumpy()
    	result = []
    	for i in range(detections.shape[0]):
        		det = detections[i, :, :]
        		res = det[np.where(det[:, 0] >= 0)[0]]
        		result.append(res)

        # Prepare the output
    	dets = result[0]
    	classes = self.labels
    	width = self.input_width #original input image width
    	height = self.input_height  #original input image height
    	response = []
    	for i in range(dets.shape[0]):
        	   cls_id = int(dets[i, 0])
        	   if cls_id >= 0:
                   score = dets[i, 1]
                   if score > self.threshold:
                       xmin = int(dets[i, 2] * width)
                       ymin = int(dets[i, 3] * height)
                       xmax = int(dets[i, 4] * width)
                       ymax = int(dets[i, 5] * height)
                       class_name = str(cls_id)
                       if classes and len(classes) > cls_id:
                           class_name = classes[cls_id]
                           response.append((class_name, xmin, ymin, xmax, ymax))
    	return response
