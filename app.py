'''
Author: https://github.com/AtriSaxena

'''

import flask
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import io
from PIL import Image
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
labels = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
	# load the pre-trained Pytorch model (here we are using a model
	# pre-trained on VOC Dataset using SSD , but you can
	# substitute in your own networks just as easily)
	global model
	model = build_ssd('test',300,21)
	model.load_weights('weights/ssd300_mAP_77.43_v2.pth')

def prepare_image(image):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
	x = cv2.resize(image, (300, 300)).astype(np.float32)
	x -= (104.0, 117.0, 123.0)
	x = x.astype(np.float32)
	x = x[:, :, ::-1].copy()
	x = torch.from_numpy(x).permute(2, 0, 1)

	# return the processed image
	return x

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			img = flask.request.files["image"]
			img = Image.open(img)
			# preprocess the image and prepare it for classification
			image = prepare_image(img)

			xx = Variable(image.unsqueeze(0))     # wrap tensor in Variable
			if torch.cuda.is_available():
				xx = xx.cuda()
			y = model(xx)
			detections = y.data

			scale = torch.Tensor(img.size).repeat(2)

			# loop over the results and add them to the list of
			# returned predictions
			data['predictions'] = []
			for i in range(detections.size(1)):
				j = 0
				while detections[0,i,j,0] >= 0.6: # set the probabilty filter
					score = detections[0,i,j,0]
					label_name = labels[i-1]
					pt = (detections[0,i,j,1:]*scale).cpu().numpy()
					coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
					r = {"label": label_name, "probability": float(score),"coords":str(coords)}
					data["predictions"].append(r)
					j+=1

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run(host='0.0.0.0')