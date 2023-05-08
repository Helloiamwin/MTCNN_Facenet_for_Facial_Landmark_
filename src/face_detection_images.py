import cv2
import numpy as np
import torch
import argparse
import utils
from PIL import Image
from facenet_pytorch import MTCNN

# create the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True,
                    help='path to the input data')
args = vars(parser.parse_args())


# computation device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# create the MTCNN model, `keep_all=True` returns all the detected faces 
mtcnn = MTCNN(keep_all=True, device=device)

# read the image 
image = Image.open(args['input']).convert('RGB')
# create an image array copy so that we can use OpenCV functions on it
image_array = np.array(image, dtype=np.float32)
# cv2 image color conversion
image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

# the detection module returns the bounding box coordinates and confidence ...
# ... by default, to get the facial landmarks, we have to provide ...
# ... `landmarks=True`
bounding_boxes, conf, landmarks = mtcnn.detect(image, landmarks=True)
print(f"Bounding boxes shape: {bounding_boxes.shape}")
print(f"Landmarks shape: {landmarks.shape}")

# draw the bounding boxes around the faces
image_array = utils.draw_bbox(bounding_boxes, image_array)
# plot the facial landmarks
image_array = utils.plot_landmarks(landmarks, image_array)

# set the save path
save_path = f"../outputs/{args['input'].split('/')[-1].split('.')[0]}.jpg"
# save image
cv2.imwrite(save_path, image_array)
# shoe the image
cv2.imshow('Image', image_array/255.0)
cv2.waitKey(0)