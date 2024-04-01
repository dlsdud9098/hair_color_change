import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from glob import glob
import dlib
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import time

def predict(image, height=224, width=224):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    """Preprocess the input image before prediction"""
    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)
    
    pred = model.predict(im)
    
    mask = pred.reshape((224, 224))

    return mask

def imShow(image):
  import matplotlib.pyplot as plt
  %matplotlib inline

  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
  
  def Change_hair_color(img, color):

    image = cv2.imread(img)
    mask = predict(image)
    thresh = 0.7  # Threshold used on mask pixels

    """Create 3 copies of the mask, one for each color channel"""
    blue_mask = mask.copy()
    blue_mask[mask > thresh] = color[0]
    blue_mask[mask <= thresh] = 0

    green_mask = mask.copy()
    green_mask[mask > thresh] = color[1]
    green_mask[mask <= thresh] = 0

    red_mask = mask.copy()
    red_mask[mask > thresh] = color[2]
    red_mask[mask <= thresh] = 0

    blue_mask = cv2.resize(blue_mask, (image.shape[1], image.shape[0]))
    green_mask = cv2.resize(green_mask, (image.shape[1], image.shape[0]))
    red_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]))

    """Create an rgb mask to superimpose on the image"""
    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = blue_mask
    mask_n[:, :, 1] = green_mask
    mask_n[:, :, 2] = red_mask

    alpha = 0.85
    beta = (1.0 - alpha)
    out = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

    name = 'test/results/' + img.split('/', 1)[0]
    imShow(out)
    #cv2.imwrite(name, out)

def img_cascade(image):
    face_casecade = cv2.CascadeClassifier('./save_model/haarcascade_frontalface_default.xml')
    faces = face_casecade.detectMultiScale(image, 1.1, 5)

    return image

def faces_detection(faces, image):
    for (left, top, right, bottom) in faces:
        
        # 영역 키우기
        size = 200
        bottom = min(top + bottom + size, image.shape[0])
        right = min(left + right + size, image.shape[1])
        top = max(top - size, 0)
        left = max(left - size, 0)
        
        detection_img = image[top:bottom, left:right]
        
        return detection_img

if __name__ == 'main':
    model = load_model('./save_model/03-25-2024_01-49-24/checkpoint.h5')
    
    img = cv2.imread('./test/images/IU.jpg')
    faces = img_cascade(img)
    detection_img = faces_detection(faces, img)
    
    