import random

import cv2
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16, 16)

import constants as const
from display import label_objects, label_people, letterbox
from face_recog import create_fr_encodings


frame_num = 0
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(const.NAMES)}

faces_encs = create_fr_encodings()

interpreter = tf.lite.Interpreter(model_path='yolov7_model.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# camera "Mighty Oak"
local_source = cv2.VideoCapture(0)

# lan: Object Detection Device
# remote_video = cv2.VideoCapture('rtsp://odd.home.lan:9100/stream0')

while local_source.isOpened():

    frame_num += 1
    if frame_num % const.FRAME_SKIP:
        continue

    _, frame = local_source.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # might need to happen

    # object detection
    img, ratio, scale_deltas = letterbox(frame, auto=False)
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32)
    img /= 255

    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    tfl_out = interpreter.get_tensor(output_details[0]['index'])

    frame = label_objects(frame=frame,
                          scale_deltas=scale_deltas,
                          ratio=ratio,
                          colors=colors,
                          outputs=tfl_out)

    frame = label_people(frame=frame, encodings=faces_encs, colors=colors)
    cv2.imshow('deep learning magic', frame)

    cv2.waitKey(1)

# release resources
local_source.release()
# remote_video.release()
