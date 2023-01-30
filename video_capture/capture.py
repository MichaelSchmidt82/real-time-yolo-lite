import os

import cv2
import random
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16, 16)


YOLO_INPUT_SHAPE = (640, 640) #* You can not change the shape the model was trained on.
FRAME_SKIP = 2
CUDA = False
PRJ_PATH = os.getcwd()
NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
         'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
         'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
         'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
         'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch','potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
         'keyboard', 'cell phone','microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
         'clock', 'vase', 'scissors', 'teddy bear','hair drier', 'toothbrush')


def display(frame, scale_deltas, ratio, colors, outputs, names=NAMES):

    for (_, x0, y0, x1, y1, cls_id, score) in outputs:
        bound_box = np.array([x0, y0, x1, y1])
        bound_box -= np.array(scale_deltas * 2)
        bound_box /= ratio
        bound_box = bound_box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        score = round(score * 100)
        name = names[cls_id]
        color = colors[name]
        label = f'{name} {score}%'
        cv2.rectangle(img=frame,
                      pt1=bound_box[:2],
                      pt2=bound_box[2:],
                      color=color,
                      thickness=2)

        # We do it backwards. get size/coordinates before *actually* create the text
        (w, h), _ = cv2.getTextSize(text=label,
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.6,
                                    thickness=1)
        # class label
        cv2.rectangle(frame,
                      pt1=(bound_box[0]-1, bound_box[1]),
                      pt2=(bound_box[0]+w, bound_box[1]-h-6),
                      color=color,
                      thickness=-1)

        cv2.putText(img=frame,
            text=label,
            org=(bound_box[0]+5, bound_box[1] - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=.5,
            color=[255, 255, 255],
            thickness=1,
            lineType=cv2.LINE_AA)

        cv2.imshow('VIDEO', frame)


def letterbox(input_img, new_shape=YOLO_INPUT_SHAPE, color=(114, 114, 114), auto=True, scale_up=True, stride=32):

    # Resize and pad image while meeting stride-multiple constraints
    old_shape = input_img.shape[:2] # current shape [height, width]

    # Scale ratio (new / old)
    ratio = min(new_shape[0] / old_shape[0], new_shape[1] / old_shape[1])
    if not scale_up:  # only scale down, do not scale up (for better val mAP!)
        ratio = min(ratio(input_img=input_img, ), 1.0)

    # Compute padding, HxW
    hw_padding = int(round(old_shape[1] * ratio)), int(round(old_shape[0] * ratio))
    delta_w, delta_h = new_shape[1] - hw_padding[0], new_shape[0] - hw_padding[1]

    # minimum rectangle
    if auto:
        delta_h, delta_w = np.mod(delta_h, stride), np.mod(delta_w, stride)

    # divide padding into 2 sides
    delta_w /= 2
    delta_h /= 2

    # resize
    if old_shape[::-1] != hw_padding:
        output_img = cv2.resize(input_img, hw_padding, interpolation=cv2.INTER_LINEAR)
    else:
        output_img = input_img

    top, bottom = int(round(delta_h - 0.1)), int(round(delta_h + 0.1))
    left, right = int(round(delta_w - 0.1)), int(round(delta_w + 0.1))
    output_img = cv2.copyMakeBorder(output_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return output_img, ratio, (delta_w, delta_h)

frame_num = 0
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(NAMES)}

interpreter = tf.lite.Interpreter(model_path='yolov7_model.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# lan: "Mighty Oak"
local_source = cv2.VideoCapture(0)
# lan: Object Detection Device
# remote_video = cv2.VideoCapture('rtsp://odd.home.lan:9100/stream0')

while local_source.isOpened():

    frame_num += 1
    if frame_num % FRAME_SKIP:
        continue

    _, frame = local_source.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # might need to happen sooner

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
    display(frame=frame, scale_deltas=scale_deltas, ratio=ratio, colors=colors, outputs=tfl_out)

    cv2.waitKey(1)

# release resources
local_source.release()
# remote_video.release()
