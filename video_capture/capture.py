import cv2
import numpy as np


FRAME_SKIP = 30


def draw_boxes(frame, data):
    """initial docstring

    I'm actually unsure if this is needed since yolov7's export.py end2end
    parameter includes "end-to-end ONNX graph which does both bounding box
    prediction and NMS."

    Args:
        frame (_type_): a cv2.VideoCapture frame
        data (dict): data passing key/values pairs

            - type: the classification of the object
            - box: x1, y1 and x2, y2 coordinance (NYI)
    """

    color = (0, 255, 0) # bgr
    for item in data:
        bounding_box = item.get('box')
        label = item.get('type')

    cv2.rectangle(img=frame,
                  pt1=(bounding_box[0], bounding_box[1]),
                  pt2=(bounding_box[2], bounding_box[3]),
                  color=color,
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                  thickness=2,
                  lineType=cv2.LINE_AA)

    cv2.putText(img=frame,
                text=label,
                org=(bounding_box[0], bounding_box[1] - 10), # above rectangle
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)

frame_cnt = 0

# lan: "Mighty Oak"
local_source = cv2.VideoCapture('rtsp://oak.home.lan:8100/stream0')

# lan: Object Detection Device
# remote_video = cv2.VideoCapture('rtsp://odd.home.lan:9100/stream0')
while local_source.isOpened():

    frame_cnt += 1
    if frame_cnt % FRAME_SKIP:
        continue

    _, frame = local_source.read()
    _, jpg = cv2.imencode('.jpg', frame) # for debugging for now
    cv2.imshow('current frame', jpg)
    # Work on a single JPEG for now
    # cv2.imwrite(f'frame{frame_cnt}.jpg', jpg)

    #TODO use Queue object and mutex/lock pattern for stream frames

    cv2.waitKey(1)

# release resources
local_source.release()
# remote_video.release()