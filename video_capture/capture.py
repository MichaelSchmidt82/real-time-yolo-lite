import cv2
import numpy as np


FRAME_SKIP = 30


# Draws bounding box around detections
def draw_boxes(frame, data):
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
                org=(bounding_box[0], bounding_box[1] - 10), # offset, above rectangle
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)


local_source = cv2.VideoCapture('rtsp://oak.home.lan:8100/stream0') # lan: "Mighty Oak"
# remote_video = cv2.VideoCapture('rtsp://odd.home.lan:9100/stream0') # lan: "Object Detection Device"

while local_source.isOpened():

    frame_cnt += 1
    if frame_cnt % FRAME_SKIP:
        continue
    else:
        frame_cnt = 0

    _, frame = local_source.read()
    _, jpg = cv2.imencode('.jpg', frame) # for debuggit for now

    #TODO Work on JPEG or stream will occur here

    cv2.waitKey(1) # 1 ms

# release resources
local_source.release()
# remote_video.release()