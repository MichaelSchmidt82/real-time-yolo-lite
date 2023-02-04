import random

import cv2
import numpy as np
import face_recognition as fr

import common.constants as const

people_colors = {}

def label_objects(frame, scale_deltas, ratio, colors, outputs, names=const.NAMES):

    for (_, x0, y0, x1, y1, cls_id, score) in outputs:
        bounds = np.array([x0, y0, x1, y1])
        bounds -= np.array(scale_deltas * 2)
        bounds /= ratio
        bounds = bounds.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        score = round(score * 100)
        name = names[cls_id]
        color = colors[name]
        label = f'{name} {score}%'

        frame = create_labels(frame=frame, bounds=bounds, label=label, color=color)

    return frame

#TODO This won't scale well: O(faces * people), maybe numpy can help us
def label_people(frame, peoples_faces, colors):

    names = list(peoples_faces.keys())
    if not names:
        return frame

    locations = fr.face_locations(frame)
    if not locations:
        return frame

    #TODO cache this
    all_faces = []
    for _, face in peoples_faces.items():
        all_faces.append(face)

    for loc in locations:
        face = frame[loc[0]:loc[2], loc[3]:loc[1]]
        #TODO use distance to add padding?

        enc_face = fr.face_encodings(face)
        if not enc_face:
            continue

        results = fr.api.compare_faces(known_face_encodings=all_faces,
                                      face_encoding_to_check=enc_face[0])

        for i, res in enumerate(results):
            if res:
                frame = create_labels(frame=frame,
                                      bounds=(loc[3], loc[2], loc[1], loc[0]),
                                      label=names[i],
                                      color=(114, 114, 114))
                                      #TODO Non-static color

    return frame


def letterbox(image,
              new_shape=const.YOLO_INPUT_SHAPE,
              color=(114, 114, 114),
              auto=True,
              scale_up=True,
              stride=32):

    # Resize and pad image while meeting stride-multiple constraints
    old_shape = image.shape[:2] # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    ratio = min(new_shape[0] / old_shape[0], new_shape[1] / old_shape[1])
    if not scale_up:  # only scale down, do not scale up (for better val mAP!)
        ratio = min(ratio, 1.0)

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
        image = cv2.resize(image, hw_padding, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(delta_h - 0.1)), int(round(delta_h + 0.1))
    left, right = int(round(delta_w - 0.1)), int(round(delta_w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return image, ratio, (delta_w, delta_h)



def create_labels(frame, bounds, label, color):

    cv2.rectangle(img=frame,
                pt1=bounds[:2],
                pt2=bounds[2:],
                color=color,
                thickness=2)

    # We do it backwards. get size/coordinates before *actually* create the text
    (w, h), _ = cv2.getTextSize(text=label,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.6,
                                thickness=1)
    # class label, letters and number are arbitrary.  tune to your needs
    cv2.rectangle(frame,
                pt1=(bounds[0] - 1, bounds[1]),
                pt2=(bounds[0] + w, bounds[1] - h - 6),
                color=color,
                thickness= -1)

    cv2.putText(img=frame,
                text=label,
                org=(bounds[0] + 5, bounds[1] - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=.5,
                color=(255, 255, 255), # TODO decide
                thickness=1,
                lineType=cv2.LINE_AA)

    return frame

