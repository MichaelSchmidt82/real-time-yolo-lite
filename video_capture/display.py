import random

import cv2
import numpy as np
import face_recognition as fr

import constants as const

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


def label_people(frame, encodings, color):

    people = {}
    face_locations = fr.face_locations(frame)
    # need to segment here??

    # create known faces before hand? ordered dict?
    for name, enc in encodings.items():
        #print(name, enc.shape)
        result = fr.api.compare_faces(known_face_encodings=[encodings[name]], face_encoding_to_check=enc)
        print('name:', name, result[0])
        if result and face_locations:
            print(face_locations)
            people['name'] = face_locations
            frame = create_labels(frame=frame,
                                  bounds=face_locations[0],
                                  label=name,
                                  color=color)
    return frame


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
    # class label
    cv2.rectangle(frame,
                pt1=(bounds[0]-1, bounds[1]),
                pt2=(bounds[0]+w, bounds[1]-h-6),
                color=color,
                thickness=-1)

    cv2.putText(img=frame,
        text=label,
        org=(bounds[0]+5, bounds[1] - 5),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=.5,
        color=[255, 255, 255],
        thickness=1,
        lineType=cv2.LINE_AA)

    return frame

