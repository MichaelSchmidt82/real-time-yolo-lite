import os
from collections import OrderedDict

import face_recognition as fr

import constants as const


def create_fr_encodings() -> dict:

    encodings = OrderedDict()
    face_files = os.listdir(f'{const.PRJ_PATH}/content/people/')

    for face in face_files:
        key = face.split('.')[0] # name of person
        encodings[key] = fr.load_image_file(f'{const.PRJ_PATH}/content/people/{face}')
        encodings[key] = fr.face_encodings(encodings[key])[0]

    return encodings