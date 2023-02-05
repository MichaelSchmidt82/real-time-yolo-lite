import os
from collections import OrderedDict

import face_recognition as fr

import common.constants as const


def create_face_encodings() -> dict:

    peoples_faces = OrderedDict()
    face_files = os.listdir(f'{const.PRJ_PATH}/content/people/')

    for face in face_files:
        key = face.split('.')[0] # name of person
        peoples_faces[key] = fr.load_image_file(f'{const.PRJ_PATH}/content/people/{face}')
        peoples_faces[key] = fr.face_encodings(peoples_faces[key])[0]

    return peoples_faces