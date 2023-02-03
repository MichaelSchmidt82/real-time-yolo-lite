import os
from collections import OrderedDict

import face_recognition as fr

import common.constants as const


def create_fr_encodings() -> dict:

    people = OrderedDict()
    face_files = os.listdir(f'{const.PRJ_PATH}/content/people/')

    for face in face_files:
        key = face.split('.')[0] # name of person
        people[key] = fr.load_image_file(f'{const.PRJ_PATH}/content/people/{face}')
        people[key] = fr.face_encodings(people[key])[0]

    return people