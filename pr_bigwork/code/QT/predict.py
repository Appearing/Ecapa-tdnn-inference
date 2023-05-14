from PIL import Image
from facenet import Facenet
import os
import numpy


def get_image_feature():
    image_path = 'D:\workPlace\PycharmProjects\PR_homework_1\data_image\data_face_160_160'
    model = Facenet()
    feature = []
    for filename in os.listdir(image_path):
        image = Image.open(image_path+'\\'+filename)
        out = model.get_feature(image)
        feature.append(out)
    return model, feature


def getlabel(image, model, feature):
    newfeature = model.get_feature(image)
    lenmin = 10
    label = 0
    for i in range(len(feature)):
        lenss = numpy.linalg.norm(newfeature - feature[i], axis=1)
        if lenss < lenmin:
            lenmin = lenss
            label = i
    return label

model, feature = get_image_feature()
image_path = 'D:\workPlace\PycharmProjects\PR_homework_1\data_image\data_face_160_160'
for filename in os.listdir(image_path):
    image = Image.open(image_path + '\\' + filename)
    print(type(image))
    image = numpy.array(image)
    image = Image.fromarray(image.astype(numpy.uint8))
    print(getlabel(image, model, feature))
