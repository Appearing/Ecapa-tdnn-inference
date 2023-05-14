import cv2
import os
import random
import shutil  # file copy to move
from tqdm import tqdm  # progress bar
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def video_to_frames(video_path, frames_path, filename):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(frames_path, filename + "_" + f"{count}.jpg"), image)
        success, image = vidcap.read()
        count += 1
        if count > 80:
            break


def video_to_frames_all():
    video_path = 'data_video'
    frames_path = 'data_image\data_frame'
    for filename in tqdm(os.listdir(video_path)):  # get list dir from path
        if filename == '29.mp4':
            in_path = video_path + "\\" + filename
            video_to_frames(in_path, frames_path, filename[:-4])
    print("All the video to frames complete!")


def extract_faces(image_path, face_cascade, face_path, face_resize_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.21, 1)
    for (x, y, w, h) in faces:
        face_image = image[y:y+h, x:x+w]
        # cv2.imwrite(f"face_{x}_{y}.jpg", face_image)
        # cv2.imwrite(face_path, face_image)
        # resized_image = cv2.resize(face_image, (32, 32))
        resized_image = cv2.resize(face_image, (160, 160))

        # cv2.imwrite('resized_image.jpg', resized_image)
        cv2.imwrite(face_resize_path, resized_image)


def extract_faces_all(face_cascade_path):
    frame_path = 'data_image\data_frame'
    face_path = 'data_image\data_face'
    resize_path = 'data_image\data_face_160_160'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    for filename in tqdm(os.listdir(frame_path)):  # get list dir from path
        in_path = frame_path + "\\" + filename
        face_face_path = face_path + "\\" + filename
        face_resize_path = resize_path + "\\" + filename
        extract_faces(in_path, face_cascade, face_face_path, face_resize_path)
    print("Get face from frames all done!")


def data_sort():
    from_path = 'data_image\data_face_32_32'
    to_train_path = 'data_image\data_train'
    to_test_path = 'data_image\data_test'
    for filename in tqdm(os.listdir(from_path)):  # get list dir from path
        in_path = from_path + "\\" + filename
        train_path = to_train_path + "\\" + filename
        test_path = to_test_path + "\\" + filename
        if random.random() > 0.5:
            shutil.copy(in_path, train_path)
        else:
            shutil.copy(in_path, test_path)
    print("The data set sorting task is complete!")


def knn_train(neighbors):
    train_path = 'data_image\data_train'
    feature_train = []
    purpose_train = []
    for filename in tqdm(os.listdir(train_path)):
        in_path = train_path + "\\" + filename
        image = cv2.imread(in_path)
        hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
        feature_train.append(((hist / 255).flatten()))
        purpose_train.append(int(filename.split('_')[0]))
    print("Train data set done!")
    knn = KNeighborsClassifier(n_neighbors=neighbors).fit(feature_train, purpose_train)
    return knn


def knn_test(knn):
    test_path = 'data_image\data_test'
    feature_test = []
    purpose_test = []
    for filename in tqdm(os.listdir(test_path)):
        in_path = test_path + "\\" + filename
        image = cv2.imread(in_path)
        hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
        feature_test.append(((hist / 255).flatten()))
        purpose_test.append(int(filename.split('_')[0]))
    print("Test data set done!")
    predictions = knn.predict(feature_test)
    print('算法评价:')
    print((classification_report(purpose_test, predictions, zero_division=1)))


if __name__ == '__main__':
    # dataset
    video_to_frames_all()
    face_cascade_path = 'D:\software\Python\Anaconda\\anaconda3\envs\pytorch_gpu\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
    extract_faces_all(face_cascade_path)
    # data_sort()
    # modul train and test
    # knn = knn_train(11)
    # knn_test(knn)
    print("!ok!")