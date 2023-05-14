import numpy as np
from moviepy.editor import *
from tqdm import tqdm
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import cv2
from pydub import AudioSegment
import os


def get_voice_from_video():
    video_path = 'data_video'
    voice_path = 'data_voice'
    for filename in os.listdir(video_path):  # get list dir from path
        in_path = video_path + "\\" + filename
        out_path = voice_path + "\\" + filename[:-4] + ".mp3"
        video = VideoFileClip(in_path)
        audio = video.audio
        audio.write_audiofile(out_path)


def voice_segment(split_length=1):
    voice_path = 'data_voice'
    out_path = 'voice_segment'
    for filename in tqdm(os.listdir(voice_path)):  # get list dir from path
        in_path = voice_path + "\\" + filename
        audio = AudioSegment.from_file(in_path, format="mp3")
        for i, chunk in enumerate(audio[::split_length * 1000]):
            chunk.export(os.path.join(out_path, filename[:-4] + "_" + f"{i}.mp3"), format="mp3")
    print("!voice segment done!")


def video_data_set():
    train_path = r"data_image\data_train"
    feature_train = []
    purpose_train = []
    for filename in tqdm(os.listdir(train_path)):
        in_path = train_path + "\\" + filename
        image = cv2.imread(in_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature_train.append(np.reshape(gray, (1, -1)))
        purpose_train.append(int(filename.split('_')[0]))
    print("!Train data set done!")
    test_path = r"data_image\data_test"
    feature_test = []
    purpose_test = []
    for filename in tqdm(os.listdir(test_path)):
        in_path = test_path + "\\" + filename
        image = cv2.imread(in_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature_test.append(np.reshape(gray, (1, -1)))
        purpose_test.append(int(filename.split('_')[0]))
    print("!Test data set done!")
    return feature_train, purpose_train, feature_test, purpose_test


def data_kernel_pca(data):
    data = np.concatenate(data, axis=0)
    kpca = KernelPCA(n_components=2, kernel='rbf')
    X_kpca = kpca.fit_transform(data)
    print("!KPCA done!")
    return X_kpca


def data_manifold_learning(data):
    data = np.concatenate(data, axis=0)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    digits_tsne = tsne.fit_transform(data)
    print("!Manifold Learning done!")
    return digits_tsne


def video_knn_train_test(feature_train, purpose_train, feature_test, purpose_test, neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=neighbors).fit(feature_train, purpose_train)
    print("!Train KNN done!")
    predictions = knn.predict(feature_test)
    print('算法评价:')
    print((classification_report(purpose_test, predictions, zero_division=1)))


# from audioconverter import AudioConverter
# def to_wav():
#     ac = AudioConverter()
#     video_path = 'data_video'
#     voice_path = 'wav'
#     for filename in os.listdir(video_path):  # get list dir from path
#         in_path = video_path + "\\" + filename
#         out_path = voice_path + "\\" + filename[:-4] + ".wav"
#         # video = VideoFileClip(in_path)
#         # audio = video.audio
#         # audio.write_audiofile(out_path)
#         ac.convert(input_file=in_path, output_file=out_path, output_format=".wav", codec="pcm_mulaw")


if __name__ == '__main__':
    # get_voice_from_video()
    # voice_segment()
    # feature_train, purpose_train, feature_test, purpose_test = video_data_set()
    # video_knn_train_test(data_kernel_pca(feature_train), purpose_train, data_kernel_pca(feature_test), purpose_test)
    # video_knn_train_test(data_manifold_learning(feature_train), purpose_train, data_manifold_learning(feature_test), purpose_test)
    # to_wav()
    print("!OK!")

