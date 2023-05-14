import torch, soundfile
import torch.nn.functional as F
import numpy, pickle
from lhj_model import ECAPA_TDNN


# download model
def getmodel():
    C, m, s = 1024, 0.2, 30
    n_class = 5994

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ECAPA_TDNN(C, n_class, m, s).to(device)

    model_state = model.state_dict()
    loaded_state = torch.load('../../model/pretrain.model', 'cuda')
    for name, param in loaded_state.items():
        origname = name
        if name != 'speaker_loss.weight':
            name = name[16:]
        if name not in model_state:
            name = name.replace("module.", "")
            if name not in model_state:
                print("%s is not in the model." % origname)
                continue
        if model_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s" % (
            origname, model_state[name].size(), loaded_state[origname].size()))
            continue
        model_state[name].copy_(param)
    return model


# model test data
def nntest_model(model, audio):
    model.eval()
    # data prep
    # audio, _ = soundfile.read('D:\workPlace\PycharmProjects\PR_homework_1\wav\wav_segment\\9\\9_34.wav')
    length = 300 * 160 + 240
    if audio.shape[0] <= length:  # padding
        shortage = length - audio.shape[0]
        audio = numpy.pad(audio, (0, shortage), 'wrap')
    data_1 = torch.FloatTensor(numpy.stack([audio], axis=0))
    # test
    with torch.no_grad():
        embedding_1 = model.forward(data_1, aug=False)
        embedding_1 = F.normalize(embedding_1, p=2, dim=1)
    # result
    return int(torch.argmax(embedding_1))


def get_feature_emneeding(model, eval_list, datapath='data.pkl'):
    model.eval()
    files = []
    embeddings = {}
    lines = open(eval_list).read().splitlines()
    for line in lines:
        files.append(line)
    setfiles = list(set(files))
    setfiles.sort()
    for idx, file in enumerate(setfiles):
        audio, _ = soundfile.read(file)
        # Full utterance
        data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
        # Spliited utterance matrix
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])
        feats = numpy.stack(feats, axis=0).astype(numpy.float32)
        data_2 = torch.FloatTensor(feats).cuda()
        # Speaker embeddings
        with torch.no_grad():
            embedding_1 = model.forward(data_1, aug=False)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            embedding_2 = model.forward(data_2, aug=False)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)
        embeddings[file] = [embedding_1, embedding_2]
    # print(type(embeddings))
    f = open(datapath, 'wb')  # 打开一个二进制文件用于写入
    pickle.dump(embeddings, f)  # 将字典对象序列化并写入文件
    f.close()  # 关闭文件
    f = open(datapath, 'rb')  # 打开一个二进制文件用于读取
    d = pickle.load(f)  # 从文件中读取对象并反序列化
    f.close()  # 关闭文件
    # print(type(d))


def readfeature(featurepath='data33.pkl'):
    f = open(featurepath, 'rb')  # 打开一个二进制文件用于读取
    embeddings = pickle.load(f)  # 从文件中读取对象并反序列化
    f.close()  # 关闭文件
    return embeddings


def eval_network(model, audio, embeddings, lines):
    model.eval()
    # Full utterance
    data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

    # Spliited utterance matrix
    max_audio = 300 * 160 + 240
    if audio.shape[0] <= max_audio:
        shortage = max_audio - audio.shape[0]
        audio = numpy.pad(audio, (0, shortage), 'wrap')
    feats = []
    startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
    for asf in startframe:
        feats.append(audio[int(asf):int(asf) + max_audio])
    feats = numpy.stack(feats, axis=0).astype(numpy.float32)
    data_2 = torch.FloatTensor(feats).cuda()
    # Speaker embeddings
    with torch.no_grad():
        embedding_1 = model.forward(data_1, aug=False)
        embedding_1 = F.normalize(embedding_1, p=2, dim=1)
        embedding_2 = model.forward(data_2, aug=False)
        embedding_2 = F.normalize(embedding_2, p=2, dim=1)
    embedding_11, embedding_12 = embedding_1, embedding_2
    scores = []
    # lines = open('data2.txt').read().splitlines()
    for line in lines:
        embedding_21, embedding_22 = embeddings[line]
        # Compute the scores
        score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
        score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
        score = (score_1 + score_2) / 2
        score = score.detach().cpu().numpy()
        scores.append(score)
    scoresmax = 0
    for i in range(len(scores)):
        if scores[i] > scores[scoresmax]:
            scoresmax = i
    if scores[scoresmax] > 0.30:
        return scoresmax
    else:
        return -1


# import os
if __name__ == '__main__':
    model = getmodel()
    get_feature_emneeding(model, 'data1.txt', 'data33.pkl')
    # audio, _ = soundfile.read('D:\workPlace\PycharmProjects\PR_homework_1\\wav\wav_full\\11.wav')
    # eval_network(model, audio)


    # frame_path = 'D:\workPlace\PycharmProjects\PR_homework_1\wav\wav_segment\\11'
    # for filename in os.listdir(frame_path):
    #     audio, _ = soundfile.read(frame_path + '\\' + filename)
    # # audio, _ = soundfile.read('result1.wav')
    # # audio, _ = soundfile.read('D:\workPlace\PycharmProjects\PR_homework_1\pr_bigwork\\19.wav')
    #     n_class = nntest_model(model, audio)
    #     print(n_class)
