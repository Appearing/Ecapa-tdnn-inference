import multiprocessing
import sys
import cv2
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
from AVrecord import Ui_AVRecord
import sounddevice as sd
from PyCameraList.camera_device import list_video_devices
import pyqtgraph as pg
import numpy as np
import queue                                    # 多线程程序，其中多个线程需要通过在它们之间传递数据来相互通信，先进先出 FIFO 顺序存储和检索项目
from inference import getmodel, readfeature, eval_network  # lhj
from predict import get_image_feature, getlabel

class MainIODevices:
    def __init__(self):
        self.inputDevices = []                  # 音频输入设备列表
        self.outputDevices = []                 # 音频输出设备列表
        self.cameras = []                       # 视频图像设备列表
        self.hostAPIs = []
        self.getIODevices()

        self.inDevice = None                    # 音频输入设备
        self.outDevice = None                   # 音频输出设备
        self.camnum = 0                         # 视频图像设备

    # 返回计算机上所有摄像头设备列表
    def CamDevicesList(self):
        self.cameras = list_video_devices() # 返回一个列表，其中包含所有连接到计算机上的视频设备
        devs = []
        for device in self.cameras:
            devs.append(device[1])  # 提取出其中的第二个元素（即表示设备名称的字符串）
        return devs                 # 函数返回devs，即由设备名称组成的列表，表示连接到计算机上的所有摄像头
    
    # 设置当前使用的摄像头设备，通过指定摄像头设备的名称来选择要使用的摄像头设备
    def setCamDevice(self, devName):
        # get device index in list
        index = None
        for i in range(len(self.cameras)):
            if self.cameras[i][1] == devName:
                index = i
                break

        if index != None:
            self.camnum = index
            print("Setting Camera:", self.camnum, devName)
        else:
            self.camnum = None

    # 返回默认的输入和输出设备，默认的输入设备编号为1，输出设备编号为4
    def getDefaultIODevs(self):
        # (input, output)
        return (1, 4)

    # 获取计算机上可用的音频输入和输出设备列表
    def getIODevices(self):
        devs = sd.query_devices()               # 返回一个包含所有可用音频设备信息的列表，并将其赋值给变量devs
        self.hostAPIs = sd.query_hostapis()     # 返回一个包含主机API（即与音频设备交互的软件）信息的字典列表，并将其赋值给变量self.hostAPIs
        # 遍历每个音频设备
        for device in devs:
            if device['hostapi'] != 0:
                continue
            device['hostapi_text'] = self.hostAPIs[device['hostapi']]['name']  # 获取的当前设备所使用的主机API的名称
            # 判断设备是否支持音频输入或输出
            # "inputDevices"和"outputDevices"分别包含所有可用的音频输入和输出设备信息。
            if device['max_input_channels'] > 0 and device['max_output_channels'] == 0:
                self.inputDevices.append(device)
            else:
                self.outputDevices.append(device)

    # 返回计算机上可用的音频输入设备列表
    def inputDevicesList(self):
        # arrange default items
        defDevicePos, _ = self.getDefaultIODevs() # 函数获取默认的输入和输出设备编号，其中只取输入设备的编号"defDevicePos"而舍弃输出设备编号"_”
        fullDevices = []
        fullDevices[:] = self.inputDevices
        fullDevices[0], fullDevices[defDevicePos] = fullDevices[defDevicePos], fullDevices[0]   # 确保默认输入设备在列表的第一个位置
        devs = []
        # 遍历"fullDevices"中的每个元素（即每个音频输入设备）
        for device in fullDevices:
            devs.append(device['name'])
        return devs

    # 返回计算机上可用的音频输出设备列表
    def outputDevicesList(self):
        # arrange default items
        _, defDevicePos = self.getDefaultIODevs()   # 获取默认输入和输出设备编号，其中只取输出设备的编号"defDevicePos"而舍弃输入设备编号"_”
        defDevicePos = defDevicePos - 3             # 排列所有输出设备时跳过默认设备
        fullDevices = []
        fullDevices[:] = self.outputDevices
        fullDevices[0], fullDevices[defDevicePos] = fullDevices[defDevicePos], fullDevices[0]   # 确保默认输出设备在列表的第一个位置
        devs = []
        # 遍历"fullDevices"中的每个元素（即每个音频输出设备）
        for device in fullDevices:
            devs.append(device['name'])
        return devs

    # 获取设备类型
    def getIODeviceDetail(self, kind):
        return "INPUT"

    # 设置输入设备
    def setInputDevice(self, devName):
        # 获取设备在列表中的索引
        index = None
        for i in range(len(self.inputDevices)):
            if self.inputDevices[i]['name'] == devName:
                index = i
                break

        # 如果设备存在，则设置输入设备
        if index != None:
            self.inDevice = index
            print("Setting IN device:", self.inDevice, devName)
        else:
            self.inDevice = None

    # 设置输出设备
    def setOutputDevice(self, devName):
        # 获取设备在列表中的索引
        index = None
        for i in range(len(self.outputDevices)):
            if self.outputDevices[i]['name'] == devName:
                index = i
                break
        # 如果设备存在，则设置输出设备
        if index != None:
            self.outDevice = len(self.inputDevices) + index
            print("Setting OUT device:", self.outDevice, devName)
        else:
            self.outDevice = None

class MyWindow(QMainWindow, Ui_AVRecord):
    # Initialize the class
    def __init__(self, parent=None, MainIODevices=None):
        super(MyWindow, self).__init__(parent)

        # 调用 Ui_AVRecord 配置 UI 
        self.setupUi(self)
        # 设置主I/O设备
        self.MainIODevices = MainIODevices
        # 初始化UI内容
        self.initUIContent()
        # 设置UI事件
        self.UiEvents()
        # 初始化视频捕获
        self.cap = cv2.VideoCapture()
        # 设置音频参数
        self.Channels = 2
        self.imagemodel, self.imagefeature = get_image_feature()
        self.Samplerate = 16000  # lhj
        self.audioReconSize = 16000  # lhj
        self.Blocksize = 16000  # lhj
        self.modle = getmodel()  # lhj
        self.feature = readfeature('data33.pkl')  # lhj
        self.lins = open('data1.txt').read().splitlines()  # lhj
        self.people = ['teacher', 'SY2241105', 'BY2202120', 'ZY2202418', '19231217',
                       'ZY2202521', 'BY2241110', 'ZY2102122', 'ZY2202417', 'ZF2241102',
                       'SY2202107', 'ZF2241103', '19231210', 'ZY2202205', 'SY2202122',
                       'SY2217314', 'SY2241126', 'SY2202423', '18231008', 'BY2202204',
                       'ZB2202252', 'SY2241204', 'SY2217105', 'ZY2243304', '19374340',
                       'SY2243309', 'SY2202510', 'SY2207619', 'ZY2243310', '19376210',
                       'ZY2202513', 'ZY2202402', 'ZY2243311']
        # 设置FFT图
        # 调用 pyqtgraph 库中实例化一个 PlotWidget 对象，并将其分配给 MyWindow 类的 fftGraph 属性
        self.fftGraph = pg.PlotWidget(self.graphicsView_sound)      # 显示音频输入的实时 FFT 图
        self.fftGraph.setBackground('w')                            # 将图形的背景颜色设置为白色
        self.fftGraph.setGeometry(QtCore.QRect(0, 0, 570, 140))     # 设置图形在窗口中的大小和位置
        self.fftGraph.plotItem.setLogMode(x=True, y=True)           # 设置 x 轴和 y 轴的对数刻度
        self.fftGraph.setYRange(-10, 10)                            # 将 y 轴的范围设置为 -10 到 10
        self.fftPlotPen = pg.mkPen(color=(0,0,255))                 # 绘制 FFT 图的笔的颜色设置为蓝色

        # 设置 x 轴的刻度
        # x 轴的刻度是使用值列表和生成相应标签的循环设置的。 然后使用 fftXAxis 对象的 setTicks 方法设置 x 轴的刻度。
        # 定义刻度值列表
        ticksValues = [20, 60, 100, 170, 310, 600, 1000, 3000, 6000, 12000, 20000]
        # 定义刻度列表
        ticks = []
        # 遍历刻度值列表
        for tick in ticksValues:
            if tick < 1000:                         # 如果刻度值小于1000，则将其转换为字符串
                txt = str(tick)
            else:                                   # 如果刻度值大于等于1000，则将其转换为字符串并添加 'k' 后缀
                txt = str(int(tick/1000)) + 'k'
            ticks.append((np.log10(tick), txt))     # 将刻度值和标签添加到刻度列表中
        self.fftXAxis = self.fftGraph.getPlotItem().getAxis('bottom')       # 获取 x 轴对象
        self.fftXAxis.setTicks([ticks])                                     # 设置 x 轴的刻度

        # 设置几个计时器来实时更新 FFT 图和其他 UI 元素
        self.fftTimer = QTimer()                        # 实例化计时器对象，用于更新 FFT 图
        self.fftTimer.timeout.connect(self.updateFFT)   # 将计时器对象的 timeout 信号连接到 updateFFT 方法

        self.guiTimer = QTimer()                        # 实例化计时器对象，用于更新其他 UI 元素
        self.guiTimer.timeout.connect(self.updateGUI)   # 将计时器对象的 timeout 信号连接到 updateGUI 方法

        self.timer_camera = QTimer()                            # 实例化计时器对象，用于更新摄像头
        self.timer_camera.timeout.connect(self.show_camera)     # 将计时器对象的 timeout 信号连接到 show_camera 方法

    def initUIContent(self):
        # 添加下拉框：选择输入设备
        self.comboBox_inputDevSelecter.addItems(self.MainIODevices.inputDevicesList())  
        self.comboBox_inputDevSelecter_2.addItems(self.MainIODevices.CamDevicesList())

    def UiEvents(self):
        # 添加按钮事件
        self.start.clicked.connect(self.startCallbackEvent)
        self.stop.clicked.connect(self.stopCallbackEvent)
        # 添加下拉框事件
        self.comboBox_inputDevSelecter.activated.connect(self.updateinputDev)
        self.comboBox_inputDevSelecter_2.activated.connect(self.updateCamDev)

    # 在用户点击“开始”按钮开始录音和录像时被调用
    def startCallbackEvent(self):
        # 设置音频回调和音频队列的全局变量
        global audio_callback
        global audio_queue
        global audio_queue_lhj
        # 开始 GUI 定时器
        self.guiTimer.start(500)
        audio_queue = queue.Queue()
        audio_queue_lhj = queue.Queue()
        # 设置输入和摄像设备
        self.MainIODevices.setInputDevice(self.comboBox_inputDevSelecter.currentText())
        self.MainIODevices.setCamDevice(self.comboBox_inputDevSelecter_2.currentText())
        # 开始录音
        self.audio_stream = sd.Stream(channels = self.Channels,
                                      callback = audio_callback,
                                      dtype = 'float32',
                                      samplerate = self.Samplerate,
                                      device = (self.MainIODevices.inDevice, self.MainIODevices.outDevice),
                                      blocksize = self.Blocksize)
        self.audio_stream.start()
        # 刷新 FFT 绘图并设置新的音频绘图
        self.fftGraph.plotItem.clear()
        fftFreqs = np.fft.rfftfreq(self.Blocksize, 1/self.Samplerate)
        self.fftPlotLine = self.fftGraph.plot(fftFreqs,
                                              np.zeros_like(fftFreqs),
                                              pen = self.fftPlotPen)       
        # 开始更新音频 FFT 图
        self.fftTimer.start(50)

        # 打开摄像头
        flag = self.cap.open(self.MainIODevices.camnum, cv2.CAP_DSHOW)
        if not flag:
            print("打开摄像头失败，请检查摄像机与电脑是否正确连接.")
        else:
            # 开始更新摄像头
            self.timer_camera.start(30)

    # 在用户点击“结束”按钮开始录音和录像时被调用
    def stopCallbackEvent(self):
        global audio_queue
        global audio_queue_lhj

        if self.audio_stream != None:
            self.audio_stream.close()       # 关闭音频流
        self.audio_stream = None
        audio_queue = None
        audio_queue_lhj = None

        self.guiTimer.stop()
        self.fftTimer.stop()                # 停止 FFT 定时器
        self.timer_camera.stop()            # 停止摄像机定时器
        self.cap.release()                  # 释放摄像机资源

    def updateGUI(self):
        # 从 queue 中获取数据，直到清空为止
        global audio_queue_lhj    # lhj
        global soud_recognition_result  # lhj
        data = np.array([])  # lhj
        while not audio_queue_lhj.empty():  # lhj
            data = np.append(data, audio_queue_lhj.get())  # lhj
        # 如果数据不够长，则返回
        audiolen = self.audioReconSize
        if np.shape(data)[0] < audiolen:
            return
        # data_2 = data[-audiolen:]  # 取前 fftlen 个数据样本
        soud_recognition = eval_network(self.modle, data, self.feature, self.lins)  # lhj
        if soud_recognition != -1 and soud_recognition != soud_recognition_result:
            soud_recognition_result = soud_recognition
        # 如果音频流是开放的，更新GUI与识别结果
        if self.audio_stream != None and not self.audio_stream.closed:
            self.result_label.setText(self.people[soud_recognition_result])  # lhj

    def updateinputDev(self):
        # 根据组合框中选择的选项更新输入设备
        self.MainIODevices.setInputDevice(self.comboBox_inputDevSelecter.currentText())

    def updateCamDev(self):
        # 根据组合框中选择的选项更新相机设备
        self.MainIODevices.setCamDevice(self.comboBox_inputDevSelecter_2.currentText())

    def updateFFT(self):
        # 从 queue 中获取数据，直到清空为止
        global audio_queue
        data = np.array([])
        while not audio_queue.empty():
            data = np.append(data, audio_queue.get())

        # 如果数据不够长，则返回
        fftlen = self.Blocksize
        if np.shape(data)[0] < fftlen:
            return
        data_2 = data[0:fftlen]                                         # 取前 fftlen 个数据样本

        mag = np.abs(np.fft.rfft(data_2, n=self.Blocksize))             # 计算FFT的幅度
        fftFreqs = np.fft.rfftfreq(self.Blocksize, 1/self.Samplerate)   # 计算FFT的频率
        self.fftPlotLine.setData(fftFreqs, mag)                         # 用新数据更新 FFT 图

    def show_camera(self):
        ret, frame = self.cap.read()                                    # 从相机读取一个帧
        if ret:                                                         # 如果帧读取成功
            frame = cv2.resize(frame, (640, 480))                       # 将帧大小调整为640x480
            frame = cv2.flip(frame, 1)                                  # 水平翻转框架
            frame, faces, locations = face_detect(frame)                # 调用 face_detect 函数检测帧中的人脸
            # 如果检测到人脸
            if faces is not None:
                # 对于检测到的每张人脸
                for i in range(len(faces)):
                    x, y, w, h = locations[i]                           # 获取人脸的位置
                    face = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)   # 将人脸转换为灰度，并将其大小调整为48x48
                    face = cv2.resize(face, (48, 48))
                    label = "face recognition"                          # 设置人脸识别结果的标签
                    frame = cv2.putText(frame, label, (x, y), cv2.FONT_ITALIC, 0.8, (0, 0, 250), 1, cv2.LINE_AA)    # 把标签与人脸视频帧相对应
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)              # 将帧转换为RGB格式
            showImage = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)     # 从视频帧中创建一个 QImage 格式
            self.ImgDisp.setPixmap(QPixmap.fromImage(showImage))        # 将图像显示小部件的像素图设置为QImage


# 创建一个 queue 来存储音频数据
audio_queue = queue.Queue()
audio_queue_lhj = queue.Queue()  # lhj
soud_recognition_result = 0  # lhj


# 定义一个回调函数来处理音频输入
def audio_callback(indata, outdata, frames, time, status):
    # 将输入分割为左通道和右通道
    left = indata[:, 0]
    right = indata[:, 1]
    mono = (left + right)/2             # 合并通道创建单声道信号
    audio_queue.put(mono)               # 将单声道信号添加到队列 queue
    audio_queue_lhj.put(mono)  # lhj
    # 输出输入缓冲区ADC时间(当前被注释掉)
    #print('inputBufferAdcTime: %s ' % time.inputBufferAdcTime)


# 定义一个函数来检测图像中的人脸
def face_detect(image, module=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)              # 将图像转换为灰度
    face_alt2 = cv2.CascadeClassifier("D:\workPlace\PycharmProjects\PR_homework_1\haarcascade_frontalface_default.xml") # 加载人脸检测分类器
    face_locations = face_alt2.detectMultiScale(gray, 1.2, 5)   # 检测图像中的人脸
    num = len(face_locations)
    face = []
    # 如果检测到人脸，提取它们并在它们周围绘制矩形
    if num:
        for face_location in face_locations:
            x, y, w, h = face_location
            face.append(image[y:y + h, x:x + w])
            # 在检测到的面周围画一个矩形
            if module == 1:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            else:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (255))
        # 返回图像、检测到的人脸及其位置
        return image, face, face_locations
    else:
        # 返回原始图像，对于人脸及其位置返回None
        return image, None, None


# 检查该模块是否作为主程序运行
if __name__ == '__main__':
    
    multiprocessing.freeze_support()        # 准备模块以便在冻结的可执行文件中安全使用
    app = QApplication(sys.argv)            # 创建一个新的QApplication实例
    d = MainIODevices()                     # 创建一个新的MainIODevices实例
    myWin = MyWindow(MainIODevices = d)     # 用 MainIODevices 实例作为参数创建一个新的 MyWindow 实例
    myWin.show()                            # 显示 MyWindow 实例
    sys.exit(app.exec_())                   # 启动应用程序事件循环，并在完成后退出程序
    