from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_AVRecord(object):
    def setupUi(self, AVRecord):
        AVRecord.setObjectName("AVRecord")                                              # 设置 AVRecord 的对象名称
        AVRecord.setWindowModality(QtCore.Qt.NonModal)                                  # 设置 AVRecord 的窗口模态为非模态
        AVRecord.resize(615, 650)                                                       # 调整 AVRecord 窗口，X轴宽 Y轴高

        self.start = QtWidgets.QPushButton(AVRecord)                                    # 创建一个按钮：开始
        self.start.setGeometry(QtCore.QRect(430, 10, 71, 23))                           # 设置按钮的几何形状
        self.start.setObjectName("start")                                               # 设置按钮的名称

        self.stop = QtWidgets.QPushButton(AVRecord)                                     # 创建一个按钮：结束
        self.stop.setGeometry(QtCore.QRect(510, 10, 75, 23))                            # 设置按钮的几何形状
        self.stop.setObjectName("stop")                                                 # 设置按钮的对象名称

        self.label_inputdevice = QtWidgets.QLabel(AVRecord)                             # 创建标签：音频输入设备
        self.label_inputdevice.setGeometry(QtCore.QRect(20, 10, 91, 21))                # 设置标签的几何形状
        self.label_inputdevice.setObjectName("label_inputdevice")                       # 设置标签的对象名称

        self.label_inputdevice_2 = QtWidgets.QLabel(AVRecord)                           # 创建标签：视频输入设备
        self.label_inputdevice_2.setGeometry(QtCore.QRect(20, 40, 91, 21))              # 设置标签的几何形状
        self.label_inputdevice_2.setObjectName("label_inputdevice_2")                   # 设置标签的对象名称

        self.graphicsView_sound = QtWidgets.QGraphicsView(AVRecord)                     # 创建图形视图：音频FFT区
        self.graphicsView_sound.setGeometry(QtCore.QRect(20, 70, 571, 140))             # 设置图形视图的几何形状
        self.graphicsView_sound.setObjectName("graphicsView_sound")                     # 设置图形视图的名称

        self.ImgDisp = QtWidgets.QLabel(AVRecord)                                       # 创建标签：相机捕获区/图像识别区
        self.ImgDisp.setGeometry(QtCore.QRect(20, 220, 571, 420))                       # 设置标签的几何形状
        self.ImgDisp.setObjectName("ImgDisp")                                           # 设置标签的对象名称

        self.comboBox_inputDevSelecter = QtWidgets.QComboBox(AVRecord)                  # 创建音频输入设备组合框
        self.comboBox_inputDevSelecter.setGeometry(QtCore.QRect(120, 10, 291, 22))      # 设置组合框的几何形状
        self.comboBox_inputDevSelecter.setObjectName("comboBox_inputDevSelecter")       # 设置组合框的对象名称

        self.comboBox_inputDevSelecter_2 = QtWidgets.QComboBox(AVRecord)                # 创建视频输入设备组合框
        self.comboBox_inputDevSelecter_2.setGeometry(QtCore.QRect(120, 40, 291, 22))    
        self.comboBox_inputDevSelecter_2.setObjectName("comboBox_inputDevSelecter_2")   

        self.label = QtWidgets.QLabel(AVRecord)                                         # 创建标签：语音识别
        self.label.setGeometry(QtCore.QRect(430, 29, 81, 41))                           # 设置标签的几何形状
        self.label.setObjectName("label")                                               # 设置标签的对象名称

        self.result_label = QtWidgets.QLabel(AVRecord)                                  # 创建标签：语音识别结果
        self.result_label.setGeometry(QtCore.QRect(520, 30, 111, 41))                   # 设置标签的几何形状
        self.result_label.setObjectName("result_label")                                 # 设置标签的对象名称

        self.retranslateUi(AVRecord)                                                    # 调用 retranslateUi 函数
        QtCore.QMetaObject.connectSlotsByName(AVRecord)                                 # 按名称连接插槽

    def retranslateUi(self, AVRecord):
        _translate = QtCore.QCoreApplication.translate
        # 设置窗口标题
        AVRecord.setWindowTitle(_translate("AVRecord", "模式识别"))     

        # 覆盖设置 Ui_AVRecord start 名称
        self.start.setText(_translate("AVRecord", "开始"))   
        # 覆盖设置 Ui_AVRecord stop  名称                           
        self.stop.setText(_translate("AVRecord", "结束")) 

        # 覆盖设置 Ui_AVRecord label_inputdevice    名称
        self.label_inputdevice.setText(_translate("AVRecord", "音频输入设备："))   
        # 覆盖设置 Ui_AVRecord label_inputdevice_2  名称       
        self.label_inputdevice_2.setText(_translate("AVRecord", "视频输入设备："))       

        # 覆盖设置 Ui_AVRecord label 名称 
        self.label.setText(_translate("AVRecord", "语音识别结果："))     
        # 覆盖设置 Ui_AVRecord result_label 名称                 
        self.result_label.setText(_translate("AVRecord", "VoiceLabel"))

        # 覆盖设置 Ui_AVRecord ImgDisp 名称 
        self.ImgDisp.setText(_translate("AVRecord", "图像识别区"))                         






# Form implementation generated from reading ui file 'AVrecord.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.




