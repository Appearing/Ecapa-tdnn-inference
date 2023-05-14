数据及文件路径准备（见png图）：
	新建data_video文件夹，将原始数据按序号命名放入
	新建一组空文件夹data_image

数据处理：
	主程序如下：
if __name__ == '__main__':
1    # dataset
2    # video_to_frames_all()
3    # face_cascade_path = 'D:\software\……'
4    # extract_faces_all(face_cascade_path)
5    # data_sort()
6    # modul train and test
7    # knn = knn_train(11)
8    # knn_test(knn)
    print("!ok!")
原始数据文件为视频格式在data_video文件夹下
2----将video切片为frame
3、4--提取frame里的人脸部分并降采样为32*32
5----以1：1切分数据集为训练集和测试集
7----训练模型
8----测试模型

如果data_image下只有空文件夹，可全部执行
如果对应目录下存在数据，即可仅执行后面步骤