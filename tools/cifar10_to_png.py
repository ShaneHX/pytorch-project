#!/usr/bin/env python3
# coding=UTF-8
'''
@Brief: Extract the png data from `cifar-10-python.tar.gz`
@Author: Shane
@Version: 0.1
@since: 2019-08-23 10:47:58
@lastTime: 2019-08-23 11:07:49
@LastAuthor: Shane
'''
import cv2
import numpy as np
import os
import pickle


data_dir = os.path.join("..", "data", "cifar-10-batches-py")
train_output_dir = os.path.join("..", "data", "cifar-10-png", "raw_train")
test_output_dir = os.path.join("..", "data", "cifar-10-png", "raw_test")

Train = True  

# 解压缩，返回解压后的字典
def unpickle(file):
    print(file)
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


# 生成训练集图片，
if __name__ == '__main__':
    if Train:
        for j in range(1, 6):
            data_path = os.path.join(data_dir, "data_batch_" + str(j))  # data_batch_12345
            train_data = unpickle(data_path)
            print(data_path + " is loading...")

            for i in range(0, 10000):
                img = np.reshape(train_data[b'data'][i], (3, 32, 32))
                img = img.transpose(1, 2, 0)

                label_num = str(train_data[b'labels'][i])
                o_dir = os.path.join(train_output_dir, label_num)
                my_mkdir(o_dir)

                img_name = label_num + '_' + str(i + (j - 1)*10000) + '.png'
                img_path = os.path.join(o_dir, img_name)
                cv2.imwrite(img_path, img)
            print(data_path + " loaded.")

    print("test_batch is loading...")

    # 生成测试集图片
    test_data_path = os.path.join(data_dir, "test_batch")
    test_data = unpickle(test_data_path)
    for i in range(0, 10000):
        img = np.reshape(test_data[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)

        label_num = str(test_data[b'labels'][i])
        o_dir = os.path.join(test_output_dir, label_num)
        my_mkdir(o_dir)

        img_name = label_num + '_' + str(i) + '.png'
        img_path = os.path.join(o_dir, img_name)
        cv2.imwrite(img_path, img)

    print("test_batch loaded.")