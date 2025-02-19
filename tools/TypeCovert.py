# =================================
# File : split
# Author : LGX
# Description : 
# CREATE TIME : 2023/10/19 17:00
# =================================
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob

classes = ["fire", "smoke"]  # 改为你的数据集类别   修改1


def convert(size, box):
    dw = 1.0 / (size[0] + 1)  # 这里加1是为了防止除数等于0
    dh = 1.0 / (size[1] + 1)
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_name):
    in_file = open('../data-fire-smoke/small/anns/' + image_name[:-3] + 'xml', encoding='UTF-8')  # xml文件路径   修改2
    out_file = open('../data-fire-smoke/small/labels/' + image_name[:-3] + 'txt', 'w', encoding='UTF-8')  # 转换后的txt文件存放路径  修改3   注意这里需要你自己新建一个目录label！！！
    f = open('../data-fire-smoke/small/anns/' + image_name[:-3] + 'xml', encoding='UTF-8')  # xml文件路径  修改4同修改1
    xml_text = f.read()
    root = ET.fromstring(xml_text)
    f.close()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
print(wd)

if __name__ == '__main__':

    for image_path in glob.glob("../data-fire-smoke/small/images/*.jpg"):  # 每一张图片都对应一个xml文件这里写xml对应的图片的路径   修改5
        image_name = image_path.split('/')[-1]
        image_name = image_name[7:]
        print(image_name)
        convert_annotation(image_name)