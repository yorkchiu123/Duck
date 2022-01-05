import numpy as np
import pandas as pd
from PIL import Image
import math
import csv
import cv2

def mean(num): # 平均函數
    return sum(num) / float(len(num))  

def stdev(num):  # 標準差
    avg = mean(num)
    var = sum([pow(x - avg, 2) for x in num]) / float(len(num) - 1)
    return math.sqrt(var)

def cal(x, mean, stdev):  # 高斯函數
    exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exp

filename = "CSV/train.csv" # 輸入訓練用的csv檔
lines = csv.reader(open(filename, "r"))  # 用讀的方式打開csv檔
dataset = list(lines)  # 每一行存唯一個list
for a in range(1, len(dataset)):  # 去除標頭列
    dataset[a] = [float(x) for x in dataset[a]]
train_data = dataset
train_data = train_data[1:]

part = {}
for b in range(len(train_data)):
    vector = train_data[b]
    if (vector[-1] not in part):
        part[vector[-1]] = []
    part[vector[-1]].append(vector)

model = {} # 準備模型
for c_value, instances in part.items():
    temp = [(mean(attribute), stdev(attribute)) for attribute in
                 zip(*instances[1:-2])]
    del temp[-1]
    model[c_value] = temp


img_path = "test 2.jpg" # input test img path
colimg = Image.open(img_path)
colpixels = colimg.convert("RGB")
colarray = np.array(colpixels.getdata()).reshape(colimg.size + (3,))
indicesArray = np.moveaxis(np.indices(colimg.size), 0, 2)
allArray = np.dstack((indicesArray, colarray)).reshape((-1, 5))
df = pd.DataFrame(allArray, columns=["x", "y", "red", "green", "blue"])
df.drop(['x', 'y'], axis=1, inplace=True)
data = df.to_numpy().astype(float).tolist()
test_data = data # 測試資料

pred = [] # 開始預測
for i in range(len(test_data)):
    probabilities = {}
    for c_value, c_summaries in model.items():
        probabilities[c_value] = 1
        for j in range(len(c_summaries)):
            mean, stdev = c_summaries[j]
            d = test_data[i]
            x = d[j]
            probabilities[c_value] *= cal(x, mean, stdev)
    
    result, best = None, -1
    for c_value, probability in probabilities.items():
        if result is None or probability > best:
            best = probability
            result = c_value

    pred.append(result)


final =[] # 呈現結果圖片
for d in range(len(pred)):
    if(1 == int(pred[d])):
        final.append([0,0,0])
    else:
        final.append(([255,255,255]))
final = np.array(final)
img = cv2.imread(img_path)
size = img.shape
array = np.reshape(final, (size[0], -1))

show_image = Image.fromarray(array) # show img
show_image = show_image.resize((392, 393))
show_image.show()