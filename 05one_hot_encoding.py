#coding:utf-8

from keras.utils import np_utils

# こんな感じな正解データがあったとして...
Y =[
    [0],
    [1],
    [1],
    [0],
    [2],
    [1],
    [3],
    [0],
    [3],
    [1],
    [3],
    [0],
    ]

# one-hot encoding
Y_onehot = np_utils.to_categorical(Y)

print('one-hot encoding前')
print(Y)
print('one-hot encoding後')
print(Y_onehot)

# ★多値分類問題の正解データはone-hot encodingの形式が好ましい！
