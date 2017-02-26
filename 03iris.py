#coding:utf-8

# SepalとPetalという2種類の花弁の長さ、幅からアヤメの種類を判別分類する
# 多値分類問題

import numpy
from keras.models import Sequential
from keras.layers import Dense

import pandas
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

seed = 7
numpy.random.seed(seed)

# ★ラベルを含むデータ、スペースで区切られたデータはpandasでロード
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values

X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# ★ラベルを数値に変換
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# ★数値をone hot encodingする
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)

# モデル作る
model = Sequential()

# モデルに層を積む
model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
model.add(Dense(3, init='normal', activation='sigmoid')) # 最終層の形状

# モデルをコンパイル
# ★多値分類問題の時は lossをcategorical_crossentropyにする
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 学習
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=100, batch_size=5)



