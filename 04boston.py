#coding:utf-8

# 犯罪率などのその土地の様々なデータから、その土地の地価を予測する
# 回帰問題（数値予測問題）

import numpy
from keras.models import Sequential
from keras.layers import Dense

import pandas
from sklearn.model_selection import train_test_split

seed = 7
numpy.random.seed(seed)

dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

model = Sequential()

model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))

# モデルをコンパイル
# ★回帰問題の時は lossを mean_squared_error にする
# accuracy Metricsは分類問題専用であるために利用できない。
model.compile(loss='mean_squared_error', optimizer='adam') #, metrics=['mean_squared_logarithmic_error'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=100, batch_size=5)
