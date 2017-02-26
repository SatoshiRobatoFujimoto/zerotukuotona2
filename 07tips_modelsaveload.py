#coding:utf-8

import numpy
from keras.models import Sequential
from keras.layers import Dense

import os
from keras.models import model_from_json

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()

model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, nb_epoch=10, batch_size=10) #, verbose=0)

# セーブ

# ★モデルの構造をJsonファイルにセーブ
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# ★モデルの学習したパラメータをHDF5でセーブ
model.save_weights("model.h5")
print("Saved model to disk")

# ロード

# ★モデルの構造をロード
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# ★モデルの学習したパラメータをロード
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# ロードしたパラメータは学習しなくてもevaluate, predictで利用できる

print("evaluate")
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

print("predict")
Y_predict = loaded_model.predict(X[10:11, :])
print(Y_predict)
