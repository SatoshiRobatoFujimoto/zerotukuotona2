#coding:utf-8

import numpy
from keras.models import Sequential
from keras.layers import Dense

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

X = dataset[:,0:8]
Y = dataset[:,8]

model = Sequential()

dense1 = Dense(12, init='uniform', activation='relu', input_dim=8)
dense2 = Dense(8 , init='uniform', activation='relu')
dense3 = Dense(1, init='uniform', activation='sigmoid')

model.add(dense1)
model.add(dense2)
model.add(dense3)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, Y, nb_epoch=80, batch_size=10)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# ★学習の履歴（matplotlibでグラフ化してみてね）
print(history.history.keys())
print(history.history.values())
