# coding:utf-8

# ある生体データを持ったが糖尿病であるかどうか？
# ２値分類問題

import numpy
from keras.models import Sequential
from keras.layers import Dense

# ランダム変数のシード固定（いつも同じランダム変数が出せるように）
seed = 7
numpy.random.seed(seed)

# データセットのロード
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# 入力データ(X)とそれに対応する正解データ(Y)を準備する
X = dataset[:,0:8]
Y = dataset[:,8]

# モデルを作成
model = Sequential()

# モデルの各層を定義
dense1 = Dense(12, init='uniform', activation='relu', input_dim=8)
dense2 = Dense(8 , init='uniform', activation='relu')
dense3 = Dense(1, init='uniform', activation='sigmoid')

# モデルに層を積む
model.add(dense1)
model.add(dense2)
model.add(dense3)

# コンパイル
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 学習
history = model.fit(X, Y, nb_epoch=80, batch_size=10)

# 評価（オプション）
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

