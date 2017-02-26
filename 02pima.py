# coding:utf-8

# ある生体データを持ったが糖尿病であるかどうか？
# ２値分類問題

import numpy
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

# ランダム変数のシード固定（いつも同じランダム変数が出せるように）
seed = 7
numpy.random.seed(seed)

# データセットのロード
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# 入力データ(X)とそれに対応する正解データ(Y)を準備する
X = dataset[:,0:8]
Y = dataset[:,8]

# ★学習データとテストデータに分ける（学習データ67%, テストデータ33%）
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

# モデルを作成
model = Sequential()

# モデルに層を積む
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# コンパイル
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 学習
# ★学習データとテストデータの渡し方は以下のようにする
model.fit(X_train, y_train, validation_data=(X_test,y_test), nb_epoch=150, batch_size=10)

# 参考：keras内で自動で学習データとテストデータに分ける方法もある
# model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)
