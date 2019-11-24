'''
文本分类模型
LSTM(长短时记忆)在将深度学习应用与文本等非结构化数据类型方面发挥来重要作用。
一个LSTM神经元有一个输入门、一个遗忘门和一个输出门。
'''
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.datasets import imdb

import ssl

# 全局取消证书验证，以解决：[SSL: CERTIFICATE_VERIFY_FAILED]
ssl._create_default_https_context = ssl._create_unverified_context

# 2: 使用来自Keras的IMDB数据集
n_words = 1000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=n_words)

# 3: 呈现一个训练和测试数据示例
# 注意：数据已经预处理（字词映射到矢量）
print("train seq: '{}', test seq: '{}'".format(len(x_train), len(x_test)))
print("train example: \n{}\n".format(x_train[0]))
print("test example: \n{}\n".format(x_test[0]))

# 4: 通过填充序列，为网络准备输入数据
max_len = 200
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# 5: 定义网络架构并编译
model = Sequential()
model.add(Embedding(n_words, 50, input_length=max_len))
model.add(Dropout(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 200, 50)           50000     
_________________________________________________________________
dropout_1 (Dropout)          (None, 200, 50)           0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 100)               60400     
_________________________________________________________________
dense_1 (Dense)              (None, 250)               25250     
_________________________________________________________________
dropout_2 (Dropout)          (None, 250)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 251       
=================================================================
Total params: 135,901
Trainable params: 135,901
Non-trainable params: 0
_________________________________________________________________
'''

# 6: 定义超参数并开始训练网络
batch_size = 64
n_epochs = 10

model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs)

# 7: 最后，可以在测试集上检查训练后的网络性能
print("\naccuracy on test set: '{}'".format(model.evaluate(x_test, y_test)[1]))

# 20191116计算结果: 0.85356
