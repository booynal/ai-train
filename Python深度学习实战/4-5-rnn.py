'''
双向RNN做文本的情感分类
'''
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
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
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
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
bidirectional_1 (Bidirection (None, 200)               120800    
_________________________________________________________________
dense_1 (Dense)              (None, 250)               50250     
_________________________________________________________________
dropout_2 (Dropout)          (None, 250)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 251       
=================================================================
Total params: 221,301
Trainable params: 221,301
Non-trainable params: 0
_________________________________________________________________
'''

# 6: 使用早停方法来防止过拟合
callbacks = [EarlyStopping(monitor='val_acc', patience=3)]


# 7: 定义超参数并开始训练网络
batch_size = 1024
n_epochs = 100

model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_split=0.2, callbacks=callbacks)

# 8: 最后，可以在测试集上检查训练后的网络性能
print("\naccuracy on test set: '{}'".format(model.evaluate(x_test, y_test, batch_size=batch_size)[1]))

# 20191116计算结果: 0.8311199995231628 (在15代就提前结束)
