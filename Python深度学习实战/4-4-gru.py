'''
文本分类模型
在RNN中经常使用的另一种类型的神经元类型是GRU，这些神经元实际上比LSTM神经元简单，因为它们只有两个门：
更新和复位。更新门决定内存，复位门将内存与当前输入相结合。
'''
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import GRU
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
model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
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
gru_1 (GRU)                  (None, 100)               45300     
_________________________________________________________________
dense_1 (Dense)              (None, 250)               25250     
_________________________________________________________________
dropout_2 (Dropout)          (None, 250)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 251       
=================================================================
Total params: 120,801
Trainable params: 120,801
Non-trainable params: 0
'''

# 6: 使用早停方法来防止过拟合
callbacks = [EarlyStopping(monitor='val_acc', patience=3)]


# 7: 定义超参数并开始训练网络
batch_size = 512
n_epochs = 100

model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_split=0.2, callbacks=callbacks)

# 8: 最后，可以在测试集上检查训练后的网络性能
print("\naccuracy on test set: '{}'".format(model.evaluate(x_test, y_test)[1]))

# 20191116计算结果: 0.84608 (在23代就提前结束)
