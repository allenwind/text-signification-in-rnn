import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

import rnn_weights
from dataset import SimpleTokenizer, find_best_maxlen
from dataset import load_THUCNews_title_label
from dataset import load_weibo_senti_100k
from dataset import load_simplifyweibo_4_moods
from dataset import load_hotel_comment

# 来自Transformer的激活函数，效果略有提升
def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

X, y, classes = load_THUCNews_title_label()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=7384672)

num_classes = len(classes)
tokenizer = SimpleTokenizer()
tokenizer.fit(X_train)
X_train = tokenizer.transform(X_train)

maxlen = find_best_maxlen(X_train)
maxlen = 128

X_train = sequence.pad_sequences(
    X_train, 
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0
)
y_train = tf.keras.utils.to_categorical(y_train)

num_words = len(tokenizer)
embedding_dims = 128
hdims = 128

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
x = Embedding(num_words, embedding_dims,
    embeddings_initializer="glorot_normal",
    mask_zero=True,
    input_length=maxlen)(inputs)

# whole_seq_output, final_memory_state, final_carry_state
x, xc, _ = LSTM(hdims, return_sequences=True, return_state=True)(x)

# 词重要性权重计算
w1 = rnn_weights.RNNWeight1()([x,xc])
w2 = rnn_weights.RNNWeight2()([x,xc])
w3 = rnn_weights.RNNWeight3()([x,xc])
w4 = rnn_weights.RNNWeight4()([x,xc])
w5 = rnn_weights.RNNWeight5()([x,xc])
w6 = rnn_weights.RNNWeight6()([x,xc])

x = Dense(hdims, activation=gelu)(xc)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.summary()

# 词重要性权重输出
mw_1 = Model(inputs, w1)
mw_2 = Model(inputs, w2)
mw_3 = Model(inputs, w3)
mw_4 = Model(inputs, w4)
mw_5 = Model(inputs, w5)
mw_6 = Model(inputs, w6)

mws = [mw_1, mw_2, mw_3, mw_4, mw_5, mw_6]

batch_size = 32
epochs = 2
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2
)

id_to_classes = {j:i for i,j in classes.items()}
from color import print_color_text
def visualization():
    for sample, label in zip(X_test, y_test):
        sample_len = len(sample)
        if sample_len > maxlen:
            sample_len = maxlen

        x = np.array(tokenizer.transform([sample]))
        x = sequence.pad_sequences(
            x, 
            maxlen=maxlen,
            dtype="int32",
            padding="post",
            truncating="post",
            value=0
        )

        y_pred = model.predict(x)[0]
        y_pred_id = np.argmax(y_pred)
        # 预测错误的样本跳过
        if y_pred_id != label:
            continue
        
        # 不同权重的预测效果
        for i, model_weight_outputs in enumerate(mws, start=1):
            weights = model_weight_outputs.predict(x)[0]
            weights = weights.flatten()[:sample_len]
            print("weight {} ".format(i), sep="\t", end="")
            print_color_text(sample, weights)
            print(" =>", id_to_classes[y_pred_id])

        input() # 按回车预测下一个样本

visualization()

