import tensorflow as tf


class RNNWeight1(tf.keras.layers.Layer):
    """句向量与每个词向量的内积，准确来说是平均绝对误差"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, hn = inputs
        ws = hn * x[:,:-1,:]
        ws = tf.abs(ws)
        ws = tf.reduce_mean(ws, axis=2)
        return ws

class RNNWeight2(tf.keras.layers.Layer):
    """句向量与每个词向量的差"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, hn = inputs
        ws = hn - x[:,:-1,:]
        ws = tf.abs(ws)
        ws = tf.reduce_mean(ws, axis=2)
        return ws

class RNNWeight3(tf.keras.layers.Layer):
    """词向量错位相减"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, hn = inputs
        ws = x[:,1:,:] - x[:,:-1,:]
        ws = tf.abs(ws)
        ws = tf.reduce_mean(ws, axis=2)
        return ws

class RNNWeight4(tf.keras.layers.Layer):
    """句向量与每个词向量的差，取每个时间步向量的最大值"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, hn = inputs
        ws = hn - x[:,:-1,:]
        ws = tf.reduce_max(ws, axis=2)
        return ws

class RNNWeight5(tf.keras.layers.Layer):
    """句向量与每个词向量的差并错位相减"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def call(self, inputs):
        x, hn = inputs
        ws = tf.abs(hn - x[:,1:,:]) - tf.abs(hn - x[:,:-1,:])
        ws = tf.abs(ws)
        ws = tf.reduce_mean(ws, axis=2)
        return ws

class RNNWeight6(tf.keras.layers.Layer):
    """方差"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def call(self, inputs):
        x, hn = inputs
        ws = hn - x[:,:-1,:]
        means = tf.reduce_mean(ws, axis=2, keepdims=True)
        ws = tf.square(ws - means)
        ws = tf.reduce_mean(ws, axis=2)
        return ws
