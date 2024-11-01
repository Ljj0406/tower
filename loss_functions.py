import tensorflow as tf

def binary_crossentropy_loss(y_true, y_pred):
    """二元交叉熵损失函数，用于评估生成结果的合理性。"""
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)
