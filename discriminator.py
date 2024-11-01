from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, Concatenate
from tensorflow.keras.models import Model

def build_discriminator(condition_dim):
    """构建判别器，用于判断布置方案合理性。"""
    input_data = Input(shape=(20,))
    condition_input = Input(shape=(condition_dim,))

    # 输入数据与条件拼接
    merged = Concatenate()([input_data, condition_input])

    x = Dense(256)(merged)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation='sigmoid')(x)  # 输出布置合理性标签
    discriminator = Model([input_data, condition_input], output)

    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return discriminator
