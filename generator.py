from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, Concatenate
from tensorflow.keras.models import Model

def build_generator(latent_dim, condition_dim):
    """构建生成器，用于生成塔吊布置特征。"""
    noise_input = Input(shape=(latent_dim,))
    condition_input = Input(shape=(condition_dim,))

    # 噪声与条件拼接
    merged = Concatenate()([noise_input, condition_input])

    x = Dense(128)(merged)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)

    output = Dense(20, activation='tanh')(x)  # 输出20维特征
    generator = Model([noise_input, condition_input], output)

    return generator
