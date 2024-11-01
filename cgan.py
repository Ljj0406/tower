from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_cgan(generator, discriminator, latent_dim):
    """构建 CGAN 模型，将生成器和判别器结合。"""
    discriminator.trainable = False  # 冻结判别器权重

    noise_input = Input(shape=(latent_dim,))
    condition_input = Input(shape=(1,))
    generated_data = generator([noise_input, condition_input])

    validity = discriminator([generated_data, condition_input])
    cgan = Model([noise_input, condition_input], validity)

    cgan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    return cgan
