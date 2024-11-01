import numpy as np
from generator import build_generator
from discriminator import build_discriminator
from cgan import build_cgan
from train import train_cgan
from visualize import visualize_generated_data
from hyperparameter_tuning import get_hyperparameter_grid

# 数据准备
X_train = np.random.normal(0, 1, (1000, 20))
y_train = np.random.randint(0, 2, (1000, 1))

# 超参数
params = next(iter(get_hyperparameter_grid()))
latent_dim = params['latent_dim']
batch_size = params['batch_size']
epochs = params['epochs']

# 构建模型
generator = build_generator(latent_dim, 1)
discriminator = build_discriminator(1)
cgan = build_cgan(generator, discriminator, latent_dim)

# 训练模型
train_cgan(generator, discriminator, cgan, X_train, y_train, latent_dim, epochs, batch_size)

# 可视化结果
visualize_generated_data(generator, latent_dim, epochs)
