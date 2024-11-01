import numpy as np

def train_cgan(generator, discriminator, cgan, X_train, y_train, latent_dim, epochs, batch_size):
    """训练 CGAN 模型，并打印训练进度。"""
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_data, real_labels = X_train[idx], y_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_data = generator.predict([noise, real_labels])

        d_loss_real = discriminator.train_on_batch([real_data, real_labels], real)
        d_loss_fake = discriminator.train_on_batch([generated_data, real_labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = cgan.train_on_batch([noise, real_labels], real)

        if epoch % 10 == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
