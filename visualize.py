import numpy as np
import matplotlib.pyplot as plt

def visualize_generated_data(generator, latent_dim, epoch):
    """可视化生成的塔吊布置方案。"""
    noise = np.random.normal(0, 1, (5, latent_dim))
    sample_labels = np.array([[0], [1], [0], [1], [0]])
    generated_samples = generator.predict([noise, sample_labels])

    print(f"\nEpoch {epoch}: 生成的塔吊布置方案示例:")
    print(generated_samples)