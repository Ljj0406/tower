from sklearn.model_selection import ParameterGrid

def get_hyperparameter_grid():
    """定义超参数搜索空间。"""
    param_grid = {
        'batch_size': [32, 64],
        'epochs': [100, 500],
        'latent_dim': [10, 20],
        'learning_rate': [0.0002, 0.001]
    }
    return ParameterGrid(param_grid)
