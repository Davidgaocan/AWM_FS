import numpy as np


class Adagrad:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, gradients):
        if self.cache is None:
            self.cache = np.zeros_like(gradients)

        self.cache += gradients ** 2
        adjusted_gradients = self.learning_rate * gradients / (np.sqrt(self.cache) + self.epsilon)
        return adjusted_gradients

# if __name__ == '__main__':
#     # 使用示例
#     # 初始化Adagrad优化器
#     optimizer = Adagrad(learning_rate=0.01)
#
#     # 模拟一些梯度
#     gradients = np.array([0.1, 0.2, 0.3])
#
#     # 更新参数
#     adjusted_gradients = optimizer.update(gradients)
#     # 模拟一些梯度
#     gradients = np.array([0.1, 0.2, 0.3])
#
#     # 更新参数
#     adjusted_gradients = optimizer.update(gradients)


