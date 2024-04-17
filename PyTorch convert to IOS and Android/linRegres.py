import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size=1, hidden_neurons=10, output_size=1):
        """
        Инициализация многослойного перцептрона (MLP) для выполнения линейной регрессии с возможностью обнаружения
        нелинейных зависимостей в данных.
        
        Параметры:
        - input_size (int): Размерность входного вектора. Для линейной регрессии обычно равен 1.
        - hidden_neurons (int): Количество нейронов в скрытом слое. Определяет способность сети к аппроксимации сложных функций.
        - output_size (int): Размерность выходного вектора. Для линейной регрессии равен 1.
        """
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_neurons) # Первый слой принимает входные данные и преобразует их в скрытое пространство.
        self.relu = nn.ReLU() # Нелинейная активационная функция, применяемая после первого слоя.
        self.fc2 = nn.Linear(hidden_neurons, output_size) # Второй слой преобразует данные из скрытого пространства в выходное.

    def forward(self, x):
        """
        Определение прямого прохода через сеть.
        
        Параметры:
        - x (Tensor): Входной тензор размерности (N, input_size), где N - размер батча.
        
        Возвращает:
        - Tensor: Выходной тензор размерности (N, output_size).
        """
        x = self.relu(self.fc1(x)) # Применение первого линейного преобразования и ReLU активации.
        x = self.fc2(x) # Применение второго линейного преобразования для получения выходных значений.
        return x

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        """
        Инициализация простой линейной регрессии.
        """
        super(LinearRegressionModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)  # Один линейный слой

    def forward(self, x):
        """
        Определение прямого прохода через сеть.
        """
        return self.fc(x)
    
    
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        """
        Инициализация модели линейной регрессии.
        """
        super(LinearRegressionModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)  # Единственный линейный слой

    def forward(self, x):
        """
        Определение прямого прохода через сеть.
        """
        return self.fc(x)