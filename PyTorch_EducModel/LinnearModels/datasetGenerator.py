import torch

class DatasetGenerator:
    def __init__(self, num_samples=1000, num_outliers=50, outlier_mag=20, distribution='uniform', **kwargs):
        """
        Инициализирует объект DatasetGenerator с возможностью добавления аномальных значений.

        Параметры:
        num_samples (int): Количество образцов данных, которые будут сгенерированы.
        num_outliers (int): Количество аномальных значений.
        outlier_mag (float): Величина, на которую аномальные значения отличаются от нормы.
        distribution (str): Тип распределения входных данных ('uniform', 'normal', 'log_normal', 'exponential', 'gamma').

        kwargs: Дополнительные параметры для различных распределений.
        """
        self.num_samples = num_samples

        # Создание данных в соответствии с выбранным распределением
        if distribution == 'uniform':
            start_point = kwargs.get('start', 0)
            end_point = kwargs.get('end', 2)
            self.x = torch.linspace(start_point, end_point, num_samples).unsqueeze(1)
        elif distribution == 'normal':
            mean = kwargs.get('mean', 0)
            std = kwargs.get('std', 1)
            self.x = torch.normal(mean, std, size=(num_samples,)).unsqueeze(1)
        elif distribution == 'log_normal':
            mean = kwargs.get('mean', 0)
            std = kwargs.get('std', 1)
            self.x = torch.exp(torch.normal(mean, std, size=(num_samples,))).unsqueeze(1)
        elif distribution == 'exponential':
            rate = kwargs.get('rate', 1)
            self.x = torch.distributions.Exponential(rate).sample((num_samples,)).unsqueeze(1)
        elif distribution == 'gamma':
            shape = kwargs.get('shape', 2)
            scale = kwargs.get('scale', 1)
            self.x = torch.distributions.Gamma(shape, scale).sample((num_samples,)).unsqueeze(1)
        else:
            raise ValueError("Unsupported distribution type provided.")

        # Добавление аномалий
        outlier_indices = torch.randint(0, num_samples, (num_outliers,)).long()
        outliers = torch.randn(num_outliers, 1) * outlier_mag + torch.mean(self.x)
        self.x[outlier_indices] = outliers  # Исправленный подход к присвоению аномалий

    def linear_regression(self, a=2.0, b=1.0, noise_variance=1.0):
        noise = torch.randn(self.x.size()) * noise_variance
        y = a * self.x + b + noise
        return self.x, y

    def polynomial_regression(self, coefficients=[1, 0, 2], noise_variance=1.0):
        noise = torch.randn(self.x.size()) * noise_variance
        y = torch.zeros_like(self.x)
        for power, coeff in enumerate(coefficients):
            y += coeff * torch.pow(self.x, power)
        y += noise
        return self.x, y

    def logistic_regression(self, w=1.0, b=1.0, noise_variance=0.5):
        linear_combination = w * self.x + b
        noise = torch.randn(self.x.size()) * noise_variance
        probability = torch.sigmoid(linear_combination + noise)
        y = torch.round(probability)
        return self.x, y

