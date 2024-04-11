import torch
from torch.utils.data import TensorDataset, DataLoader




'''
Создадим датасет для тестирования многослойного прецептрона linRegres.py MLPModel
'''
def MLPDataset():
    '''
    X:
    Тензор из 100 равномерно распределенных точек [-1;1]
    unsqueeze(1) - преобразование тензора с (100,) на (100, 1), те в единичную размерность
    
    Y:
    Тензор X возводим в квадрат добавляя случайный шум к кажому y
    
    1.Создаем датасет в интерпритации PyTorch
    2.Создаем DataLoader который облегчает итерацию по датасету. batch_size=5 указывает, 
    что данные будут подаваться в модель пакетами по 5 элементов. shuffle=True гарантирует, 
    что данные будут перемешиваться перед каждой эпохой обучения, что помогает предотвратить переобучение модели, 
    обеспечивая, чтобы порядок данных не влиял на процесс обучения.
    '''
    
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = x.pow(2) + 0.2*torch.rand(x.size())
    
    #Создание DataLoader
    dataset = TensorDataset(x,y)
    train_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
    
    
    
    
print(MLPDataset())