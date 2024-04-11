import torch.nn as nn
'''
Создадим пример простейшего алгоритма ML LinnearRegression в интерпритации MLP многослойной прецептронной сети
Такой подход позволит модели зхватывать нелинейные зависимости в данных
'''
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(1,10) #Первый слой 1->10
        self.relu = nn.ReLU() #Нелинейная активационная ф-ия max(0,x) - устраняет отризательные значения
        self.fc2 = nn.Linear(10,1) #Второй слой 10->1

        #Ф-ия определения поедения нейронной сети
        def forward(self,x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        
        
'''

'''