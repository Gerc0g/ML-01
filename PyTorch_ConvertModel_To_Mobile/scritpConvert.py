import torch
import coremltools as ct
from torchinfo import summary
import torch.nn as nn
from linRegres import LinearRegressionModel

'''Класс конвертации модели из pytorch в mlmodel coremltools и тест с анализом модели после окнвертации'''
'''
class convertModelCoreML():
    def __init__(self) -> None:
        pass
    
    #Конвертация под трейс или скрипт пайторч    
    #Вывод саммери для пайторч
    #Конвертация модели в mlmmodel
    #Вывод саммери для mlmodel
    #Построение графиков по входным данным mlmodel отображающих производительность
    #тесты на разных входных данных со сводками
    '''
    
 
'''Класс который описывает и тестит функцию которая импортируется из savemodel pytorch'''    
'''
class analyseInputPTModel():
    def __init__(self) -> None:
        pass
    
    #Импорт модели из сохраненного файла
    def importModel():
        pass
    
    #Вывод саммери
    def summaryModel():
        pass
    
    #Тесты на разных входных данных со сводками
    def testModelonData():
        pass
    
    #Построение графиков отображающих производительность модели до импорта
''' 


'''
nameImportModel(str): 
inputData(): 
'''
inputData = torch.randn(10, 1)
nameImportModel = 'full_model.pth'
   
#Импорт полной модели PyTorch(Модель+веса)
model = torch.load(nameImportModel)

#Вывод summary по импортированной модели, при не соответствии входных данных выводится инфо без инпутСайза
summary(model)


#Трейс модели
traced_model = torch.jit.trace(model, inputData)


# Конвертируем TorchScript модель в Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=inputData.shape)],  # Указываем тип и форму входных данных
    convert_to='neuralnetwork'
)

'''Можно запустить модель для проверки перед сохранением'''

# Вывод сводки модели
mlmodel.get_spec()

# Сохраняем модель в файл
mlmodel.save("MyModel.mlmodel")




    
    
    
