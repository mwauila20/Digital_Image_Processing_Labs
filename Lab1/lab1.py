import numpy as np
#Задание для лабораторной работы №1
#Сгенерировать матрицы А и И. Вычислить ((A+B)*(-A/2)) без промежуточных массивов в одну команду.
result = (np.random.uniform(-10,10,(3, 3)) + np.random.uniform(-10,10,(3, 3)) * (-np.random.uniform(-10,10,(3, 3)) / 2))
print(result)
