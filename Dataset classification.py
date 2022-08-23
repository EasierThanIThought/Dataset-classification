"""Dataset classification

Построение классификатора на наборе данных, полученных Национальным институтом диабета, болезней органов пищеварения и почек (National Institute of Diabetes and Digestive and Kidney Diseases). Цель состоит в том, чтобы ответить на вопрос: есть ли у пациента диабет, основываясь на определенных диагностических измерениях, включенных в набор данных, который получен из исходной базы данных наложением нескольких ограничений. В частности, в рассматриваемых в задании данных, все пациенты — женщины не менее 21 года индийского происхождения Пима.

Набор данных состоит из таких предикторов, как количество беременностей у пациентки, индекс массы тела, уровень инсулина, возраст и так далее. Отклик принимает два значение — больна (1) диабетом или нет (0).
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

"""Считывание датасета, task_data - количество выбранных из датасета строк"""

df = pd.read_csv('/diabetes.csv')
task_data = df.head(640)
df.head()

"""Вывод числа строк в получившейся выборке относящихся к классу 0 (пациент не болен диабетом)."""

len(task_data[task_data['Outcome'] == 0])

"""Разделение данных на тренировочные и тестовые. Первые 80% строк — тренировочные, остальные 20% — тестовые."""

train = task_data.head(int(len(task_data)*0.8))
test = task_data.tail(int(len(task_data)*0.2))

"""Выделение предикторов и отклик. Предикторами служат столбцы Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age. Отклик — Outcome."""

features = list(train.columns[:8])
x = train[features]
y = train['Outcome']

"""Обучение классификатора с использованием DecisionTreeClassifier с параметрами criterion='entropy', max_leaf_nodes = 25, min_samples_leaf = 15 и random_state = 2020."""

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', # критерий разделения
                              min_samples_leaf=15, # минимальное число объектов в листе
                              max_leaf_nodes=25, # максимальное число листьев
                              random_state=2020)
clf=tree.fit(x, y)

"""Подключение библиотеки для визуализации дерева. Сохранение в файл."""

from sklearn.tree import export_graphviz
import graphviz
columns = list(x.columns)
export_graphviz(clf, out_file='tree.dot', 
                feature_names=columns,
                class_names=['0', '1'],
                rounded = True, proportion = False, 
                precision = 2, filled = True, label='all')

with open('tree.dot') as f:
    dot_graph = f.read()

""" Глубина дерева"""

clf.tree_.max_depth

"""Выявление предикторов, по которым выполнено разделение на последнем уровне дерева принятия решений: последние уровни графа, значение разделения - число после <=
Графическое представление:
"""

graphviz.Source(dot_graph)

"""Выполнение предсказания для объектов из тестовой выборки:"""

features = list(test.columns[:8])
x = test[features]
y_true = test['Outcome']
y_pred = clf.predict(x)

"""Доля правильных ответов"""

from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)

"""Среднее значение метрик  F1  (Macro-F1):"""

from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average='macro')

"""Предсказание для конкретного объекта (с индексом 743) исходных данных:"""

df.loc[743, features]

"""Назначенный класс:"""

clf.predict([df.loc[743, features].tolist()])[0]