import pandas as pd
import numpy as np
import os

# 1. Загрузка исходных данных
# Автоматически вычисляем абсолютный путь к файлу относительно этого скрипта
current_dir = os.path.dirname(os.path.abspath(__file__)) # папка src
project_root = os.path.dirname(current_dir)             # корень проекта
data_path = os.path.join(project_root, 'data', 'diamonds.csv')

# Читаем файл по точному железному пути
df = pd.read_csv(data_path)
print(f"Исходный размер датасета: {df.shape}")

# Перед группировкой отсекаем критические физические аномалии (размеры не могут быть 0)
df = df.query('x > 0 and y > 0 and z > 0')

# 2. Подсчет статистики по количеству драгоценных камней
# Уникальность камня определяем по его ключевым неизменяемым характеристикам
id_stat = df.groupby(by=['carat', 'cut', 'color', 'clarity'])['carat'].count().to_frame()
id_stat.columns = ['count']
id_stat = id_stat.reset_index()

# Объединяем исходный датасет со статистикой counts
df_with_counts = df.merge(id_stat, on=['carat', 'cut', 'color', 'clarity'], how='left')

# 3. Разделение выборки на две части
# Те драгоценные камни, которые встречаются однажды
df_one = df_with_counts.query('count == 1').drop(columns=['count'])

# Те драгоценные камни, которые встречались 2 или 3 раза
df_three = df_with_counts.query('count >= 2').drop(columns=['count'])

# 4. Удаление аномальных значений и выбросов в повторяющихся камнях
# Проверяем, что у повторяющихся записей физические параметры (длина, ширина) не искажены
df_three = df_three.query('x > 0 and y > 0 and z > 0')

# Сортируем по уникальному набору характеристик (вместо 'id') и дате от старых к новым
df_three = df_three.sort_values(by=['carat', 'cut', 'color', 'clarity', 'date'])

# 5. Группировка и выбор последней по времени цены
# Метод 'last' автоматически забирает самую свежую запись для каждого камня
df_three = df_three.groupby(['carat', 'cut', 'color', 'clarity']).agg('last').reset_index()

# 6. Объединение результатов в один чистый датасет
df_cleaned = pd.concat([df_one, df_three]).reset_index(drop=True)

print(f"Размер датасета после очистки повторов и аномалий: {df_cleaned.shape}")
df_cleaned.head()
