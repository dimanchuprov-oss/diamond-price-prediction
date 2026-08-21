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

print("\n--- Расчет коэффициентов инфляции ---")

# Обычно инфляция на алмазном рынке сильнее всего зависит от веса (carat) или цвета/чистоты.
# Создадим простую категорию веса, например, округлив караты до 1 знака, 
# чтобы группы не были слишком мелкими, и добавим качественные параметры.
df_cleaned['carat_group'] = df_cleaned['carat'].round(1)
categories = ['carat_group', 'cut', 'color']

# 2. Считаем среднюю цену за карат для каждой группы в каждом месяце
df_index = df_cleaned.groupby(['date'] + categories)['price_per_carat'].mean().reset_index()

# 3. Находим максимальную (последнюю доступную) дату в датасете
date_max = df_index['date'].max()
print(f"Базовый месяц для расчета инфляции: {date_max}")

# 4. Выделяем цены на эту максимальную дату, чтобы сделать их базой (знаменателем)
df_base_prices = df_index.query('date == @date_max')[categories + ['price_per_carat']].copy()
df_base_prices = df_base_prices.rename(columns={'price_per_carat': 'price_per_carat_max'})

# 5. Объединяем базовые цены с историческим индексом
df_index = df_index.merge(df_base_prices, on=categories, how='outer')

# 6. Считаем коэффициент инфляции (отношение базовой цены к цене в конкретном месяце)
# Если в какой-то группе на максимальную дату данных не было, заполняем коэффициент единицей (1.0)
df_index['inflation'] = df_index['price_per_carat_max'] / df_index['price_per_carat']
df_index['inflation'] = df_index['inflation'].fillna(1.0)

# 7. Объединяем полученные коэффициенты инфляции с нашей основной очищенной таблицей
df_with_inf = df_cleaned.merge(df_index[['date'] + categories + ['inflation']], on=['date'] + categories, how='left')

# 8. Создаем скорректированную целевую переменную
df_with_inf['price_per_carat_adjusted'] = df_with_inf['price_per_carat'] * df_with_inf['inflation']

# Удаляем временную колонку группы каратов, чтобы не засорять датасет
df_final = df_with_inf.drop(columns=['carat_group'])

print(f"Расчет завершен. Итоговый размер таблицы с инфляцией: {df_final.shape}")
print("Пример скорректированных цен:")
print(df_final[['date', 'price_per_carat', 'inflation', 'price_per_carat_adjusted']].head())

# Сохраняем очищенный датасет в новый файл, чтобы ноутбук моделирования мог его прочитать
output_path = os.path.join(project_root, 'data', 'diamonds_cleaned.csv')
df_final.to_csv(output_path, index=False)
print(f"Очищенные данные сохранены в: {output_path}")