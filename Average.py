import pandas as pd
import numpy as np
import os

# ============================================================
# INTERACTIVE FILE NAME INPUT
# ============================================================
FILE_NAME = input("Введите имя файла с данными (например, Data_FFF.xlsx): ").strip()
if not os.path.isfile(FILE_NAME):
    print(f"Файл '{FILE_NAME}' не найден. Проверьте имя и попробуйте снова.")
    exit(1)

# Загрузка
df = pd.read_excel(FILE_NAME, header=None)

# Структура столбцов в файле (0-based индексы):
# 0: пусто
# 1-5: Roach I (a,b,c,d,e)
# 6: пусто
# 7-15: Roach II (1,2,3,4,5,6,7,8,9)
# 16: пусто
# 17-22: Perch I (a,b,c,d,e,f)
# 23: пусто
# 24-31: Perch II (1,2,3,4,5,6,7,8)

# Признаки и их стартовые строки (0-based)
traits = {
    'BL': 3,    # строки 3-22 (20 значений)
    'Ed': 23,   # строки 23-42
    'Pod': 43,  # строки 43-62
    'P1l': 63,  # строки 63-82 (в файле F1l)
    'Bd': 83,   # строки 83-102
    'Cpd': 103  # строки 103-122
}

# Определяем структуру: (колонка в файле, вид, стадия, возраст)
columns_map = [
    # Roach I
    (1, 'Roach', 'I', 'a'), (2, 'Roach', 'I', 'b'), (3, 'Roach', 'I', 'c'),
    (4, 'Roach', 'I', 'd'), (5, 'Roach', 'I', 'e'),
    # Roach II
    (7, 'Roach', 'II', '1'), (8, 'Roach', 'II', '2'), (9, 'Roach', 'II', '3'),
    (10, 'Roach', 'II', '4'), (11, 'Roach', 'II', '5'), (12, 'Roach', 'II', '6'),
    (13, 'Roach', 'II', '7'), (14, 'Roach', 'II', '8'), (15, 'Roach', 'II', '9'),
    # Perch I
    (17, 'Perch', 'I', 'a'), (18, 'Perch', 'I', 'b'), (19, 'Perch', 'I', 'c'),
    (20, 'Perch', 'I', 'd'), (21, 'Perch', 'I', 'e'), (22, 'Perch', 'I', 'f'),
    # Perch II
    (24, 'Perch', 'II', '1'), (25, 'Perch', 'II', '2'), (26, 'Perch', 'II', '3'),
    (27, 'Perch', 'II', '4'), (28, 'Perch', 'II', '5'), (29, 'Perch', 'II', '6'),
    (30, 'Perch', 'II', '7'), (31, 'Perch', 'II', '8'),
]

# Считаем средние
means = {}  # {(вид, стадия, возраст, признак): среднее}

for trait, start_row in traits.items():
    for col_idx, species, stage, age in columns_map:
        values = df.iloc[start_row:start_row+20, col_idx]
        values = pd.to_numeric(values, errors='coerce')
        mean_val = np.nanmean(values)
        means[(species, stage, age, trait)] = mean_val

# Формируем Табл. 1
# Все возрастные метки по порядку
all_ages = ['a', 'b', 'c', 'd', 'e', 'f', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Для каждого вида-стадии определяем, какие возрасты есть
roach_ages = ['a', 'b', 'c', 'd', 'e', '1', '2', '3', '4', '5', '6', '7', '8', '9']
perch_ages = ['a', 'b', 'c', 'd', 'e', 'f', '1', '2', '3', '4', '5', '6', '7', '8']

rows = []
for age in all_ages:
    row = {'Age': age}
    
    # Roach
    if age in roach_ages:
        stage = 'I' if age in 'abcde' else 'II'
        for trait in ['BL', 'Ed', 'Pod', 'P1l', 'Bd', 'Cpd']:
            val = means.get(('Roach', stage, age, trait), np.nan)
            row[f'Roach_{trait}'] = val
    else:
        for trait in ['BL', 'Ed', 'Pod', 'P1l', 'Bd', 'Cpd']:
            row[f'Roach_{trait}'] = np.nan
    
    # Perch
    if age in perch_ages:
        stage = 'I' if age in 'abcdef' else 'II'
        for trait in ['BL', 'Ed', 'Pod', 'P1l', 'Bd', 'Cpd']:
            val = means.get(('Perch', stage, age, trait), np.nan)
            row[f'Perch_{trait}'] = val
    else:
        for trait in ['BL', 'Ed', 'Pod', 'P1l', 'Bd', 'Cpd']:
            row[f'Perch_{trait}'] = np.nan
    
    rows.append(row)

result = pd.DataFrame(rows)

# Округляем
for col in result.columns:
    if col != 'Age':
        result[col] = result[col].round(2)

# Сохраняем
output_name = FILE_NAME.replace('.xlsx', '_Table1_Means.xlsx')
result.to_excel(output_name, index=False, float_format='%.2f')

# Вывод для проверки (с запятыми и прочерками)
display = result.copy()
for col in display.columns:
    if col != 'Age':
        display[col] = display[col].apply(lambda x: '———————' if pd.isna(x) else f"{x:.2f}".replace('.', ','))

print("\nТаблица средних значений успешно создана!")
print(f"Файл сохранён как: {output_name}")
print("\nПревью таблицы (с запятыми):")
print(display.to_string(index=False))