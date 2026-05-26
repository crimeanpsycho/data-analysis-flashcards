# Памятка: Решение заданий Части B (Airbnb NYC)

---

## 0. Базовая загрузка

```python
import pandas as pd

df = pd.read_csv('train_B.csv')
df['last booking'] = pd.to_datetime(df['last booking'])
```

> ⚠️ Всегда конвертируй `last booking` в datetime в самом начале!

---

## B1 — Фильтрация и агрегация

### Шаблоны задач

#### Подсчёт строк по одному или нескольким условиям
```python
# ИЛИ
len(df[(df['borough'] == 'Brooklyn') | (df['borough'] == 'Queens')])
# Более компактно через isin:
len(df[df['borough'].isin(['Brooklyn', 'Queens'])])

# ОТ И ДО (включительно)
len(df[(df['minimum nights'] >= 5) & (df['minimum nights'] <= 10)])

# По году/месяцу
len(df[df['last booking'].dt.year == 2022])
len(df[(df['last booking'].dt.year == 2018) & (df['last booking'].dt.month == 10)])

# Утренние часы (4:00 – 11:59)
len(df[(df['last booking'].dt.hour >= 4) & (df['last booking'].dt.hour <= 11)])

# Цена меньше Q1
Q1 = df['price'].quantile(0.25)
len(df[df['price'] < Q1])
```

#### Найти строку по min/max, вернуть другой столбец
```python
# Самое дешёвое жильё в 2022 → name
sub = df[df['last booking'].dt.year == 2022]
sub.loc[sub['price'].idxmin(), 'name']

# Объявление с максимальным minimum_nights среди домов 2003 года → house rules
sub = df[df['construction year'] == 2003]
sub.loc[sub['minimum nights'].idxmax(), 'house rules']

# Самое дорогое Shared room, 2009 год → borough
sub = df[(df['room type'] == 'Shared room') & (df['construction year'] == 2009)]
sub.loc[sub['price'].idxmax(), 'borough']

# Самое раннее бронирование среди minimum_nights >= 30 → room type
sub = df[df['minimum nights'] >= 30]
sub.loc[sub['last booking'].idxmin(), 'room type']

# Самое дешёвое на Staten Island → id
sub = df[df['borough'] == 'Staten Island']
sub.loc[sub['price'].idxmin(), 'id']
```

#### Агрегации по условию
```python
# Боро с наибольшим числом объявлений (вообще)
df['borough'].value_counts().idxmax()

# Боро где больше всего объявлений с price <= медианы
med = df['price'].median()
df[df['price'] <= med]['borough'].value_counts().idxmax()

# Дисперсия (ГЕНЕРАЛЬНАЯ, ddof=0!) для Private room ИЛИ топ-боро
top_boro = df['borough'].value_counts().idxmax()
filt = df[(df['room type'] == 'Private room') | (df['borough'] == top_boro)]
round(filt['price'].var(ddof=0), 2)

# Размах (max - min) по фильтру
sub = df[df['construction year'] == 2009]
sub['price'].max() - sub['price'].min()

# IQR по фильтру
sub = df[df['borough'] == 'Staten Island']
Q1 = sub['price'].quantile(0.25)
Q3 = sub['price'].quantile(0.75)
round(Q3 - Q1, 2)

# Количество уникальных host name в Brooklyn
df[df['borough'] == 'Brooklyn']['host name'].nunique()
```

---

## B2 — Создание новых признаков

> ⚠️ Все признаки создаются последовательно в ОДНОМ скрипте — следующий шаг зависит от предыдущего!

### Шаблоны задач

#### Длина строки (len name)
```python
df['len name'] = df['name'].str.len()
round(df['len name'].mean(), 2)
```

#### Индикатор по подстроке (no smoking rule)
```python
df['no smoking rule'] = df['house rules'].str.lower().str.contains('no smoking', na=False).astype(int)

# % объявлений БЕЗ запрета курения (значение 0)
round((df['no smoking rule'] == 0).sum() / len(df) * 100, 2)
```

#### Количество хозяев (num hosts)
```python
# host name разделены через ", " (запятая + пробел)
df['num hosts'] = df['host name'].str.split(', ').str.len()
# Если есть NaN — они дадут NaN; заполни если нужно:
# df['num hosts'] = df['host name'].fillna('').str.split(', ').str.len()

round(df['num hosts'].std(), 2)
```

#### Сезон по месяцу (season)
```python
import numpy as np

m = df['last booking'].dt.month
conditions = [
    m.isin([12, 1, 2]),   # зима → 1
    m.isin([3, 4, 5]),    # весна → 2
    m.isin([6, 7, 8]),    # лето → 3
    m.isin([9, 10, 11])   # осень → 4
]
choices = [1, 2, 3, 4]
df['season'] = np.select(conditions, choices)

round(df['season'].mean(), 2)
```

#### Номер недели в году (week)
```python
df['week'] = df['last booking'].dt.isocalendar().week.astype(int)

# Найти НЕДЕЛЮ с наибольшим числом бронирований
top_week = df['week'].value_counts().idxmax()

# Вернуть МЕСЯЦ этой недели
df[df['week'] == top_week]['last booking'].dt.month.mode()[0]
```

---

## B3 — Объединение таблиц

> ⚠️ В B3 всегда делается merge через id / id rent. Ключ в train_B — `id`, в доп. файле — `id rent`.

### Базовый шаблон merge
```python
df_main = pd.read_csv('train_B.csv')
df_main['last booking'] = pd.to_datetime(df_main['last booking'])

df_extra = pd.read_csv('trainX_B3.csv')

# LEFT JOIN — берём все строки из df_extra, подтягиваем данные из df_main
merged = df_extra.merge(
    df_main[['id', 'нужные_столбцы']],
    left_on='id rent',
    right_on='id',
    how='left'
)

# INNER JOIN — только совпадения в обеих таблицах
merged = df_main.merge(
    df_extra,
    left_on='id',
    right_on='id rent',
    how='inner'
)
```

---

### B3 Q1 — Сервисный сбор (train1_B3: id rent, area, service fee)
```python
df1 = pd.read_csv('train1_B3.csv')
merged = df_main.merge(df1, left_on='id', right_on='id rent', how='inner')

merged['fee_ratio'] = (merged['service fee'] / merged['price']) * 100

# Q3 для Манхэттена
manhattan = merged[merged['borough'] == 'Manhattan']
round(manhattan['fee_ratio'].quantile(0.75), 2)

# Район с наибольшей средней ценой среди районов с < 100 предложениями
counts = merged.groupby('area')['id'].count()
small_areas = counts[counts < 100].index
sub = merged[merged['area'].isin(small_areas)]
sub.groupby('area')['price'].mean().idxmax()
```

---

### B3 Q2 — Отзывы (train2_B3: id rent, number of reviews)
```python
df2 = pd.read_csv('train2_B3.csv')
merged = df2.merge(df_main[['id', 'last booking']], left_on='id rent', right_on='id', how='left')

# Среднее число отзывов для бронирований 2016 года
sub_2016 = merged[merged['last booking'].dt.year == 2016]
round(sub_2016['number of reviews'].mean(), 2)

# IQR по месяцам → месяц с наименьшим IQR
merged['month'] = merged['last booking'].dt.month
iqr_by_month = merged.groupby('month')['number of reviews'].apply(
    lambda x: x.quantile(0.75) - x.quantile(0.25)
)
iqr_by_month.idxmin()
```

---

### B3 Q3 — Доступность (train3_B3: id rent, area, availability)
```python
df3 = pd.read_csv('train3_B3.csv')
merged = df_main.merge(df3, left_on='id', right_on='id rent', how='inner')

# Медианная цена для availability == 365
sub = merged[merged['availability'] == 365]
round(sub['price'].median(), 2)

# Сколько районов с ненулевым размахом цены
price_range = merged.groupby('area')['price'].apply(lambda x: x.max() - x.min())
(price_range != 0).sum()
```

---

### B3 Q4 — Объединение таблиц (train4_B3: id, host name, price per night)
```python
df4 = pd.read_csv('train4_B3.csv')
df4 = df4.rename(columns={'price per night': 'price'})

# Объединяем через pd.concat (одинаковые колонки)
combined = pd.concat([
    df_main[['id', 'host name', 'price']],
    df4[['id', 'host name', 'price']]
], ignore_index=True)

# Хозяин с наибольшим количеством объявлений
top_host = combined['host name'].value_counts().idxmax()

# Сколько он заработает за 10 дней
earnings = combined[combined['host name'] == top_host]['price'].sum() * 10

# Для хозяев с > 1 объявлением: дисперсия медианной цены
multi = combined.groupby('host name').filter(lambda x: len(x) > 1)
median_prices = multi.groupby('host name')['price'].median()
round(median_prices.var(ddof=0), 2)   # или ddof=1 — смотри формулировку
```

---

### B3 Q5 — Плотность (train5_B3: id, area, pop, km2)
```python
df5 = pd.read_csv('train5_B3.csv')
merged = df_main.merge(df5, on='id', how='inner')

# Объединяем уникальные area с их pop/km2
area_info = df5.drop_duplicates('area')[['area', 'pop', 'km2']].copy()
area_info['pop_density'] = area_info['pop'] / area_info['km2']

# Район на Манхэттене с НАИМЕНЬШЕЙ плотностью населения
manhattan_areas = merged[merged['borough'] == 'Manhattan']['area'].unique()
manhattan_density = area_info[area_info['area'].isin(manhattan_areas)]
manhattan_density.loc[manhattan_density['pop_density'].idxmin(), 'area']

# Плотность предложений (id/km2) по районам
listing_counts = merged.groupby('area')['id'].count().reset_index(name='listing_count')
listing_density = listing_counts.merge(area_info[['area', 'km2']], on='area')
listing_density['rent_density'] = listing_density['listing_count'] / listing_density['km2']

# Район с наибольшей плотностью → значение
max_row = listing_density.loc[listing_density['rent_density'].idxmax()]
round(max_row['rent_density'], 2)
```

---

## Шпаргалка — частые ошибки

| Ситуация | Правильно |
|----------|-----------|
| Дисперсия — генеральная или выборочная? | В задании почти всегда **генеральная** → `var(ddof=0)` |
| Объединение по разным именам ключей | `merge(left_on='id', right_on='id rent')` |
| `value_counts().idxmax()` vs `groupby().count().idxmax()` | Для одного столбца — `value_counts()`; для групп с несколькими столбцами — `groupby()` |
| Фильтр ОТ до ДО включительно | `>=` и `<=` (не `>` и `<`) |
| Утренние часы до 11:59 | `dt.hour <= 11` (не `<= 12`!) |
| `na=False` в `str.contains` | Обязательно, иначе NaN → ошибка |
| Номер недели | `dt.isocalendar().week.astype(int)` (не `dt.week` — устарело) |
| `pd.concat` vs `merge` | concat — вертикальное склеивание (одинаковые колонки); merge — горизонтальное по ключу |
