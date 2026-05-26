# Памятка: Решение заданий Части C (ML Pipeline)

---

## ГЛАВНОЕ: Дерево выбора статистического теста

```
Какого типа переменные?
│
├── Оба признака КАТЕГОРИАЛЬНЫЕ (пол × тип тренировки)
│       → ХИ-КВАДРАТ  (chi2_contingency)
│       → Шапиро НЕ нужен!
│
├── Числовые значения, сравниваем СРЕДНИЕ или РАСПРЕДЕЛЕНИЯ
│       │
│       ├── 1 группа vs константа  → Одновыборочный t-тест (ttest_1samp)
│       │
│       ├── Одни и те же люди до/после  → Парный t-тест (ttest_rel)
│       │
│       └── Две НЕЗАВИСИМЫЕ группы:
│               Шапиро на каждой группе
│               ├── Оба p > 0.05  → Нормальное → t-тест (ttest_ind)
│               └── Хотя бы один p ≤ 0.05  → Манна-Уитни (mannwhitneyu)
│
└── Сравниваем ДОЛИ (проценты)
        → Z-тест для пропорций (proportions_ztest)
        → ИЛИ Манна-Уитни на бинарной переменной 0/1
```



> ⚠️ Задания C выполняются СТРОГО ПОСЛЕДОВАТЕЛЬНО: C1 → C2 → C3 → C4 → C5.
> Каждый шаг меняет датафрейм. Нельзя запускать задание отдельно!

---

## Два варианта — РАЗНЫЕ паттерны

| Шаг | Стандартный вариант | Вариант «фитнес/другой» |
|-----|---------------------|------------------------|
| C1  | Заполнить пропуски **модой по группам** (groupby + transform) | Заполнить пропуски по **условию** (если X > среднего → 1, иначе → 0) |
| C2  | **LabelEncoder** (A=0, B=1...) | **OneHotEncoder** (добавить отдельные столбцы) |
| C3  | Выбросы **1.5×IQR** | Выбросы **2×STD** |
| C4  | LogisticRegression + log_loss | LinearRegression + log_loss (ambiguous) |
| C5  | Шапиро → t-test / Манна-Уитни для средних | Шапиро → Манна-Уитни для **долей** (бинарная переменная) |

---

## C1 — Заполнение пропусков

### Вариант A: Мода по группам (стандартный)
```python
print(df['category'].isnull().sum())   # C1a: количество пропусков

df['category'] = df.groupby('region')['category'].transform(
    lambda x: x.fillna(x.mode()[0])
)

share_A = round((df['category'] == 'A').sum() / len(df), 2)  # C1b: доля категории
```

### Вариант B: Условное заполнение (по порогу)
```python
print(df['goal_achieved'].isnull().sum())  # C1a: количество пропусков

mean_steps = df['avg_daily_steps'].mean()  # порог = среднее
mask = df['goal_achieved'].isnull()

df.loc[mask & (df['avg_daily_steps'] >  mean_steps), 'goal_achieved'] = 1
df.loc[mask & (df['avg_daily_steps'] <= mean_steps), 'goal_achieved'] = 0

# C1b: доля НЕ достигших цели (значение 0)
share_0 = round((df['goal_achieved'] == 0).sum() / len(df), 2)
```

> **Ключ различия:** если в задании написано "если значение X выше среднего → 1, иначе → 0" — это вариант B с `.loc[mask & condition]`.

---

## C2 — Кодирование категорий

### Вариант A: LabelEncoder (порядковые / одна колонка)
```python
from sklearn.preprocessing import LabelEncoder

df['category_encoded'] = LabelEncoder().fit_transform(df['category'])
# A=0, B=1, C=2, D=3  (по алфавиту)

print(round(df['category_encoded'].std(), 2))
```

### Вариант B: OneHotEncoder (номинальные / несколько колонок)
```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, dtype=int)
encoded  = ohe.fit_transform(df[['workout_type']])

# Названия столбцов: workout_type_C, workout_type_D, workout_type_S, workout_type_Y
new_cols = [f'workout_type_{cat}' for cat in ohe.categories_[0]]
ohe_df   = pd.DataFrame(encoded, columns=new_cols, index=df.index)
df       = pd.concat([df, ohe_df], axis=1)

print(len(new_cols))  # количество новых столбцов
```

> **Правило именования:** `столбец_значение` → `workout_type_C`, `color_green` и т.д.
> **Количество новых столбцов = количество уникальных значений** в исходном столбце.

---

## C3 — Удаление выбросов

### Вариант A: 1.5×IQR (стандартный)
```python
Q1  = df['score'].quantile(0.25)
Q3  = df['score'].quantile(0.75)
IQR = Q3 - Q1

df_clean = df[(df['score'] >= Q1 - 1.5*IQR) & (df['score'] <= Q3 + 1.5*IQR)]
print(len(df_clean))
```

### Вариант B: 2×STD (по среднему ± 2 стандартных отклонения)
```python
mean_val = df['age'].mean()
std_val  = df['age'].std()

df_clean = df[(df['age'] >= mean_val - 2*std_val) &
              (df['age'] <= mean_val + 2*std_val)].copy()
print(len(df_clean))
```

> **Как читать задание:**
> - «1.5×IQR» → вариант A
> - «2×STD», «2 стандартных отклонения» → вариант B

---

## C4 — Признак X1, разбивка, модель

### Нахождение X1 (наибольшая монотонная связь с целевой)
```python
# Исключаем целевую, id, OHE-столбцы
exclude    = {'avg_daily_steps', 'user_id'} | set(new_cols)
candidates = [c for c in df_clean.select_dtypes(include=[np.number]).columns
              if c not in exclude]

corr_s = df_clean[candidates + ['avg_daily_steps']].corr(method='spearman')
X1     = corr_s['avg_daily_steps'].drop('avg_daily_steps').abs().idxmax()
print("X1 =", X1)
```

### Разбивка и среднее X1 в train
```python
from sklearn.model_selection import train_test_split

X = df_clean[[X1, 'workout_type_C']]
y = df_clean['avg_daily_steps']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(round(X_train[X1].mean(), 2))   # C4b
```

### Линейная регрессия + предсказание
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Предсказать для X1 = мода, workout_type_C = 1
X1_mode   = df_clean[X1].mode()[0]
new_point = pd.DataFrame({X1: [X1_mode], 'workout_type_C': [1]})
pred      = round(model.predict(new_point)[0], 2)
print("Предсказание:", pred)   # C4c
```

### Log loss (если в задании линейная регрессия → всё равно строим LogisticRegression)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Целевая для классификации — goal_achieved (бинарная)
y_train_cls = df_clean.loc[X_train.index, 'goal_achieved'].astype(int)
y_test_cls  = df_clean.loc[X_test.index,  'goal_achieved'].astype(int)

log_model    = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train_cls)

ll = round(log_loss(y_test_cls, log_model.predict_proba(X_test)), 2)
print("Log loss:", ll)   # C4d
```

---

## Хи-квадрат: категориальный × категориальный

```python
from scipy.stats import chi2_contingency

# Таблица сопряжённости
ct = pd.crosstab(df_clean['gender'], df_clean['workout_type'])
print(ct)

# Тест
chi2, p, dof, expected = chi2_contingency(ct)
print(f"χ² = {round(chi2, 2)},  p = {p:.4f},  dof = {dof}")

# Вывод
alpha = 0.05
if p < alpha:
    print("Связь между признаками статистически значима")
else:
    print("Статистически значимой связи не обнаружено")
```

> **Степени свободы:** `dof = (кол-во строк - 1) × (кол-во столбцов - 1)`
> Например: 2 пола × 4 типа тренировки → dof = (2-1)×(4-1) = **3**

---

## C5 — Тестирование гипотез о долях

### Схема выбора теста для бинарных переменных (0/1)
```
Шапиро-Уилка на 0/1 данных → ВСЕГДА отклонит нормальность (p ≤ 0.05)
→ Выбираем НЕ-параметрический тест

Сравниваем ДОЛИ (пропорции) 2 групп?
├── Можно: Z-тест для пропорций (proportions_ztest)
└── Или:   U-тест Манна-Уитни (mannwhitneyu)
```

### Полный код C5
```python
from scipy import stats

free_data = df_clean[df_clean['subscription'] == 'free']['goal_achieved'].astype(int)
pro_data  = df_clean[df_clean['subscription'] == 'pro' ]['goal_achieved'].astype(int)

# Шапиро
_, p_free = stats.shapiro(free_data)
_, p_pro  = stats.shapiro(pro_data)
print(f"Шапиро: free p={p_free:.4f}, pro p={p_pro:.4f}")

# Оба > 0.05 → t-тест. Хотя бы один ≤ 0.05 → Манна-Уитни
if p_free > 0.05 and p_pro > 0.05:
    stat, p_val = stats.ttest_ind(free_data, pro_data, equal_var=True)
    test_name = "t-тест"
else:
    stat, p_val = stats.mannwhitneyu(free_data, pro_data, alternative='two-sided')
    test_name = "Манна-Уитни"

print(f"{test_name}: stat={round(stat, 2)}, p={p_val:.4f}")

# Вывод на уровне значимости 3%
alpha = 0.03
if p_val < alpha:
    print(f"p={p_val:.4f} < {alpha} → H0 ОТВЕРГАЕТСЯ — доли значимо различаются")
else:
    print(f"p={p_val:.4f} >= {alpha} → H0 НЕ отвергается — различий не обнаружено")
```

---

## Быстрая сравнительная шпаргалка

### Когда какой энкодер?

| Признак | Тип | Энкодер |
|---------|-----|---------|
| A, B, C, D (порядок важен) | Порядковый | LabelEncoder → A=0, B=1… |
| Кардио, Йога, Силовые (нет порядка) | Номинальный | OneHotEncoder → отдельные столбцы |

### Когда какой метод удаления выбросов?

| Метод | Когда | Формула |
|-------|-------|---------|
| 1.5×IQR | По умолчанию, робастный к форме распределения | Q1−1.5·IQR ÷ Q3+1.5·IQR |
| 2×STD | Когда в задании явно написано «2 стандартных отклонения» | mean−2·std ÷ mean+2·std |

### Когда Z-тест для пропорций?
```python
# Если задание спрашивает именно о ДОЛЯХ (пропорциях)
from statsmodels.stats.proportion import proportions_ztest

count = np.array([free_data.sum(), pro_data.sum()])
nobs  = np.array([len(free_data), len(pro_data)])
stat, p_val = proportions_ztest(count, nobs)
# pip install statsmodels если не установлен
```
