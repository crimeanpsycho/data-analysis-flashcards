#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Решение Части C (вариант 3) — фитнес-приложение
Запуск: python3 solve_C3.py   (положи data_C_3.csv рядом)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from scipy import stats

# ────────────────────────────────────────────────────────────────
# Загрузка
# ────────────────────────────────────────────────────────────────
df = pd.read_csv('data_C_3.csv')
print("Форма датасета:", df.shape)
print("Первые строки:")
print(df.head())
print()

# ════════════════════════════════════════════════════════════════
# C1a  Пропуски в goal_achieved
# ════════════════════════════════════════════════════════════════
missing = df['goal_achieved'].isnull().sum()
print(f"C1a  Пропусков в goal_achieved: {missing}")

# ════════════════════════════════════════════════════════════════
# C1b  Заполнить по условию avg_daily_steps
# ════════════════════════════════════════════════════════════════
mean_steps = df['avg_daily_steps'].mean()
mask = df['goal_achieved'].isnull()

df.loc[mask & (df['avg_daily_steps'] >  mean_steps), 'goal_achieved'] = 1
df.loc[mask & (df['avg_daily_steps'] <= mean_steps), 'goal_achieved'] = 0

share_0 = round((df['goal_achieved'] == 0).sum() / len(df), 2)
print(f"C1b  Доля НЕ достигших цели (0): {share_0}")
print()

# ════════════════════════════════════════════════════════════════
# C2  OneHotEncoder для workout_type
# ════════════════════════════════════════════════════════════════
ohe = OneHotEncoder(sparse_output=False, dtype=int)
encoded = ohe.fit_transform(df[['workout_type']])

# Имена столбцов: workout_type_C, workout_type_D, ...
new_cols = [f'workout_type_{cat}' for cat in ohe.categories_[0]]
ohe_df   = pd.DataFrame(encoded, columns=new_cols, index=df.index)
df       = pd.concat([df, ohe_df], axis=1)

print(f"C2   Новых столбцов после OHE: {len(new_cols)}")
print(f"     Названия: {new_cols}")
print()

# ════════════════════════════════════════════════════════════════
# C3  Удаление выбросов в age по критерию 2×STD
# ════════════════════════════════════════════════════════════════
mean_age = df['age'].mean()
std_age  = df['age'].std()
lower    = mean_age - 2 * std_age
upper    = mean_age + 2 * std_age

df_clean = df[(df['age'] >= lower) & (df['age'] <= upper)].copy()
print(f"C3   Наблюдений после удаления выбросов: {len(df_clean)}")
print(f"     (удалено: {len(df) - len(df_clean)})")
print()

# ════════════════════════════════════════════════════════════════
# C4  Линейная регрессия: avg_daily_steps ~ X1 + workout_type_C
# ════════════════════════════════════════════════════════════════

# a) X1 — признак с наибольшей монотонной связью с avg_daily_steps
# Кандидаты: числовые столбцы кроме целевой, id и OHE-столбцов
exclude = {'avg_daily_steps', 'user_id'} | set(new_cols)
candidates = [c for c in df_clean.select_dtypes(include=[np.number]).columns
              if c not in exclude]

corr_spearman = df_clean[candidates + ['avg_daily_steps']].corr(method='spearman')
X1 = corr_spearman['avg_daily_steps'].drop('avg_daily_steps').abs().idxmax()
print(f"C4a  X1 = '{X1}'  (Спирмен = {corr_spearman.loc[X1, 'avg_daily_steps']:.4f})")

# b) Разбивка 80/20, random_state=42
X = df_clean[[X1, 'workout_type_C']]
y = df_clean['avg_daily_steps']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

mean_X1_train = round(X_train[X1].mean(), 2)
print(f"C4b  Среднее {X1} в train: {mean_X1_train}")

# c) Обучение линейной регрессии, предсказание
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

X1_mode  = df_clean[X1].mode()[0]
new_point = pd.DataFrame({X1: [X1_mode], 'workout_type_C': [1]})
pred_c   = round(lin_model.predict(new_point)[0], 2)
print(f"C4c  X1_mode={X1_mode}, workout_type_C=1 → avg_daily_steps ≈ {pred_c}")
print(f"     Коэффициенты: w0={lin_model.intercept_:.4f}, "
      f"w1({X1})={lin_model.coef_[0]:.4f}, "
      f"w2(workout_type_C)={lin_model.coef_[1]:.4f}")

# d) Log loss — в задании сказано "логистической функции потерь"
#    Интерпретируем как LogisticRegression на goal_achieved с теми же признаками
y_train_cls = df_clean.loc[X_train.index, 'goal_achieved'].astype(int)
y_test_cls  = df_clean.loc[X_test.index,  'goal_achieved'].astype(int)

log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train_cls)
y_pred_proba = log_model.predict_proba(X_test)
ll = round(log_loss(y_test_cls, y_pred_proba), 2)
print(f"C4d  Log loss = {ll}")
print()

# ════════════════════════════════════════════════════════════════
# C5  Сравнение долей goal_achieved: free vs pro
# ════════════════════════════════════════════════════════════════
free_data = df_clean[df_clean['subscription'] == 'free']['goal_achieved'].astype(int)
pro_data  = df_clean[df_clean['subscription'] == 'pro' ]['goal_achieved'].astype(int)

print(f"C5   free: n={len(free_data)}, достигли цели={free_data.sum()}, доля={free_data.mean():.3f}")
print(f"C5   pro:  n={len(pro_data)},  достигли цели={pro_data.sum()},  доля={pro_data.mean():.3f}")

# Проверка нормальности Шапиро (для бинарных данных всегда отклонится)
_, p_free = stats.shapiro(free_data)
_, p_pro  = stats.shapiro(pro_data)
print(f"\nC5a  Шапиро-Уилка: free p={p_free:.4f}, pro p={p_pro:.4f}")
if p_free <= 0.05 or p_pro <= 0.05:
    print("     Нормальность НЕ подтверждена → U-тест Манна-Уитни")
    chosen_test = "mannwhitney"
else:
    print("     Нормальность подтверждена → t-тест")
    chosen_test = "ttest"

# C5b: выбранный тест
if chosen_test == "mannwhitney":
    stat, p_val = stats.mannwhitneyu(free_data, pro_data, alternative='two-sided')
    test_name = "U-тест Манна-Уитни"
else:
    stat, p_val = stats.ttest_ind(free_data, pro_data, equal_var=True)
    test_name = "t-тест"

print(f"\nC5b  {test_name}: статистика = {round(stat, 2)},  p-value = {p_val:.4f}")

# C5c: вывод при α = 3% (0.03)
alpha = 0.03
print(f"\nC5c  Уровень значимости α = {alpha*100:.0f}%")
if p_val < alpha:
    print(f"     p={p_val:.4f} < α={alpha} → H0 ОТВЕРГАЕТСЯ")
    print("     Вывод: доля достигших цели ЗНАЧИМО отличается между free и pro")
else:
    print(f"     p={p_val:.4f} >= α={alpha} → H0 НЕ отвергается")
    print("     Вывод: статистически значимых различий в доле достигших цели не обнаружено")
