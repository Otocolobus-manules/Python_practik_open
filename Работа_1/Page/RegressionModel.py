import numpy as np
import streamlit as stm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, max_error, explained_variance_score
from Scripts.page_parameters import base_param
from Scripts.RegressionLearn import regression_learning
from Scripts.FormatData import format_reg


base_param(page_title="Модель Регрессии")


dataset = stm.file_uploader(label="Загрузка датасета", type=["csv"])
if dataset is not None:
    stm.markdown("### Информация о датасете")
    dataset = pd.read_csv(dataset, sep=",")
    dataset = format_reg(dataset)
    stm.dataframe(dataset)

    stm.markdown("### Тепловая карта")
    fig = plt.figure(figsize=(16, 5))
    sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", fmt=".3f")
    stm.pyplot(fig)

    stm.markdown("### Актуальные и предсказанные значения")
    predict, actual = regression_learning(data=dataset, sign="price")
    stm.dataframe(np.array([predict, actual]))

    stm.markdown("### Метрики и графическое представление")
    fig = plt.figure()
    plt.plot(actual.to_numpy()[:400], label="Actual")
    plt.plot(predict[:400], label="Prediction")
    stm.pyplot(fig)

    stm.markdown(f"r^2: {r2_score(actual, predict)}")
    stm.markdown(f"Средняя абсолютная погрешность: {mean_absolute_error(actual, predict)}")
    stm.markdown(f"Максимальная ошибка: {max_error(actual, predict)}")
    stm.markdown(f"Показатель дисперсии: {explained_variance_score(actual, predict)}")
