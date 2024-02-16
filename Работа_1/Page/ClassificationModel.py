import numpy as np
import streamlit as stm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score, roc_auc_score
from Scripts.page_parameters import base_param
from Scripts.ClassificationLearn import classification_learning
from Scripts.FormatData import format_class


base_param(page_title="Модель Классификации")


dataset = stm.file_uploader(label="Загрузка датасета", type=["csv"])
if dataset is not None:
    stm.markdown("### Информация о датасете")
    dataset = pd.read_csv(dataset, sep=",")
    dataset = format_class(dataset)
    stm.dataframe(dataset)

    stm.markdown("### Тепловая карта")
    fig = plt.figure(figsize=(16, 5))
    sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm", fmt=".3f")
    stm.pyplot(fig)

    stm.markdown("### Актуальные и предсказанные значения")
    predict, actual = classification_learning(data=dataset, sign="fraud")
    stm.dataframe(np.array([predict, actual]))

    stm.markdown("### Метрики и графическое представление")
    fig = plt.figure()
    plt.plot(actual.to_numpy()[:400], label="Actual")
    plt.plot(predict[:400], label="Prediction")
    stm.pyplot(fig)

    stm.markdown(f"Точность: {accuracy_score(actual, predict)}")
    stm.markdown(f"Средняя абсолютная ошибка: {mean_absolute_error(actual, predict)}")
    stm.markdown(f"значение f1: {f1_score(actual, predict)}")
    stm.markdown(f"Roc: {roc_auc_score(actual, predict)}")
