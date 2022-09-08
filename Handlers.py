import logging
import os

import matplotlib.pyplot as plt  # библиотека визуализации
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats  # библиотека для расчетов
from scipy.stats import norm
from scipy.stats import t

plt.style.use('ggplot')


def get_columns_null_info_df(df):
    """
    Функция для получиения информации по количеству и % нулевых значнений и типа данных в каждом из столбцов

    :param df: Датафрейм для анализа
    :return: Итоговый датафрейм с разультатами анализа
    """

    # создаём пустой список
    unique_list = []

    # пробегаемся по именам столбцов в таблице
    for col in df.columns:
        item = (col, df[col].isnull().sum(), round(df[col].isnull().mean() * 100, 2), df[col].dtype)
        # добавляем кортеж в список
        unique_list.append(item)
        # создаю датафрейм который будет возвращаться
    unique_counts = pd.DataFrame(
        unique_list,
        columns=['Column Name', 'Count Null', '% Null', 'Type']
    )
    # .sort_values(by='Num Unique', ignore_index=True)

    return unique_counts


def get_top_unique_values(df, level=0):
    """
    Функция для получения инфомации по уникальным значениям в признаках

    :param df: Датафрейм для анализа
    :param level: Уровень уникальности в %, признаки ниже этого уровня не выводятся
    :return: Возвращает датафрейм с именем признака, 
    количестве уникальных значений, 
    сколько % от всей выборки занимают уникальные значения,
    наиболее часто повторяющимся уникальным значением,
    количество повторов,
    сколько % от выборки это значение занимает
    
    Когда % Unique' > 30, то это повод задуматься об уменьшении числа категорий
    """

    cols = list(df.columns)

    df_len = df.shape[0]

    # создаём пустой список
    unique_list = []

    for col in cols:
        col_lev = round(df[col].value_counts(normalize=True).values[0] * 100, 2)

        if col_lev > level:
            item = (col,
                    df[col].nunique(),
                    round(df[col].nunique() / df_len * 100, 2),
                    df[col].value_counts(normalize=True).index[0],
                    df[col].value_counts().values[0],
                    col_lev
                    )
            # добавляем кортеж в список
            unique_list.append(item)

    unique_values = pd.DataFrame(
        unique_list,
        columns=['Column Name', 'Count Unique', '% Unique', 'Top Value', 'Top Value Count', 'Top Value %']
    )

    return unique_values


def get_columns_isnull_info_df(df, in_percent=True):
    """
    Функия для получения количества пропущенных элементов в % по каждому из столбцов

    :param df: Датафрейм для анализа
    :return: Итоговый датафрейм с разультатами анализа
    """

    if in_percent:
        cols_null = df.isnull().mean() * 100
        col_name = '% of null'
    else:
        cols_null = df.isnull().sum()
        col_name = 'Count null'

    cols_with_null = cols_null[cols_null > 0].sort_values(ascending=False)

    if len(cols_with_null) > 0:
        cols_with_null = cols_with_null.to_frame().reset_index()
        cols_with_null.columns = ['feature', col_name]

        return cols_with_null
    else:
        return 'No one null values!'


def outliers_iqr(df, feature, log_scale=False, left=1.5, right=1.5):
    """
    Функция для определения выбросов по методу Тьюки.

    :param df: Исходный датафрейм
    :param feature: Фитча датафрейма для определения выбросов
    :param log_scale: Нужно ли логарифмировать рассмативаемый признак
    :param left: Множитель для определения левой границы выброса, по умолчанию 1.5
    :param right: Множитель для определения правой границы выброса, по умолчанию 1.5
    :return: Функция возвращает датафрейм с выбросами и отчищенный от выбросов датафрейм
    """

    x = df[feature]

    if log_scale:
        x = np.log(x)

    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75),
    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * left)
    upper_bound = quartile_3 + (iqr * right)

    outliers = df[(x < lower_bound) | (x > upper_bound)]
    cleaned = df[(x >= lower_bound) & (x <= upper_bound)]
    info = f'Выбросы: {outliers.shape[0]} строк ({outliers.shape[0] / df.shape[0] * 100:.2f}%).'

    return info, outliers, cleaned


def outliers_z_score(df, feature, log_scale=False, left=3, right=3):
    """
    Функция для определения выбросов по методу 3х сигм

    :param df: Исходный датафрейм
    :param feature: Фитча датафрейма для определения выбросов
    :param log_scale: Нужно ли логарифмировать рассмативаемый признак
    :return: Функция возвращает датафрейм с выбросами и отчищенный от выбросов датафрейм
    """

    x = df[feature]

    if log_scale:
        # Если в серии минимальное значение 0,
        # то небходимо добавить 1 во всю серии, т.к. логарифм от 0 невозможен
        x = np.log(x + 1)

    mu = x.mean()
    sigma = x.std()

    lower_bound = mu - left * sigma
    upper_bound = mu + right * sigma

    outliers = df[(x < lower_bound) | (x > upper_bound)]
    cleaned = df[(x >= lower_bound) & (x <= upper_bound)]
    info = f'Выбросы: {outliers.shape[0]} строк ({outliers.shape[0] / df.shape[0] * 100:.2f}%).'

    return info, outliers, cleaned


def get_low_inform_features_list(df, level=0.95):
    """
    Метод для получения списка признаков с низкой информативностью

    :param df: Датафрейм для анализа признаков
    :param level: Уровень попадания в низкоинформативные, по умолчанию 0.95
    :return: Возвращает список признаков датафрейма, которые имеют низкую информативность
    """

    # список неинформативных признаков
    low_inform_features = []

    # цикл по всем столбцам
    for col in df.columns:
        # наибольшая относительная частота в признаке
        top_freq = df[col].value_counts(normalize=True).max()
        # доля уникальных значений от размера признака
        nunique_ratio = df[col].nunique() / df[col].count()

        # сравниваем наибольшую частоту с порогом
        if top_freq > level:
            low_inform_features.append(col)
            print(f'{col}: {round(top_freq * 100, 2)}% одинаковых значений')
        # сравниваем долю уникальных значений с порогом
        elif nunique_ratio > level:
            low_inform_features.append(col)
            print(f'{col}: {round(nunique_ratio * 100, 2)}% уникальных значений')

    if low_inform_features:
        return low_inform_features

    return 'Нет малоинформативных признаков!'


def QQ_Plots(df, column_name):
    """
    Метод для получения графиков Q-Q Plots для проверки нормальности распределения фитчи

    :param df: Датафрейм для анализа признаков
    :param column_name: Имя столбца по которому построить графики
    """

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)  # задаем сетку рисунка количество строк и столбцов
    stats.probplot(df[column_name], plot=plt)  # qq plot

    plt.subplot(1, 2, 2)  # располагаем второй рисунок рядом
    plt.hist(df[column_name])  # гистограмма распределения признака
    plt.title(column_name)

    plt.tight_layout()  # чтобы графики не наезжали другу на друга, используем tight_layout

    plt.show()  # просмотр графика


def Hist_and_Box_Plots(df, column_name, bins=30):
    """
    Функция для построения диаграммы распределения и коробки с усами
    для оценки нормальности распределения и поиска выбросов

    :param df: Исходный датафрейм
    :param column_name: Имя признака для анализа
    :param bins: Количество групп по умолчанию 30
    :return: Выводит график
    """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))
    axes[0].ticklabel_format(style='plain', axis='x')
    histplot = sns.histplot(data=df, x=column_name, bins=bins, ax=axes[0])

    histplot.set_title(f'{column_name} Histogram')
    axes[1].ticklabel_format(style='plain', axis='x')
    boxplot = sns.boxplot(data=df, x=column_name, ax=axes[1])
    boxplot.set_title(f'{column_name} Boxplot')

    plt.tight_layout()
    plt.show()


def null_heatmap_plot(df):
    """
    Функция для построения тепловой карты пропуска в столбцах датафрейма.
    Выводятся только столбцы в которых есть пропуски!

    :param df:
    :return:
    """

    cols_null_perc = df.isnull().mean() * 100
    cols_null_perc = cols_null_perc[cols_null_perc > 0]
    cols_null_perc = cols_null_perc.index.to_list()

    colors = ['blue', 'yellow']
    fig = plt.figure(figsize=(16, 10))
    ax = sns.heatmap(df[cols_null_perc].isnull(), cmap=colors, cbar=False)
    ax.set_title('Null Heatmap')


def confidence_interval_v0(n, x_mean, sigma, gamma=0.95):
    """
    Функци для расчета доверительного интервала, когда известно истинное стандартное отклонение на всей совокупности

    :param n: Количество элементов в выборке
    :param x_mean: Выборочное среднее
    :param sigma: Истинное стандартное отклонение
    :param gamma: Уровень надежности
    :return: Возвращает кортеж границ доверительного интервала
    """

    alpha = 1 - gamma
    z_crit = -norm.ppf(alpha / 2)
    print(z_crit)
    eps = z_crit * sigma / (n ** 0.5)  # погрешность
    lower_bound = x_mean - eps  # левая (нижняя) граница
    upper_bound = x_mean + eps  # правая (верхняя) граница
    return round(lower_bound, 2), round(upper_bound, 2)


def confidence_interval(n, x_mean, x_std, gamma=0.95):
    """
    Функци для расчета доверительного интервала, когда известно выборочное стандартное отклонение

    :param n: Количество элементов в выборке
    :param x_mean: Выборочное среднее
    :param x_std: Выборочное стандартное отклонение
    :param gamma: Уровень надежности
    :return: Возвращает кортеж границ доверительного интервала
    """

    alpha = 1 - gamma
    k = n - 1
    t_crit = -t.ppf(alpha / 2, k)  # t-критическое
    eps = t_crit * x_std / (n ** 0.5)  # погрешность
    lower_bound = x_mean - eps  # левая (нижняя) граница
    upper_bound = x_mean + eps  # правая (верхняя) граница

    return round(lower_bound, 2), round(upper_bound, 2)


def proportions_confidence_interval(n, x_p, gamma=0.95):
    """
    Функция расчета доверительного интервала для конверсий.
    Конверсия - доля пользователей совершивших целевое действие.

    :param n: Размер выборки
    :param x_p: Выборочная пропорция или конверсия x/n
    :param gamma: Уровень надежности
    :return: Возвращает кортеж из границ доверительного интервала
    """

    alpha = 1 - gamma  # уровень значимости
    z_crit = -norm.ppf(alpha / 2)  # z критическое
    eps = z_crit * (x_p * (1 - x_p) / n) ** 0.5  # погрешность
    lower_bound = x_p - eps  # левая (нижняя) граница
    upper_bound = x_p + eps  # правая (верхняя) граница
    # возвращаем кортеж из округлённых границ интервала
    return round(lower_bound * 100, 2), round(upper_bound * 100, 2)


def diff_proportions_confidence_interval(n, xp, gamma=0.95):
    """
    Функция для расчета доверительного интервала разницы конверсий

    :param n: Список размеров выборки для варианта A и B
    :param xp: Список выборочных пропорция или конверсий для варианта A и B
    :param gamma: Уровень надежности
    :return: Возвращает кортеж из границ доверительного интервала
    """

    alpha = 1 - gamma  # уровень значимости
    diff = xp[1] - xp[0]  # выборочная разница конверсий групп B и A
    z_crit = -norm.ppf(alpha / 2)  # z критическое
    eps = z_crit * (xp[0] * (1 - xp[0]) / n[0] + xp[1] * (1 - xp[1]) / n[1]) ** 0.5  # погрешность
    lower_bound = diff - eps  # левая (нижняя) граница
    upper_bound = diff + eps  # правая (верхняя) граница
    # возвращаем кортеж из округлённых границ интервала
    return round(lower_bound * 100, 2), round(upper_bound * 100, 2)


def get_logger(path, file):
    """
    Функция для создания лог-файла и записи в него информации

    :param path: путь к директории
    :param file: имя файла
    :return: Возвращает объект логгера
    """

    # проверяем, существует ли файл
    log_file = os.path.join(path, file)

    # если  файла нет, создаем его
    if not os.path.isfile(log_file):
        open(log_file, "w+").close()
    # формат логирования
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"
    # формат даты
    date_format = "%Y-%m-%d %H:%M:%S"
    # конфигурируем лог-файл
    logging.basicConfig(format=file_logging_format, datefmt=date_format, encoding='utf-8')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # создадим хэнлдер для записи лога в файл
    handler = logging.FileHandler(log_file, encoding='utf-8')
    # установим уровень логирования
    handler.setLevel(logging.DEBUG)
    # создадим формат логирования, используя file_logging_format
    formatter = logging.Formatter(file_logging_format, date_format)
    handler.setFormatter(formatter)
    # добавим хэндлер лог-файлу
    logger.addHandler(handler)
    return logger


def get_X_y_dataset(train_data, target_feature):
    """
    Функция для получения из полного датасета двух, тренировочного и датасета, который содержит лишь целевой признак

    :param train_data: Исходный, полный датасет для тренировки
    :param target_feature: Имя целевого признака
    :return: Возвращает кортеж датасетов (X, y)
    """

    X = train_data.drop([target_feature], axis=1)
    y = train_data[target_feature]

    return X, y


def merge_train_and_test_to_full(train_data, test_data, target_feature):
    """
    Функция для объединения тернировочного и тестового датасета

    :param train_data: Тренировочный датасет с целевым признаком
    :param test_data: Тестовый датасет, столбца целового признака нет
    :param target_feature: Имя целевого признака
    :return: Возвращает объединенный датасет
    """

    # Реформатирую порядок столбцов
    train_data = reformat_columns(train_data, target_feature)

    train_data['dataset'] = 'train'  # Помечаю, что это тренировочный датафрейм

    test_data[target_feature] = 0  # На тестовом датафрейме целевой признак заполняю нулями
    test_data['dataset'] = 'test'  # Помечаю, что это тестовый датафрейм

    full_data = train_data.append(test_data).reset_index(drop=True)
    full_data['dataset'] = full_data['dataset'].astype('category')

    full_data = full_data.convert_dtypes()

    return full_data


def split_full_to_train_and_test(full_data, target_feature):
    """
    Функция разбивает полный датасет на train и test части

    :param full_data: Полный датасет, содержит признак dataset со значениями train, test
    :param target_feature: Имя целевого признака
    :return: Возвращает кортеж из train и test датасета
    """
    
    # Реформатирую порядок столбцов
    full_data = reformat_columns(full_data, target_feature)

    # Обработанный датасет делим назад на треин и тест части
    train_data = full_data.query('dataset == "train"').drop(['dataset'], axis=1)
    test_data = full_data.query('dataset == "test"').drop(['dataset', target_feature], axis=1)

    return train_data, test_data


def reformat_columns(full_data, target_feature):
    """
    Функция реформатирует порядок расположения признаков. Целевой признак будет теперь в конце

    :param full_data: Исходный датафрейм
    :param target_feature: Целевой признак, который нужно переместить в конец
    :return:
    """

    columns = list(full_data.columns)
    cols_to_end = ['dataset', target_feature]

    for col in cols_to_end:
        if col in columns:
            columns.remove(col)
            columns.append(col)

    return full_data[columns]


def calculate_target_feature_per_category(full_data, category_feature: str, TARGET_FEATURE: str, func='mean'):
    """
    Функция для расчета среднего (мединаы, СКО) значения для целеового по каждой категории выбранного признака

    :param full_data: Полный датасет
    :param category_feature: Имя категориального признака в разрезе которого нужно посчитать целевой
    :param TARGET_FEATURE: Имя целевого признака
    :param func: Функция по которой проводить аггрегацию. mean, median, std
    :return:
    """

    # Из датасета получение датафрейма категорий с их долей в % ко всему датасету
    df = pd.DataFrame(full_data[category_feature].value_counts(True) * 100).reset_index()
    df.columns = [category_feature, '%']  # переименовываю столбцы

    # Проверяю, что передан полный датасет, в котором есть признак отвечающий за разделение выборок на треин и тест
    if 'dataset' in full_data.columns:
        # Получаю треин датасет в котором есть значения целевого признака
        train_data = full_data.query('dataset == "train"').drop(['dataset'], axis=1)
    else:
        train_data = full_data

    # Группирую по категориям получаю агрегированное по указанной функции значения целевого признака
    group_category = train_data.groupby([category_feature])[TARGET_FEATURE].agg([func]).reset_index()

    return df.merge(group_category, on=category_feature, how='left')


def plot_categories(full_data, category_feature, level=5):
    """
    Столбчатая диаграмма для категориального признака

    :param full_data: Датасет
    :param category_feature: Имя признака
    :param level: Уровень значимости в %

    :return: Выводит график
    """

    df = pd.DataFrame(full_data[category_feature].value_counts(True) * 100).reset_index()
    df.columns = [category_feature, '%']  # переименовываю столбцы

    col = df.columns

    fig, ax = plt.subplots(figsize=(16, 8))
    plt.xticks(df.index, df[category_feature], rotation=90)

    ax.bar(df.index, df[col[1]], color='lightgrey')
    ax.axhline(y=level, color='red')
    ax.set_ylabel(f'Category share, {col[1]}')
    ax.set_xlabel(f'Category {category_feature}')
    ax.grid(False)
    plt.show()


def plot_categories_and_targer(full_data, category_feature, TARGET_FEATURE, func='mean', level=5):
    """
    Столбчатая диаграмма для категориального признака со средним (медиана, СКО)
    значением целевого признака для каждой категории

    :param full_data: Датасет
    :param category_feature: Имя категориального признака
    :param TARGET_FEATURE: Имя целевого признака
    :param func: Функция по которой проводить аггрегацию. mean, median, std
    :param level: Уровень отсечения доли признака. По умолчанию 5%.
    Отображается на графике горизонтальной красной линией
    :return: Выводит график на экран
    """

    df = calculate_target_feature_per_category(full_data, category_feature, TARGET_FEATURE, func)

    col = df.columns

    fig, ax = plt.subplots(figsize=(16, 8))
    plt.xticks(df.index, df[category_feature], rotation=90)

    ax2 = ax.twinx()
    ax.bar(df.index, df[col[1]], color='lightgrey')
    ax2.plot(df.index, df[col[2]], color='green', label='Seconds')
    ax.axhline(y=level, color='red')
    ax.set_ylabel(f'Category share, {col[1]}')
    ax.set_xlabel(f'Category {category_feature}')
    ax2.set_ylabel(f'{func.capitalize()} {TARGET_FEATURE} per category')
    ax.grid(False)
    ax2.grid(False)
    plt.show()


def group_rare_labels(full_data, category_feature, level=5):
    """
    Функция для снижения количества категорий в указанном признаке

    :param full_data: Датасет
    :param category_feature: Имя категориального признака
    :param level: Уровень отсечения доли признака. По умолчанию 5%.
    Все что ниже заменяется на RARE.
    :return: Возвращает измененный датасет
    """

    # Получаю частоту повторения категорий в датасете в %
    category_series = full_data[category_feature].value_counts(True) * 100

    # Оставляю лишь те категории, частота которых выше level
    category_series = category_series[category_series > level]
    # Получаю список категорий
    category_list = list(category_series.index)

    full_data[category_feature] = full_data[category_feature].apply(lambda x: x if x in category_list else 'RARE')

    return full_data


def target_feature_boxplot_per_category(full_data, target_feature, category_feature):
    """
    Коробка с усами для целевого признкака по указанному категориальному признаку
    
    :param full_data: Объединенный датасет
    :param target_feature: Целевой признак
    :param category_feature: Категориальный признак
    :return: 
    """
    
    current_train, _ = split_full_to_train_and_test(full_data, target_feature)

    fig = plt.figure(figsize=(18, 14))
    sns.boxplot(data=current_train, y=target_feature, x=category_feature)
    plt.title(target_feature + ' per ' + category_feature)
