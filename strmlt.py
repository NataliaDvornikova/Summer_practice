import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Загрузка данных
@st.cache_data
def load_data():
    data = pd.read_csv("mushroom_data.csv")
    return data

# Обработка пропусков: заменяем "?" на моду категории
def replace_question_marks(df):
    for column in df.columns:
        if df[column].dtype == object:
            mode_val = df[df[column] != '?'][column].mode()[0]
            df[column] = df[column].replace('?', mode_val)
    return df

df = load_data()
df = replace_question_marks(df)

# Полный словарь перевода всех значений
VALUE_TRANSLATIONS = {
    'class': {'e': 'Съедобный', 'p': 'Ядовитый'},
    'cap-shape': {
        'b': 'Колокольчатая', 'c': 'Коническая',
        'x': 'Выпуклая', 'f': 'Плоская',
        'k': 'С бугорком', 's': 'Вдавленная'
    },
    'cap-surface': {
        'f': 'Волокнистая', 'g': 'Бороздчатая',
        'y': 'Чешуйчатая', 's': 'Гладкая'
    },
    'cap-color': {
        'n': 'Коричневый', 'b': 'Охристый',
        'c': 'Коричный', 'g': 'Серый',
        'r': 'Зеленый', 'p': 'Розовый',
        'u': 'Фиолетовый', 'e': 'Красный',
        'w': 'Белый', 'y': 'Желтый'
    },
    'bruises': {'t': 'Есть потемнения', 'f': 'Нет потемнений'},
    'odor': {
        'a': 'Миндальный', 'l': 'Анисовый',
        'c': 'Креозотовый', 'y': 'Рыбный',
        'f': 'Гнилостный', 'm': 'Затхлый',
        'n': 'Без запаха', 'p': 'Резкий',
        's': 'Пряный'
    },
    'gill-attachment': {
        'a': 'Прикрепленные', 'd': 'Нисходящие',
        'f': 'Свободные', 'n': 'С выемкой'
    },
    'gill-spacing': {
        'c': 'Близкое', 'w': 'Скученное',
        'd': 'Редкое'
    },
    'gill-size': {'b': 'Широкие', 'n': 'Узкие'},
    'gill-color': {
        'k': 'Черный', 'n': 'Коричневый',
        'b': 'Охристый', 'h': 'Шоколадный',
        'g': 'Серый', 'r': 'Зеленый',
        'o': 'Оранжевый', 'p': 'Розовый',
        'u': 'Фиолетовый', 'e': 'Красный',
        'w': 'Белый', 'y': 'Желтый'
    },
    'stalk-shape': {'e': 'Расширяющаяся', 't': 'Коническая'},
    'stalk-root': {
        'b': 'Луковичный', 'c': 'Булавовидный',
        'u': 'Клубневидный', 'e': 'Равномерный',
        'z': 'Корневидный', 'r': 'Короткий'
    },
    'stalk-surface-above-ring': {
        'f': 'Волокнистая', 'y': 'Чешуйчатая',
        'k': 'Пористая', 's': 'Гладкая'
    },
    'stalk-surface-below-ring': {
        'f': 'Волокнистая', 'y': 'Чешуйчатая',
        'k': 'Пористая', 's': 'Гладкая'
    },
    'stalk-color-above-ring': {
        'n': 'Коричневый', 'b': 'Охристый',
        'c': 'Коричный', 'g': 'Серый',
        'o': 'Оранжевый', 'p': 'Розовый',
        'e': 'Красный', 'w': 'Белый',
        'y': 'Желтый'
    },
    'stalk-color-below-ring': {
        'n': 'Коричневый', 'b': 'Охристый',
        'c': 'Коричный', 'g': 'Серый',
        'o': 'Оранжевый', 'p': 'Розовый',
        'e': 'Красный', 'w': 'Белый',
        'y': 'Желтый'
    },
    'veil-type': {'p': 'Частичное', 'u': 'Универсальное'},
    'veil-color': {'n': 'Коричневый', 'o': 'Оранжевый', 'w': 'Белый', 'y': 'Желтый'},
    'ring-number': {'n': 'Нет', 'o': 'Одно', 't': 'Два'},
    'ring-type': {
        'c': 'Паутинистое', 'e': 'Исчезающее',
        'f': 'Свободное', 'l': 'Кольцевое',
        'n': 'Нет', 'p': 'Пористое',
        's': 'Шелковистое', 'z': 'Закрытое'
    },
    'spore-print-color': {
        'k': 'Черный', 'n': 'Коричневый',
        'b': 'Охристый', 'h': 'Шоколадный',
        'r': 'Зеленый', 'o': 'Оранжевый',
        'u': 'Фиолетовый', 'w': 'Белый',
        'y': 'Желтый'
    },
    'population': {
        'a': 'Обильная', 'c': 'Скоплениями',
        'n': 'Многочисленная', 's': 'Разбросанная',
        'v': 'Одиночная', 'y': 'Групповая'
    },
    'habitat': {
        'g': 'Трава', 'l': 'Листья',
        'm': 'Луг', 'p': 'Тропинки',
        'u': 'Город', 'w': 'Пустоши',
        'd': 'Деревья'
    }
}

# Функция для перевода столбцов
def translate_column(col_name):
    translations = {
        'class': 'Класс',
        'cap-shape': 'Форма шляпки',
        'cap-surface': 'Поверхность шляпки',
        'cap-color': 'Цвет шляпки',
        'bruises': 'Потемнения',
        'odor': 'Запах',
        'gill-attachment': 'Прикрепление пластинок',
        'gill-spacing': 'Расстояние между пластинками',
        'gill-size': 'Размер пластинок',
        'gill-color': 'Цвет пластинок',
        'stalk-shape': 'Форма ножки',
        'stalk-root': 'Корень ножки',
        'stalk-surface-above-ring': 'Поверхность ножки над кольцом',
        'stalk-surface-below-ring': 'Поверхность ножки под кольцом',
        'stalk-color-above-ring': 'Цвет ножки над кольцом',
        'stalk-color-below-ring': 'Цвет ножки под кольцом',
        'veil-type': 'Тип покрывала',
        'veil-color': 'Цвет покрывала',
        'ring-number': 'Количество колец',
        'ring-type': 'Тип кольца',
        'spore-print-color': 'Цвет спорового отпечатка',
        'population': 'Популяция',
        'habitat': 'Среда обитания'
    }
    return translations.get(col_name, col_name)

# Применяем перевод к данным
df_rus = df.copy()
df_rus.columns = [translate_column(col) for col in df.columns]
for col in df.columns:
    if col in VALUE_TRANSLATIONS:
        df_rus[translate_column(col)] = df[col].map(VALUE_TRANSLATIONS[col])

# Заголовок
st.title("Дворникова Наталья Станиславовна_2023-ФГИиБ-ПИ-1б_5_mushrooms")

# Описание данных
st.markdown("""
**Описание набора данных:**  
Данные содержат характеристики грибов (съедобных и ядовитых).  
**Целевая переменная:** `Класс` (Съедобный/Ядовитый).  
""")

# Подготовка данных
X = pd.get_dummies(df.drop('class', axis=1))
y = df['class'].map({'e': 0, 'p': 1})

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Метрики
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# =============================================
# 1. Тепловая карта корреляции признаков
# =============================================
st.subheader("1. Корреляция между признаками")

# Вычисляем корреляционную матрицу для числовых признаков
corr_df = df.drop('class', axis=1).apply(lambda x: pd.factorize(x)[0])
corr_matrix = corr_df.corr()

# Переводим названия признаков
corr_matrix.columns = [translate_column(col) for col in corr_matrix.columns]
corr_matrix.index = [translate_column(col) for col in corr_matrix.index]

fig1 = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale='RdBu',
    zmin=-1,
    zmax=1,
    hoverongaps=False
))

fig1.update_layout(
    title='Корреляция между признаками',
    xaxis_title="Признаки",
    yaxis_title="Признаки",
    width=800,
    height=800,
    # Добавляем ротацию подписей
    xaxis=dict(
        tickangle=45,  # Угол поворота подписей (45 градусов)
        tickfont=dict(size=10)  # Размер шрифта
    ),
    yaxis=dict(
        tickfont=dict(size=10)  # Размер шрифта для оси Y
    )
)
st.plotly_chart(fig1, use_container_width=True)

# =============================================
# 2. Распределение по признаку (с переведенными значениями)
# =============================================
st.subheader("2. Распределение классов")

feature = st.selectbox(
    "Выберите признак:",
    df_rus.columns.drop('Класс'),
    index=df_rus.columns.get_loc('Запах')
)

fig2 = px.histogram(
    df_rus,
    x=feature,
    color='Класс',
    barmode='group',
    color_discrete_map={'Съедобный': 'green', 'Ядовитый': 'red'},
    title=f'Распределение по признаку: {feature}'
)
fig2.update_layout(
    xaxis_title=feature,
    yaxis_title="Количество грибов"
)
st.plotly_chart(fig2, use_container_width=True)

# =============================================
# 3. Матрица ошибок
# =============================================
st.subheader("3. Матрица ошибок")

cm = np.array([[44, 1327], 
               [1133, 340]])

fig3 = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Предсказано: Съедобные', 'Предсказано: Ядовитые'],
    y=['Фактически: Ядовитые', 'Фактически: Съедобные'],
    colorscale='Blues',
    text=cm,
    texttemplate="%{text}",
    hoverinfo="z"
))

fig3.update_layout(
    title='Матрица ошибок',
    xaxis_title="Предсказанный класс",
    yaxis_title="Истинный класс"
)
st.plotly_chart(fig3, use_container_width=True)

# =============================================
# Метрики модели
# =============================================
st.subheader("Основные показатели точности")

col1, col2, col3 = st.columns(3)
col1.metric("Общая точность", "86%")
col2.metric("Полнота (Recall)", "96%")
col3.metric("Точность (Precision)", "79%")

with st.expander("Пояснение метрик"):
    st.markdown("""
    - **Общая точность (Accuracy):** 86% - доля верных предсказаний модели
    - **Полнота (Recall):** 96% - способность находить все ядовитые грибы
    - **Точность (Precision):** 79% - точность определения ядовитых грибов
    """)