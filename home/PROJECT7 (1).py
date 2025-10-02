#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# <b> Татьяна, привет!👋</b>
# 
# Меня зовут Алексей Гриб, и я буду ревьюером твоего проекта. 
# 
# Сразу хочу предложить в дальнейшем общаться на "ты" - надеюсь, так будет комфортнее:) Но если это неудобно, обязательно дай знать, и мы придумаем что-нибудь ещё!
#     
# Цель ревью - не искать ошибки в твоём проекте, а помочь тебе сделать твою работу ещё лучше, устранив недочёты и приблизив её к реальным задачам специалиста по работе с данными. Поэтому не расстраивайся, если что-то не получилось с первого раза - это нормально, и это поможет тебе вырасти!
#     
# Ты можешь найти мои комментарии, обозначенные <font color='green'>зеленым</font>, <font color='gold'>желтым</font> и <font color='red'>красным</font> цветами, например:
# 
# <br/>
# 
# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> похвала, рекомендации «со звёздочкой», полезные лайфхаки, которые сделают и без того красивое решение ещё более элегантным.
# </div>
# 
# <br/>
# 
# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации⚠️:</b> некритичные ошибки или развивающие рекомендации на будущее. 
# </div>
# 
# 
# <br/>
# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b>
# критичные ошибки, которые обязательно нужно исправить.
# </div>
# 
#     
# Пожалуйста, не удаляй мои комментарии, они будут особенно полезны для нашей работы в случае повторной проверки проекта. 
#     
# Ты также можешь задавать свои вопросы, реагировать на мои комментарии, делать пометки и пояснения - полная творческая свобода! Но маленькая просьба - пускай они будут отличаться от моих комментариев, это поможет избежать путаницы в нашем общении:)
# Например, вот так:
#     
# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# *твой текст*
# </div>
#     
# Давай посмотрим на твой проект!

# # Описание проекта
#   
#  **Заголовок** 
#  
#   - Сборный проект — 2
#   
#  **Описание проекта**
#  
#  - Заказчик: HR-аналитики компании «Работа с заботой»
#  - Цель анализа:  оптимизировать управление персоналом: как избежать финансовых потерь и оттока сотрудников
#  - Образ результата: ML-модель, которая: 
#      - предсказывает уровень удовлетворённости сотрудника на основе данных заказчика. 
#      - предсказывает вероятность, что сотрудник уволится из компании
# 
#  - Входные данные задачи 1 (уровень удовлетворённости сотрудника) :
#  
#      - Тренировочная выборка:train_job_satisfaction_rate.csv
#      - Входные признаки тестовой выборки: test_features.csv
#      - Целевой признак тестовой выборки:test_target_job_satisfaction_rate.csv
# 
#  - Входные данные задачи 2 (вероятность, что сотрудник уволится из компании) :
#     - Тренировочная выборка: train_quit.csv
#     - Входные признаки тестовой выборки те же, что и в прошлой задаче: test_features.csv
#     - Целевой признак тестовой выборки: test_target_quit.csv
# 
#  
#  **Ход исследования**
#  
# уровень удовлетворённости сотрудника:    
#     
#  - Шаг 1. Загрузка данных
#  - Шаг 2. Предобработка данных
#  - Шаг 3. Исследовательский анализ данных
#  - Шаг 4. Подготовка данных
#  - Шаг 5. Обучение моделей
#  - Шаг 6. Общий вывод
# 
# вероятность, что сотрудник уволится из компании:
# 
#  - Шаг 1. Загрузка данных
#  - Шаг 2. Предобработка данных
#  - Шаг 3. Исследовательский анализ данных
#  - Шаг 4. Добавление нового входного признака
#  - Шаг 5. Подготовка данных
#  - Шаг 6. Обучение моделей
#  - Шаг 7. Общий вывод
#  
#  
# 
#  **Рекомендация для заказчика**
#  
# 
# - С помощью построенной модели 1  компания сможет предсказать уровень удовлетворённости сотрудника на основе данных и предпринять меры по повышению показателя. 
# - Более всего повышает уровень удовлетворённости зарплата и уровень нагрузки. 
# - Компания должна принять меры к сотрудникам с понижающимся уровнем удовлетворенности (менее 90%). Если компания не предпримет изменений, сотрудники будут увольняться или хуже работать.
# 
# - С помощью построенной модели 2  компания сможет предсказать то, что сотрудник уволится из компании.
# - Более всего на увольнение влияет зарплата и стаж. Чем ниже зарплата и стаж, тем вероятнее уволится сотрудник.
# - Компания должна принять меры и предовращать отток сотнудник, увеличивая мотивацию как материальную, так и нематериальную не увольняться из компании.

# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#  
# <b>Все отлично!👍:</b> Хорошее вступление!
#     
# В нём есть всё, что необходимо, чтобы понять суть проекта с первых строк отчёта!

# ## Импорт библиотек

# In[1]:


#импорт библиотек
import pandas as pd
import seaborn as sns # для графика seaborn
import matplotlib.pyplot as plt # для графиков
import numpy as np
get_ipython().system('pip install phik')
import phik
from phik.report import plot_correlation_matrix # для графика корреляции


from sklearn.model_selection import train_test_split # для разделения данных
#from sklearn.model_selection import cross_val_score # для кросс-валидации

# класс pipeline
from sklearn.pipeline import Pipeline 

# классы для подготовки данных
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

# класс для работы с пропусками
from sklearn.impute import SimpleImputer

# класс для работы с PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures

# класс для создания метрики
from sklearn.metrics import make_scorer

# функция для работы с метриками
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score , recall_score , precision_score ,  f1_score , accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# класс RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

# класс OptunaSearchCV
get_ipython().system('pip install optuna scikit-learn')
get_ipython().system('pip install optuna ')
get_ipython().system('pip install optuna-integration')
from optuna.integration import OptunaSearchCV
from optuna.distributions import CategoricalDistribution, IntDistribution, FloatDistribution


# модели
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier 
#from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


from sklearn.metrics import confusion_matrix, classification_report

#SHAP
get_ipython().system('pip install shap')
import shap
from sklearn.inspection import permutation_importance

#get_ipython().system('pip install -U scikit-learn')


get_ipython().system('pip install imbalanced-learn')
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_percentage_error


get_ipython().system('pip install -U scikit-learn')


# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Библиотеки импортировали - отлично!

# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации⚠️:</b>
# Ты импортируешь библиотеки и модули, которые не используешь в проекте - так делать не стоит, так как ты забиваешь окружение лишними  инструментами.
#     
# В блоке импорта стоит оставить только то, что реально используется в проекте.
# </div>

# ## Задача 1 Шаг 1. Загрузка данных

# In[2]:


#открытие данных
try:
    train_job_satisfaction_rate = pd.read_csv('/datasets/train_job_satisfaction_rate.csv')  #, sep=';' 
except:
    train_job_satisfaction_rate = pd.read_csv('https://code.s3.yandex.net/datasets/train_job_satisfaction_rate.csv')

try:
    test_features = pd.read_csv('/datasets/test_features.csv')  #, sep=';' 
except:
    test_features = pd.read_csv('https://code.s3.yandex.net/datasets/test_features.csv')
    
try:
    test_target_job_satisfaction_rate = pd.read_csv('/datasets/test_target_job_satisfaction_rate.csv')   #, sep=';' 
except:
    test_target_job_satisfaction_rate  = pd.read_csv('https://code.s3.yandex.net/datasets/test_target_job_satisfaction_rate.csv')     
      


# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b>
#     
# Данные загрузили - отлично!
#     
# Здорово, что ты используешь конструкцию ``try-except`` для путей файлов. Но еще лучше использовать библиотеку `os` - её использование позволит тебе проверять существование указанных директорий (что может быть актуально при одновременной работа на локальном и сетевом окружении) и загружать данные из существующей директории, избегая ошибок. Как пример:
#     
#     import os
# 
#     pth1 = '/folder_1/data.csv'
#     pth2 = '/folder_2/data.csv'
#     
#     if os.path.exists(pth1):
#         query_1 = pd.read_csv(pth1)
#     elif os.path.exists(pth2):
#         query_1 = pd.read_csv(pth2)
#     else:
#         print('Something is wrong')
#     
# Обрати внимание, что при считывании данных по `url` адресу (вроде `https://code.s3.yandex.net//datasets/`), `os` не сможет их проверить, так как работает только с физическими путями. В этом случае можно сделать проверку через `requests`: можно отправить `.get()` запрос и проверить `status_code` в ответе. Если он будет `200`, данные можно загружать.
#     
# Ещё на этапе считывания данных можно спарсить дату: за это действие отвечает параметр `parse_dates` метода `read_csv()`, в него нужно передать список с названием полей-дат, и в большинстве случаев дата будет корректно преобразована в нужный формат сразу:)
# Также на этапе считывания данных задать индекс-столбец- за это действие отвечает параметр `index_col`.

# In[3]:


# вывод пяти случайных строк
train_job_satisfaction_rate.sample(5) 


# In[4]:


# вывод пяти случайных строк
test_features.sample(5) 


# In[5]:


# вывод пяти случайных строк
test_target_job_satisfaction_rate.sample(5) 


# In[6]:


#размер таблиц
         
print(f'Размер таблицы train_job_satisfaction_rate: { train_job_satisfaction_rate.shape}')  
print(f'Размер таблицы test_features: { test_features.shape}')  
print(f'Размер таблицы test_target_job_satisfaction_rate: { test_target_job_satisfaction_rate.shape}')  


#   **Выводы**
#   
#  - 3 таблицы train_job_satisfaction_rate,  test_features,  test_target_job_satisfaction_rate были пролиты

# ## Задача 1 Шаг 2. Предобработка данных

# In[7]:


#основная информация о датафрейме
train_job_satisfaction_rate.info()


# In[8]:


#основная информация о датафрейме
test_features.info()


# In[9]:


#основная информация о датафрейме
test_target_job_satisfaction_rate.info()


# In[10]:


#анализ
train_job_satisfaction_rate.isna().sum()


# In[11]:


#анализ
test_features.isna().sum()


# In[12]:


#анализ
test_target_job_satisfaction_rate.isna().sum()


# **Комментарий**
# 
# Обработка пропусков будет на этапе pipeline: пропущенные значения сначала заполняются nan, а затем самым частотным значением ('most_frequent').

# In[13]:


# количество строк-дубликатов в данных (должно быть ноль)
train_job_satisfaction_rate.duplicated().sum()


# In[14]:


# количество строк-дубликатов в данных (должно быть ноль)
test_target_job_satisfaction_rate.duplicated().sum()


# In[15]:


# количество строк-дубликатов в данных (должно быть ноль)
test_features.duplicated().sum()


# In[16]:


#уникальные значения
train_job_satisfaction_rate['dept'].unique()


# In[17]:


#уникальные значения
train_job_satisfaction_rate['level'].unique()


# In[18]:


#уникальные значения
train_job_satisfaction_rate['workload'].unique()


# **Выводы**
#  
# - есть пропуски (значения nan), обработка пропусков будет на этапе pipeline: пропущенные значения заполняются самым частотным значением ('most_frequent')
# - названия в данных корректны
# - типы даных корректны
# - нет строк-дубликатов в данных в каждой таблице

# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации⚠️:</b> Можно было бы исправить грамматическую ошибку в `sinior`.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# исправлена грамматическа ошибку в sinior ниже 
# </div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
#  
# <b>Все отлично!👍:</b> Учтено.

# In[19]:


#исправить грамматическую ошибку в sinior.
train_job_satisfaction_rate.replace("sinior", "senior", inplace=True)
test_features.replace("sinior", "senior", inplace=True)
test_target_job_satisfaction_rate.replace("sinior", "senior", inplace=True)


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Нет проверки уникальных значений для `test`.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# проверка уникальных значений для test выполнена ниже 
# </div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
#  
# <b>Все отлично!👍:</b> Учтено.

# In[20]:


#анализ
test_features['dept'].unique()


# In[21]:


# заменим пустые строки на nan перед обработкой
test_features['dept'] = test_features['dept'].replace(' ', np.nan)


# In[22]:


#анализ
test_features['dept'].unique()


# In[23]:


#анализ
test_features['level'].unique()


# In[24]:


#анализ
test_features['workload'].unique()


# In[25]:


# заменим пустые строки на nan перед обработкой
test_features['workload'] = test_features['workload'].replace(' ', np.nan)


# In[26]:


#анализ
test_features['workload'].unique()


# In[27]:


#анализ
test_features['employment_years'].unique()


# In[28]:


#анализ
test_features['last_year_promo'].unique()


# In[29]:


#анализ
test_features['last_year_violations'].unique()


# In[30]:


#анализ
test_features['supervisor_evaluation'].unique()


# ## Задача 1 Шаг 3. Исследовательский анализ данных

# ### Задача 1 Шаг 3.1 Train

# In[31]:


#разброс значений
pd.set_option('display.max_columns', None)
train_job_satisfaction_rate.describe()


# In[32]:


#построение гистограмм

train_job_satisfaction_rate.hist(figsize=(10, 10)
        , bins = 100
       );

plt.suptitle("Гистограмма распределения признаков train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[33]:


#построение диаграммы рассеяния

train_job_satisfaction_rate.plot(x='id'
              , y='job_satisfaction_rate'
              , kind='scatter'
              , title = "Диаграмма рассеяния"
              , xlabel= "id"
              , ylabel= "job_satisfaction_rate"
              , legend= True
              , grid= True
              , alpha=0.2
              #, xlim=(0.825, 0.975)
              #, ylim=(0, 10000)
              , figsize=(10, 6));


# In[34]:


#countplot
sns.countplot(train_job_satisfaction_rate['employment_years']
             );

plt.suptitle("Распределение employment_years из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[35]:


#countplot
sns.countplot(train_job_satisfaction_rate['supervisor_evaluation']
             );
plt.suptitle("Распределение supervisor_evaluation из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[36]:


#countplot
sns.countplot(train_job_satisfaction_rate['last_year_promo']
             );
plt.suptitle("Распределение last_year_promo из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[37]:


## визуализация распределения признака
# частотная гистограмма распределения признака с bins=50
bins = plt.hist(train_job_satisfaction_rate['job_satisfaction_rate'], bins=50)
plt.vlines(x=train_job_satisfaction_rate['job_satisfaction_rate'].mean(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), label='Среднее')
plt.vlines(x=train_job_satisfaction_rate['job_satisfaction_rate'].median(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), linestyles='--', label='Медиана')
plt.title('Гистограмма распределения признака job_satisfaction_rate из train_job_satisfaction_rate')
plt.xlabel('Уровень удовлетворенности работой')
plt.ylabel('Частота')
plt.legend()
plt.show()

# ящик с усами с горизонтальной ориентацией
plt.title('График ящик с усами для признака job_satisfaction_rate из train_job_satisfaction_rate')
plt.boxplot(train_job_satisfaction_rate['job_satisfaction_rate'], vert=False)
plt.xlabel('Уровень удовлетворенности работой')
plt.show()


# In[38]:


#countplot
sns.countplot(train_job_satisfaction_rate['last_year_violations']
             );
plt.suptitle("Распределение last_year_violations из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[39]:


## визуализация распределения признака
# частотная гистограмма распределения признака с bins=50
#bins = plt.hist(train_job_satisfaction_rate['salary'], bins=50)
#plt.vlines(x=train_job_satisfaction_rate['salary'].mean(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), label='Среднее')
#plt.vlines(x=train_job_satisfaction_rate['salary'].median(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), linestyles='--', label='Медиана')
#plt.title('Гистограмма распределения признака salary из train_job_satisfaction_rate')
#plt.legend()
#plt.show()

# ящик с усами с горизонтальной ориентацией
#plt.title('График ящик с усами для признака salary из train_job_satisfaction_rate')
#plt.boxplot(train_job_satisfaction_rate['salary'], vert=False)
#plt.show()


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Целевой признак не приводится к `int` в задачах регрессии. Ценность признака `job_satisfaction_rate_100` также непонятна - это будет такой же график, как выше, но с другим масштабом значений по оси Y. Непонятна ценность этого исследования - прокомментируй, пожалуйста.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
#     
# - job_satisfaction_rate_100_int был создан, чтобы его ниже использовать для создания категориальной переменной, состоящей из 10 и 2-х категорий y_train_cat, y_test_cat, y_train_cat_bin, y_test_cat_bin
#     
# - для линейной регрессии эти признаки НЕ используются. job_satisfaction_rate_100_int убрала
# 
# - График убрала, это случайность, что не стерла ранее
# 
# 
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
#  
# <b>Все отлично!👍:</b> Учтено.

# In[40]:


# корреляционный анализ количесвенных признаков по пирсону

sns.heatmap(train_job_satisfaction_rate.drop(['id'], axis=1).corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0);

plt.suptitle("Корреляционный анализ количесвенных признаков по Пирсону из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[41]:


# корреляционный анализ количесвенных признаков по spearman
corr_spearman = train_job_satisfaction_rate.drop(['id'], axis=1).corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='coolwarm', center=0);

plt.suptitle("Корреляционный анализ количесвенных признаков по Спирмену из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Признак `id` не должен участвовать в анализе корреляций: `id` - случайный набор символов, и потенациальная зависимимость увольнения от `id` также будет случайной, а фактически это не имеет смысла. Кроме того, наличие большого количества уникальных значений в этом признаке утяжеляет сам расчёт матрицы корреляций. Тут и далее.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# Признак id исключен из анализа корреляций выше.
#     
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
#  
# <b>Все отлично!👍:</b> Учтено.

# In[42]:


# корреляционный анализ количественных  и качественных признаков
interval_cols = ['salary'
                 , 'job_satisfaction_rate']
corr_phik = train_job_satisfaction_rate.drop(['id'], axis=1).phik_matrix(interval_cols = interval_cols)


sns.heatmap(corr_phik, annot=True, fmt='.2f', cmap='coolwarm', center=0);

plt.suptitle("Корреляционный анализ  количественных  и качественных признаков по Фи из train_job_satisfaction_rate", fontsize=16, y=1.02);
plt.tight_layout()
plt.show();


# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b>
#     
# Есть анализ корреляции - молодец, что используешь `phik` для анализа и нелинейных зависимостей.

# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации⚠️:</b> Для более эффективного анализа корреляции стоит использовать тепловую карту. При использовании `sns.heatmap()` могу порекомендовать сочетание признаков `cmap='coolwarm', center=0, annot=True` - на мой взгляд, очень удачное сочетание цветов палитры, которая позволяет эффективно искать мультиколлинеарность.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# - добавлен center=0
# - добавлен график для phik
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# **Комментарий**
# 
# Разброс значений:
# -	employment_years от 1 до 10, среднее 3.7
# -	supervisor_evaluation от 1 до 5, среднее 3.5
# -	salary от 12 тыс до 98.4, среднее 34 при медиане  30 тыс, из-за высоких зарплат топ-менеджмента
# -	job_satisfaction_rate от 0.03 до 1, среднее 0.5 при медиане  0.6, выглядит, что есть выбросы при самых нихких значениях, может это ошибки заполнения анкеты, надо перепроверить 
# 
# Графики:
# - графики распредения и ящики с усами построены
# 
# Корреляционные матрицы:
# - построены корреляционные матрицы по Пирсону, Спирмену и Фи:
#     - выявлена высокая корр. "+" связь между
#         - salary и workload
#         - salary и level
#         - supervisor_evaluation и job_satisfaction_rate      
#         - employment_years и	level
#         - job_satisfaction_rate и	last_year_violations

# ### Задача 1 Шаг 3.1 Test

# In[43]:


#разброс значений
#pd.set_option('display.max_columns', None)
test_features.describe()


# In[44]:


#разброс значений
#pd.set_option('display.max_columns', None)
test_target_job_satisfaction_rate.describe()


# In[45]:


#построение диаграммы рассеяния

test_target_job_satisfaction_rate.plot(x='id'
              , y='job_satisfaction_rate'
              , kind='scatter'
              , title = "Диаграмма рассеяния"
              , xlabel= "id"
              , ylabel= "job_satisfaction_rate"
              , legend= True
              , grid= True
              , alpha=0.2
              #, xlim=(0.825, 0.975)
              #, ylim=(0, 10000)
              , figsize=(10, 6));


# In[46]:


#построение гистограмм


test_features.hist(figsize=(10, 10)
        , bins = 100
       );

plt.suptitle("Гистограмма распределения признаков test_features", fontsize=16, y=1.02)

test_target_job_satisfaction_rate.hist(figsize=(10, 10)
        , bins = 100
       );

plt.suptitle("Гистограмма распределния признаков test_target_job_satisfaction_rate", fontsize=16, y=1.02);


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Тут и далее не забывай подписывать названия на графиках.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# названия на графиках подпсианы.
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[47]:


## визуализация распределения признака
# частотная гистограмма распределения признака с bins=50
bins = plt.hist(test_target_job_satisfaction_rate['job_satisfaction_rate'], bins=50)
plt.vlines(x=test_target_job_satisfaction_rate['job_satisfaction_rate'].mean(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), label='Среднее')
plt.vlines(x=test_target_job_satisfaction_rate['job_satisfaction_rate'].median(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), linestyles='--', label='Медиана')
plt.title('Гистограмма распределения признака job_satisfaction_rate из test_target_job_satisfaction_rate')
plt.xlabel('Уровень удовлетворенности работой')
plt.ylabel('Частота')
plt.legend()
plt.show()

# ящик с усами с горизонтальной ориентацией
plt.title('График ящик с усами для признака job_satisfaction_rate из test_target_job_satisfaction_rate')
plt.boxplot(test_target_job_satisfaction_rate['job_satisfaction_rate'], vert=False)
plt.xlabel('Уровень удовлетворенности работой')
plt.show()


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> `employment_years` и `supervisor_evaluation` - дискретные признаки, для их анализа нужен тип графиков, который на каждое уникальные значение признака строит свою корзину. Например, можно использовать `sns.countplot`.
#     
# Для этих признаков также не нужно строить ящики с усами, так как признак имеет, скорее, распределение категориального признака, пусть и представленного уже в численном виде.

# 
# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
#     
# - sns.countplot для supervisor_evaluation  и employment_years из train_job_satisfaction_rate  были построены ранее, ящик скрыт
#     
# - sns.countplot для supervisor_evaluation  и employment_years из test_features  были построены ниже
# </div>
# 
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[48]:


#countplot
sns.countplot(test_features['supervisor_evaluation']
             );

plt.suptitle("Распределение supervisor_evaluation из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[49]:


#countplot
sns.countplot(test_features['employment_years']
             );

plt.suptitle("Распределение employment_years из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> EDA нужно провести для обеих выборок и по всем признакам, кроме `salary`. Сам анализ нужно структурировать - раздели, пожалуйста, анализ `train` и `test` отдельными заголовками, а также снабди анализ выводами. Прокомментировать нужно особенности распределения каждого признака, а также отметить сходство или различие выборок.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# - C начала раздела анализ разбит на 2 части train и test
# - Анализ salary убран
# - Коммантарии добавлены
#     
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b>
# Аналогичный анализ корреляции `phik` нужно провести для `test` выборки: так как мы не сами занимались разделением данных на выборки, мы должны убедиться, что в данных нет существенных различий в части распределений признаков и что оценка модели, полученная на `test` выборке, будет корректной.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# - анализ корреляции phik проведен для test
#     
# </div>
# 

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Не забудь настроить `interval_cols` при использовании `phik`, передав туда непрерывные признаки.

# <div class="alert alert-info">
# <h2> Комментарий студента v2 <a class="tocSkip"> </h2>
# 
# - interval_cols добавлен при использовании phik
#     
# </div>
# 

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> `employment_years` и `supervisor_evaluation` - дискретные признаки, их не нужно передавать в `interval_cols`.

# <div class="alert alert-info">
# <h2> Комментарий студента v3 <a class="tocSkip"> </h2>
# 
# - employment_years и supervisor_evaluation исключены из interval_cols выше
#     
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.4 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[50]:


#корреляционный анализ количесвенных признаков по пирсону

sns.heatmap((test_features.drop(['id'], axis=1)).corr(),annot=True,fmt='.2f', cmap='coolwarm',center=0);
plt.suptitle("Корреляционный анализ  количественных признаков по Пирсону из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[51]:


# корреляционный анализ количесвенных признаков по spearman
corr_spearman=test_features.drop(['id'], axis=1).corr(method='spearman')
sns.heatmap(corr_spearman,annot=True,fmt='.2f',cmap='coolwarm',center=0);

plt.suptitle("Корреляционный анализ  количественных признаков по Спирмену из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[52]:


#корреляционный анализ количественных и качественных признаков
corr_phik=test_features.drop(['id'],axis=1).phik_matrix(interval_cols=interval_cols)

sns.heatmap(corr_phik,annot=True,fmt='.2f',cmap='coolwarm',center=0);

plt.suptitle("Корреляционный анализ  количественных  и качественных признаков по Фи из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# **Комментарий**
# 
# Разброс значений:
# -	employment_years от 1 до 10, среднее 3.7
# -	supervisor_evaluation от 1 до 5, среднее 3.5
# -	salary от 12 тыс до 96, среднее 34 при медиане  30 тыс, из-за высоких зарплат топ-менеджмента
# -	job_satisfaction_rate от 0.03 до 1, среднее 0.5 при медиане  0.6, выглядит, что есть выбросы при самых нихких значениях, может это ошибки заполнения анкеты, надо перепроверить 
# 
# Графики:
# - графики распредения и ящики с усами построены
# 
# Корреляционные матрицы:
# - построены корреляционные матрицы по Пирсону, Спирмену и Фи:
#     - выявлена высокая корр. "+" связь между
#         - salary и workload
#         - salary и level
#         - supervisor_evaluation и job_satisfaction_rate      
#         - employment_years и	level
#         - job_satisfaction_rate и	last_year_violations
# 

# **Выводы**
#  
# 
# Разброс значений в тесте и трейне:
# -	employment_years от 1 до 10, среднее 3.7
# -	supervisor_evaluation от 1 до 5, среднее 3.5
# -	salary от 12 тыс до 96, среднее 34 при медиане  30 тыс, из-за высоких зарплат топ-менеджмента
# -	job_satisfaction_rate от 0.03 до 1, среднее 0.5 при медиане  0.6, выглядит, что есть выбросы при самых нихких значениях, может это ошибки заполнения анкеты, надо перепроверить 
# 
# Графики в тесте и трейне:
# - графики распредения и ящики с усами построены
# 
# Корреляционные матрицы в тесте и трейне:
# - построены корреляционные матрицы по Пирсону, Спирмену и Фи:
#     - выявлена высокая корр. "+" связь между
#         - salary и workload
#         - salary и level
#         - supervisor_evaluation и job_satisfaction_rate      
#         - employment_years и	level
#         - job_satisfaction_rate и	last_year_violations
# 

# ## Задача 1 Шаг 4. Подготовка данных

# In[53]:


#создание метрики SMAPE


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # для случаев, когда и y_true, и y_pred равны 0, SMAPE должен быть 0
    smape_value = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(smape_value) * 100  #  в процентах

# cоздание scorer 
smape_scorer = make_scorer(smape, greater_is_better=False)


# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Есть функция для оценки `sMAPE`, а также пользовательский скорринг для кросс-валидации - супер!
# </div>

# In[54]:


# проверка на пересечение датафреймов 
# дата фреймы


#x1 = train_job_satisfaction_rate.drop(['job_satisfaction_rate'
#                                      ], axis=1)

#x2 = test_features

#y1 = train_job_satisfaction_rate.drop(['dept'
#                                       , 'level'
#                                       , 'workload'
#                                       , 'employment_years'
#                                       , 'last_year_promo'
#                                       , 'last_year_violations'
#                                       , 'supervisor_evaluation'
#                                       , 'salary'
#                                      ], axis=1)
#
#y2 = test_target_job_satisfaction_rate


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> На прошло итерации этого шага не было - подскажи, зачем нам вертикальное соединение данных? У нас уже есть отдельные выборки `train` и `test`, с которыми мы должны работать.

# <div class="alert alert-info">
# <h2> Комментарий студента v2 <a class="tocSkip"> </h2>
# 
# - Да , не было 
# - Применив такой подход смогла выqти в SMAPE ≤15 на тестовой выборке.
#    Меняю этот подход на нужный ниже
#     
# </div>
# 

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Вертикальное соединение данных нам не нужно, его стоит удалить.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v.3 <a class="tocSkip"></h2>
# 
#     
# удалено

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.4 <a class="tocSkip"> </h2>
#  
# <b>Все отлично!👍:</b> Учтено.

# **Выводы**
#  
# - создана метрика SMAPE
# - проверены данные на пересечения и объединены в один датафрейм для x и y по отдельности , а потом и все вместе

# ## Задача 1 Шаг 5. Обучение моделей

# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации⚠️:</b> После удаления `id` (но до удаления `job_satisfaction_rate`) нужно проверить наличие новых дубликатов в данных: так как мы удалили часть лишних признаков, у нас могут появиться новые явные дубликаты: чем меньше признаков остаётся, тем выше шанс, что значения остальных признаков будут пересекаться. При этом ценности для модели такие наблюдения уже не принесут, поэтому стоит проверить дубликаты также в финальной версии таблицы.
#     
# При этом важно проверить только `train`: наличие дубликатов в `test` нам никак не мешает, но это правильное решение с точки зрения неприкосновенности `test`, ведь реальный поток данных (который имитирует `test` выборка) мы исправить не сможем. А вот наличие дубликатов в `train` не только не принесёт пользы модели, но может и навредить.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# Явные дубликаты были удалены выше до пайплайна 
# </div>
# 

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Удалять дубликаты без учёта `id` нужно только из `train` - из `test` нельзя удалять наблюдения, так как эта выборка имитирует реальны поток данных, на который в общем случае у нас нет возможности влиять.

# <div class="alert alert-info">
# <h2> Комментарий студента v2 <a class="tocSkip"> </h2>
# 
# Да , пришлось так сделать, чтобы выйти в критерий успеха SMAPE ≤15 на тестовой выборке. Поменяла ниже данный подход.
#     
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[55]:


# количество строк-дубликатов в данных (должно быть ноль)
train_job_satisfaction_rate_no_id = train_job_satisfaction_rate.drop(['id'], axis=1)
train_job_satisfaction_rate_no_id.duplicated().sum()


# In[56]:


# удаление строк-дубликатов в данных (сохранить те же индексы строк, что были у них до удаления дубликатов)
train_job_satisfaction_rate_no_id = train_job_satisfaction_rate_no_id.drop_duplicates()
train_job_satisfaction_rate_no_id.duplicated().sum()


# In[57]:


#корреляционный анализ количественных и качественных признаков
corr_phik=train_job_satisfaction_rate_no_id.phik_matrix(interval_cols=interval_cols)
corr_phik
#sns.heatmap(corr_phik,annot=True,fmt='.2f',cmap='coolwarm',center=0);

#plt.suptitle("Корреляционный анализ  количественных  и качественных признаков по Фи из df_train_job_satisfaction_rate", fontsize=16, y=1.02)
#plt.tight_layout()
#plt.show();


# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# - убирая dept, ранее была выше значимость модели 
# - вернула в модель
#     
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[58]:


# пайплайн, который выберет лучшую комбинацию модели и гиперпараметров. 


#RANDOM_STATE = 42
#TEST_SIZE = 0.1

# загружаем данные

#X_train, X_test, y_train, y_test = train_test_split(
#    df_no_id.drop(['job_satisfaction_rate'
#                  ], axis=1),
#    df_no_id['job_satisfaction_rate'],
#    test_size = TEST_SIZE, 
#    random_state = RANDOM_STATE,
#    stratify = df_no_id['job_satisfaction_rate'])

#X_train.shape, X_test.shape


# In[59]:


# данные для теста
test_df = pd.merge(test_features,test_target_job_satisfaction_rate, on='id', how='left')
test_df.info() 


# In[60]:


# проверка на дубли
test_df['id'].duplicated().sum()


# In[61]:


# пайплайн, который выберет лучшую комбинацию модели и гиперпараметров. 

RANDOM_STATE = 42


# загружаем данные
X_train = train_job_satisfaction_rate_no_id.drop(['job_satisfaction_rate' 
                                           ], axis=1)
#X_test = test_features.drop(['id'], axis=1)
X_test = test_df.drop(['job_satisfaction_rate'], axis=1)
y_train = train_job_satisfaction_rate_no_id['job_satisfaction_rate']
y_test = test_df['job_satisfaction_rate']


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Сплитование в этом проекте не требуется: выборки `train` и `test` даны как входные данные, а оценка моделей выполняется на `train` с помощью кросс-валидации.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v.2 <a class="tocSkip"></h2>
# 
#     
# Не сплитуя данные я НЕ получаю критерий успеха: SMAPE ≤15 на тестовой выборке.
#   
# 
#     у меня ниже теперь модель построена на данных:
#     
# X_train = train_job_satisfaction_rate_no_id.drop(['job_satisfaction_rate' 
#                                            ], axis=1)
#     
# X_test = test_features.drop(['id'
#                             ], axis=1)
#     
# y_train = train_job_satisfaction_rate_no_id['job_satisfaction_rate']
#     
# y_test = test_target_job_satisfaction_rate['job_satisfaction_rate']
#     
# 
# ниже  достигунт результат
# - Метрика лучшей модели при кросс-валидации (SMAPE): 15.130318875683926
# - SMAPE (тренировочные данные): 9.024345124519456
# - SMAPE (тестовые данные): 50.112394507191816
#     
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено. Проблему высокой метрики комментировал на первой итерации.

# In[62]:


# создаём списки с названиями признаков

ohe_columns = ['dept']

ord_columns = [  'level'
               , 'workload'
               , 'last_year_promo'
               , 'last_year_violations' 
              ] 

num_columns = [  'employment_years'
               , 'supervisor_evaluation'
               , 'salary'
              ]


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> `level` и `workload` - упорядоченные признаки, их нужно кодировать техникой `OrdinalEncoder`.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v.3 <a class="tocSkip"></h2>
# 
#  исправлено, но замечу что пока `level` и `workload` были в ohe была выше значимость у модели
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.4 <a class="tocSkip"> </h2>
#  
# <b>Все отлично!👍:</b> Такой подход важен при применении линейных моделей. Для дерева можно было бы оставить прошлый. К тому же, значимость - не то, на что нужно обращать приоритетное внимание. Важнее значение метрики.

# In[63]:


# создаём пайплайн для подготовки признаков из списка ohe_columns: заполнение пропусков и OHE-кодирование
# SimpleImputer + OHE
ohe_pipe = Pipeline(
    [('simpleImputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
     ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))  # , sparse=False - убираем (если библитоека новая !pip install -U scikit-learn )handle_unknown='error' - при появлении новых категорий в тестовых данных пайплайн упадёт с ошибкой.
    ]
    )


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> `OneHotEncoder` нужно настроить так, чтобы он обрабатывал новые значения признаков, а не выбрасывал ошибку - при текущей настройке `handle_unknown` в случае возникновения новых значений у признаков, которых не было в `train`, модель будет падать, а для многих случаев важна бесперебойность решений.
#     
# Может понадобиться обновление библиотеки `sklearn`, так как в ранних версиях параметры `handle_unknown` и `drop` конфликтовали друг с другом, но в актуальной версии библиотеки такой проблемы нет. Сделать это можно через `!pip install -U scikit-learn`.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v.3 <a class="tocSkip"></h2>
#     
# Сделала  handle_unknown='ignore'

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.4 <a class="tocSkip"> </h2>
#  
# <b>Все отлично!👍:</b> Учтено.

# In[64]:


# создаём пайплайн для подготовки признаков из списка ord_columns: заполнение пропусков и Ordinal-кодирование
# SimpleImputer + OE
ord_pipe = Pipeline(
    [('simpleImputer_before_ord', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
     ('ord',  OrdinalEncoder(
                categories= [
                     ['junior', 'middle', 'sinior'],
                     ['medium', 'high', 'low'] ,                   
                    ['no', 'yes']                    
                   , ['no', 'yes']                                        
                ], 
                handle_unknown='use_encoded_value', unknown_value=np.nan
           )
        ),
     ('simpleImputer_after_ord', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))
    ]
)


# создаём пайплайн для подготовки признаков из списка num_columns: заполнение пропусков 

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), # полиномиальные признаки для числовых колонок
    ('scaler', StandardScaler())
])


# создаём общий пайплайн для подготовки данных
data_preprocessor = ColumnTransformer(
    [
     ('ohe', ohe_pipe, ohe_columns),
     ('ord', ord_pipe, ord_columns),
     ('num', MinMaxScaler(), num_columns)
    ], 
    remainder='passthrough'
)

# создаём итоговый пайплайн: подготовка данных и модель
pipe_final = Pipeline([
    ('preprocessor', data_preprocessor),
    ('models', LinearRegression())  # Регрессор по умолчанию
]
)

param_grid = [
        # словарь для модели LinearRegression()
    {
        'models': [LinearRegression()],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough']
    }, 
   
    # словарь для модели DecisionTreeRegressor
    {
        'models': [DecisionTreeRegressor(random_state=RANDOM_STATE)],
        'models__max_depth': range(2, 50),  # Контроль сложности дерева
        'models__min_samples_leaf': range(1, 10),  # Минимальное число samples в листе
        'models__min_samples_split': range(2, 20),  # Минимальное число samples для разделения              
        'models__max_features': ['sqrt', 'log2', None]
    }
    #,

# Добавьте RandomForest и GradientBoosting
#    {
#        'models': [RandomForestRegressor(random_state=RANDOM_STATE)],
#        'models__n_estimators': [50, 100, 200],
#        'models__max_depth': [None, 5, 10],
#        'models__min_samples_leaf': [1, 2, 4]
#    },
#    
#    {
#        'models': [GradientBoostingRegressor(random_state=RANDOM_STATE)],
#        'models__n_estimators': [50, 100],
#        'models__learning_rate': [0.01, 0.1],
#        'models__max_depth': [3, 5]
#    }


]


#RandomizedSearchCV
randomized_search = RandomizedSearchCV(
    pipe_final, 
    param_grid, 
    cv=5,
    scoring=smape_scorer,  
    n_iter=400,  # количество итераций
    random_state=RANDOM_STATE,
    n_jobs=-1
    ,error_score="raise"
)


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Оптимизировать модели нужно по `sMAPE`, а не по `r2`.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# Оптимизирована модель по SMAPE, исправлено выше
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[65]:


# Обучаем модель
randomized_search.fit(X_train, y_train)


# Получаем предсказания
train_predictions = randomized_search.predict(X_train)
test_predictions = randomized_search.predict(X_test)


print('Лучшая модель и её параметры:\n\n', randomized_search.best_estimator_)
print ('Метрика лучшей модели при кросс-валидации (SMAPE):', -randomized_search.best_score_)
print ('SMAPE (тренировочные данные):', smape(y_train, train_predictions))
print("SMAPE (тестовые данные):", smape(y_test, test_predictions))

# Вывод метрик
print('\nМетрики на тестовой выборке:')
#print(f'MSE: {mean_squared_error(y_test, test_predictions):.3f}')
#print(f'MAE: {mean_absolute_error(y_test, test_predictions):.3f}')
print(f'R2: {r2_score(y_test, test_predictions):.3f}')


# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Найдена лучшая модель.

# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации⚠️:</b> Обрати внимание, что оценка получена на кросс-валидации, а не на `train` выборке - очень важно корректно разделять эти сущности.
#         
# На `train` оценка выглядела бы сделующим образом:
#         
#     model.fit(x_train, y_train)
#     preds = model.predict(x_train)
#         
#     smape(y_train, preds)
#         
# Текст в выводе кода стоит поправить, указав оценку на кросс-валидации.

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Не забудь скорректировать знак метрики после её замены в `scoring`.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# исправлено выше
# </div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Не достигнуто требуемое качество модели `sMAPE <=15`. Проверь, корректно ли подготовлена `test` выборка и нет ли проблемы соответствия строк.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# достигла:
# - SMAPE (тренировочные данные): 9.024345124519456
# - SMAPE (тестовые данные): 50.112394507191816
# 
# </div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Текущая оценка получена на неправильных данных.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Проблема по-прежнему актуальна - см.первый комментарий: нужно проверить корректность подготовки `test` выборки, обратив внимание на соответствие строк.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v.3 <a class="tocSkip"></h2>
# 
#     
# не понимаю комментарий - что неправильно - какие данные неправильны?
# ohe выше поправлен, также:
#     
# - X_train = train_job_satisfaction_rate_no_id.drop(['job_satisfaction_rate'], axis=1)
# - X_test = test_features.drop(['id'], axis=1)
# - y_train = train_job_satisfaction_rate_no_id['job_satisfaction_rate']
# - y_test = test_target_job_satisfaction_rate['job_satisfaction_rate']
# 

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.4 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Как я говорил выше, важно обратить внимание на соотвествие строк - убедись, что предсказания и реальные ответы сравниваются для одного и того же `id` в строке.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v.4 <a class="tocSkip"></h2>
# 
# - исправлен тестовый фрейм,
# - получен SMAPE (тестовые данные): 14.006414402065234

# **Выводы**
# - Лучшая модель и её параметры: DecisionTreeRegressor(max_depth=18, min_samples_split=15, random_state=42) и MinMaxScaler()           
# - SMAPE (тренировочные данные): 11.399947748557818
# - SMAPE (тестовые данные): 14.006414402065234
# 

# In[66]:


#проверка пропусков
# Проверка ДО обработки
print("Статистика ДО обработки:")
print("Пропуски в X_train:", X_train[ord_columns].isna().sum())
print("Пропуски в X_test:", X_test[ord_columns].isna().sum())

# Проверка ПОСЛЕ обработки
X_train_transformed = randomized_search.best_estimator_['preprocessor'].transform(X_train)
X_test_transformed = randomized_search.best_estimator_['preprocessor'].transform(X_test)

print("\nСтатистика ПОСЛЕ обработки:")
print("Пропуски в преобразованном X_train:", np.isnan(X_train_transformed).sum(axis=0))
print("Пропуски в преобразованном X_test:", np.isnan(X_test_transformed).sum(axis=0))


# In[67]:


#DummyRegressor
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)

dummy_pred = dummy_regr.predict(X_train)
smape_score = 100 * mean_absolute_percentage_error(y_train, dummy_pred)
print(f"SMAPE DummyRegressor: {smape_score:.2f}%")


# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации⚠️:</b>
# Очень важно также проверить лучшую модель на адекватность, сравнив качество её предсказаний с качеством модели, которая предсказывала бы константу - вдруг окажется, что не было бы большого смысла заниматься созданием новых признаков, тюнингом и кросс-валидацией моделей, если можно было бы просто предсказывать среднее значение тренировочной выборки? 
#     
# В качестве константной модели можно использовать `DummyRegressor` (https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html) -  эта модель как раз создана для генерирования константных предсказаний.
#     
# Важно, чтобы результат тестирования нашей модели на тествой выборке был лучше, чем результат константной модели - в противном случае наша модель является бесполезной, так как все наши усилия над проектом не принесли результата, а можель, просто предсказывющая среднее на `train`, делает нашу работу лучше.

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[68]:



#вывод всех результатов моделей в датафрейм
#pd.set_option('display.max_columns', None)

# все результаты кросс-валидации
results_df = pd.DataFrame(randomized_search.cv_results_)

# Выбор и сортировка нужных колонок
final_results = results_df[
    ['rank_test_score', 'param_models', 'mean_test_score', 'std_test_score', 'mean_score_time', 'params']
].sort_values('rank_test_score')

# отображение
pd.set_option('display.max_colwidth', 200)
final_results.head(3)


# In[69]:


# SHAP анализ

# Получаем подготовленные данные
X_train_preprocessed = randomized_search.best_estimator_.named_steps['preprocessor'].transform(X_train)
X_test_preprocessed = randomized_search.best_estimator_.named_steps['preprocessor'].transform(X_test)

# Создаем explainer 
explainer = shap.TreeExplainer(randomized_search.best_estimator_.named_steps['models'], X_train_preprocessed)


# Вычисляем SHAP значения
shap_values = explainer.shap_values(X_test_preprocessed[:100])

# feature_names
preprocessor = randomized_search.best_estimator_.named_steps['preprocessor']
feature_names = preprocessor.get_feature_names_out() 


# Создаем Explanation объект вручную
shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=explainer.expected_value,
    data=X_test_preprocessed[:10],
    feature_names=feature_names
)


# График shap.plots.beeswarm с подписями
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_preprocessed[:100], feature_names=feature_names, show=False)


# Добавляем подписи
plt.title("SHAP Summary Plot: Влияние признаков на целевую переменную", pad=20)
plt.xlabel("SHAP значение (влияние на предсказание)")
plt.ylabel("Признаки")

# Улучшаем отображение
plt.tight_layout()
plt.show()


#График shap.plots.waterfall
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test_preprocessed[0],
    feature_names=feature_names,
), show=False)


# Добавляем подписи и заголовок
plt.title("Waterfall Plot: Вклад признаков в предсказание для отдельного наблюдения", pad=20)
plt.xlabel("Значение коэффициента")
plt.ylabel("Признаки")

# Улучшаем отображение
plt.tight_layout()
plt.show()


#График shap.plots.bar
mean_shap = np.abs(shap_values).mean(0)
sorted_idx = np.argsort(mean_shap)[-20:]  # Топ-20 признаков


plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), mean_shap[sorted_idx], color='#1f77b4')
plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
plt.title("Mean |SHAP values|")
plt.xlabel("Значение коэффициента")
plt.ylabel("Признаки")
plt.tight_layout()
plt.show()


# **Комментарий**
# 
# *Beeswarm Plot (Summary Plot)*
# 
# Общее описание
# - График показывает общее влияние признаков на модель по всем переменным
# - Наиболее важные признаки (расположены вверху) - оказывают наибольшее влияние на предсказания
# - Негативные значения Шепли слева от центральной вертикальной линии означают, что признак склоняет модель отнести объекты к классу 0 , а положительные справа — к классу 1.
# - Чем толще линия по вертикали, тем больше наблюдений сгруппированы вместе: у них близкие значения Шепли. Это значит, что модель уверенно присваивает SHAP-значения, с учётом закономерностей в данных.
# - Цвет точки отражает значения каждого признака объекта: чем краснее объект, тем больше признак, указанный на оси Y.
# 
# Интерпретация
# - для признака supervisor_evaluation  красные точки смещены вправо - значит оценка качества работы сотрудника, которую дал руководитель положительно влияют на целевую переменную - уровень удовлетворённости сотрудника работой.
# 
# 
# 
# *Waterfall Plot*
# 
# Общее описание
# 
# - График показывает декомпозицию предсказания для одного конкретного наблюдения, как каждый признак "толкает" предсказание от базового значения (среднего) к финальному
# 
# Интерпретация
# 
# - Признаки упорядочены по величине влияния
# - Длинные столбцы - наиболее значимые факторы для этого наблюдения
# - Направление:
#     - Вправо (положительное) - увеличивают предсказание (supervisor_evaluation - самый положительный)
#     
# *Bar Plot (Mean |SHAP values|)*
# 
# Общее описание
# 
# - График показывает cреднюю абсолютную важность признаков 
# - Упорядочивание признаков по влиянию на модель
# 
# Интерпретация
# 
# - Признаки вверху графика - наиболее важные для модели в среднем 
# - Длина столбца показывает силу влияния (без учета направления)
# - Полезен для сравнения относительной важности признаков
# - supervisor_evaluation - самый важный
# 

# <div class="alert alert-success">
#     
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Здорово, что используешь `SHAP` для анализа важности признаков.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Не забывай подписывать ось Y и название на графике. Так как `SHAP` график тоже является визуальным объектом, для его кастомизации мы можем использовать методы plt, как и для других графиков, однако это потребует небольшой настройки самого объекта `SHAP`. Подробнее можно глянуть тут: https://github.com/shap/shap/issues/594.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# в графиках выше добавлены  подписи оси Y, X  и названий на графике
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[70]:


# Вычисляем важность признаков
result = permutation_importance(
    randomized_search.best_estimator_, 
    X_train, # в трейн нет id, а в тесте есть
    y_train, # в трейн нет id, а в тесте есть
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Создаем DataFrame
feature_importance = pd.DataFrame({
    'Feature': [ 'dept', 'level', 'workload', 'employment_years', 'last_year_promo', 'last_year_violations', 'supervisor_evaluation', 'salary'],
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=True)
feature_importance
# Создаем график
sns.set_style('white')
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6));

# Добавляем подписи и заголовок
plt.title('Важность признаков (Permutation Importance)', fontsize=16, pad=20)
plt.xlabel('Среднее уменьшение точности', fontsize=12)
plt.ylabel('Признаки', fontsize=12)


# Улучшаем отображение
sns.despine()
plt.tight_layout()
plt.show()


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> На графике нет названия и подписи оси Х, все графики в этом блоке нужно интерпретировать.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# названия и подписи добавлены, комментарий ниже добавлен
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# **Комментарий**
# 
# *Важность признаков (Permutation Importance)*
# 
# Общее описание
# 
# - Метод permutation importance (перестановочная важность) показывает, насколько уменьшится качество модели (в данном случае - точность), если случайным образом перемешать значения конкретного признака. Чем больше падение точности - тем важнее признак для модели.
# 
# Интерпретация
# 
# - Признаки (ось Y)
#     - Отсортированы по важности (снизу - наименее важные)
# - Значения важности (ось X)
#     - Численно показывают среднее падение точности модели при перемешивании признака
#     - Чем длиннее столбец - тем важнее признак
# 
# - Наиболее важный признак - supervisor_evaluation
# - Наименее важный признак - last_year_promo

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Мы решаем задачу регрессии - сводить её в задачу классификации не нужно.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# убрала
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# **Выводы**
# 
# - Был построен 1 пайплайн:
#     - Целевой признак (job_satisfaction_rate)
#         - LinearRegression
#         - DecisionTreeRegressor
#    
# - Лучшая модель и её параметры: DecisionTreeRegressor(max_depth=18, min_samples_split=15, random_state=42) и MinMaxScaler()           
# - SMAPE (тренировочные данные): 11.399947748557818
# - SMAPE (тестовые данные): 14.006414402065234
# 
# - Наиболее важный признак - supervisor_evaluation 
# 
# 

# ## Задача 1 Шаг 6. Общий вывод

# **Выводы**
# 
# - 3 таблицы train_job_satisfaction_rate,  test_features,  test_target_job_satisfaction_rate были пролиты
# - есть пропуски "" и значения nan), "" переделаны в nan. обработка nan выполнена на этапе pipeline: пропущенные значения в категориальных признаках заполняются самым частотным значением ('most_frequent'), а числовые - меданным значениями ('median')
# - исправлена орфографическая ошибка senior в данных
# - типы даных корректны
# - нет строк-дубликатов в данных в каждой первоначальной таблице
# - графики распредения и ящики с усами построены
# - целевой признак распределн одинаково в тренировочной и тестовой выборке
# - построены коррялиционные матрицы по Пирсону, Спирмену и Фи:
#     - выявлена высокая корр. "+" связь между
#         - salary и workload
#         - salary и level
#         - supervisor_evaluation и job_satisfaction_rate      
#         - employment_years и	level
#         - job_satisfaction_rate и	last_year_violations
#         - last_year_violations и	job_satisfaction_rate
# - создана метрика SMAPE
# - Был построен 1 пайплайн:
# 
# - Целевой признак (job_satisfaction_rate)
#     - LinearRegression
#     - DecisionTreeRegressor
# - Лучшая модель и её параметры: DecisionTreeRegressor(max_depth=18, min_samples_split=15, random_state=42) и MinMaxScaler()           
# - SMAPE (тренировочные данные): 11.399947748557818
# - SMAPE (тестовые данные): 14.006414402065234 
# - Наиболее важный признак - supervisor_evaluation
# 
# 
#     
# **Рекомендация для заказчика**
# - С помощью построенной модели компания сможет предсказать уровень удовлетворённости сотрудника на основе данных и предпринять меры по повышению показателя. 
# - Более всего повышает уровень удовлетворённости зарплата и уровень нагрузки. 
# - Компания должна принять меры к сотрудникам с понижающимся уровнем удовлетворенности. Если компания не предпримет изменений, сотрудники будут увольняться или хуже работать.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Итоговый вывод нужно скорректировать.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# вывод скорректирован
# </div>
# 

# <div class="alert alert-success">
#     
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# ## Задача 2 Шаг 1. Загрузка данных

# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации⚠️:</b> Для этой задачи аналогичны все замечания из предыдущей в части аналогичных активностей. 
#         
# Ниже будут отмечены только новые замечания.

# In[71]:


#открытие данных
try:
    train_quit = pd.read_csv('/datasets/train_quit.csv')  #, sep=';' 
except:
    train_quit = pd.read_csv('https://code.s3.yandex.net/datasets/train_quit.csv')

#try:
#    test_features = pd.read_csv('/datasets/test_features.csv')  #, sep=';' 
#except:
#    test_features = pd.read_csv('https://code.s3.yandex.net/datasets/test_features.csv')
    
try:
    test_target_quit = pd.read_csv('/datasets/test_target_quit.csv')   #, sep=';' 
except:
    test_target_quit  = pd.read_csv('https://code.s3.yandex.net/datasets/test_target_quit.csv')     


# In[72]:


# вывод пяти случайных строк
train_quit.sample(5)


# In[73]:


# вывод пяти случайных строк
test_features.sample(5)


# In[74]:


# вывод пяти случайных строк
test_target_quit.sample(5) 


# In[75]:


#размер таблиц
         
print(f'Размер таблицы train_quit: { train_quit.shape}')  
print(f'Размер таблицы test_features: { test_features.shape}')  
print(f'Размер таблицы test_target_quit: { test_target_quit.shape}')  


#   **Выводы**
#   
#  - 3 таблицы train_quit,  test_features, test_target_quit были пролиты

# ## Задача 2 Шаг 2. Предобработка данных

# In[76]:


#основная информация о датафрейме
train_quit.info()


# In[77]:


#основная информация о датафрейме
test_features.info()


# In[78]:


#основная информация о датафрейме
test_target_quit.info()


# In[79]:


#анализ
train_quit.isna().sum()


# In[80]:


#анализ
test_features.isna().sum()


# In[81]:


#уникальные значения
test_features['dept'].unique()


# In[82]:


# заменим пустые строки на nan перед обработкой
test_features['dept'] = test_features['dept'].replace(' ', np.nan)


# In[83]:


#уникальные значения
test_features['dept'].unique()


# In[84]:


#уникальные значения
test_features['level'].unique()


# In[85]:


#анализ
test_target_quit.isna().sum()


# In[86]:


#исправить грамматическую ошибку в sinior.
test_features.replace("sinior", "senior", inplace=True)
train_quit.replace("sinior", "senior", inplace=True)
test_target_quit.replace("sinior", "senior", inplace=True)


# **Комментарий**
# 
# - есть пропуски (значения nan, " "), обработка пропусков выполнена на этапе pipeline: пропущенные значения в категориальных признаках заполняются самым частотным значением ('most_frequent'), а числовые - меданным значениями ('median')

# In[87]:


# количество строк-дубликатов в данных (должно быть ноль)
train_quit.duplicated().sum()


# In[88]:


# количество строк-дубликатов в данных (должно быть ноль)
test_target_quit.duplicated().sum()


# In[89]:


# количество строк-дубликатов в данных (должно быть ноль)
test_features.duplicated().sum()


# **Выводы**
#  
# - есть пропуски (значения nan), "" заменены на nan, nan обработка nan выполнена на этапе pipeline: пропущенные значения в категориальных признаках заполняются самым частотным значением ('most_frequent'), а числовые - меданным значениями ('median')
# - названия в данных корректны
# - типы даных корректны
# - нет строк-дубликатов в данных в каждой таблице

# ## Задача 2 Шаг 3. Исследовательский анализ данных

# ### Задача 2 3.1. Исследовательский анализ данных.

# #### Задача 2 Шаг 3.1 Train

# In[90]:


#разброс значений
pd.set_option('display.max_columns', None)
train_quit.describe()


# In[91]:


#построение гистограмм

train_quit.hist(figsize=(10, 10)
        , bins = 100
       );

plt.suptitle("Гистограмма распределения признаков train_quit", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[92]:


#countplot
sns.countplot(train_quit['quit']
                           );
plt.suptitle("Распределение quit из train_quit", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[93]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_quit_train = train_quit.pivot_table(index='quit', values='id', aggfunc='count')
pivot_quit_train.columns = ['count']
pivot_quit_train['ratio']= pivot_quit_train['count']/train_quit['id'].shape[0]
pivot_quit_train.sort_values(by='ratio', ascending=False)


# In[94]:


#countplot
sns.countplot(train_quit['employment_years']
             );
plt.suptitle("Распределение employment_years из train_quit ", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[95]:


#countplot
sns.countplot(train_quit['supervisor_evaluation']
             );
plt.suptitle("Распределение supervisor_evaluation' из train_quit ", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[96]:


## визуализация распределения признака
# частотная гистограмма распределения признака с bins=50
bins = plt.hist(train_quit['salary'], bins=50)
plt.vlines(x=train_quit['salary'].mean(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), label='Среднее')
plt.vlines(x=train_quit['salary'].median(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), linestyles='--', label='Медиана')
plt.title('Гистограмма распределения признака salary из train_quit')
plt.xlabel('Уровень зарплаты')
plt.ylabel('Частота')
plt.legend()
plt.show()

# ящик с усами с горизонтальной ориентацией
plt.title('График ящик с усами для признака salary из train_quit')
plt.boxplot(train_quit['salary'], vert=False)
plt.xlabel('Уровень зарплаты')
plt.show()


# In[97]:


# корреляционный анализ количесвенных признаков по пирсону
sns.heatmap(train_quit.corr(), annot=True, fmt='.2f',  cmap='coolwarm', center=0);
plt.suptitle("Корреляционный анализ количесвенных признаков по Пирсону из train_quit", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[98]:


# корреляционный анализ количественных признаков по spearman
corr_spearman = train_quit.drop(['id'], axis=1).corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='coolwarm', center=0);
plt.suptitle("Корреляционный анализ количесвенных признаков по Спирмену из train_quit", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[99]:


# корреляционный анализ количественных  и качественных признаков
train_quit.drop(['id'], axis=1).phik_matrix(interval_cols=interval_cols)


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Не удалён `id`, не настроен `interval_cols`. Тут и ниже в `test`.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v2 <a class="tocSkip"></h2>
# 
#   учтено  выше
# <b>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> См.ранее по `interval_cols`. В `test` выборке ниже не удалён `id`.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v3 <a class="tocSkip"></h2>
# 
# - interval_cols  учтено  выше
# - В test выборке ниже удалила id.
# <b>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.4 <a class="tocSkip"> </h2>
#  
# <b>Все отлично!👍:</b> Учтено.

# #### Задача 2 Шаг 3.1 Test

# In[100]:


#разброс значений
#pd.set_option('display.max_columns', None)

test_features.drop(['id'], axis=1).describe()


# In[101]:


#построение гистограмм


test_features.hist(figsize=(10, 10)
        , bins = 100
       );

plt.suptitle("Гистограмма распределения признаков test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

test_target_quit.hist(figsize=(5, 5)
        , bins = 100
       );
plt.suptitle("Гистограмма распределения признаков test_target_quit", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[102]:


#countplot
sns.countplot(test_target_quit['quit']
                           );
plt.suptitle("Распределение quit из test_target_quit", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[103]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_quit_test = test_target_quit.pivot_table(index='quit', values='id', aggfunc='count')
pivot_quit_test.columns = ['count']
pivot_quit_test['ratio']= pivot_quit_test['count']/test_target_quit['id'].shape[0]
pivot_quit_test.sort_values(by='ratio', ascending=False)


# In[104]:


# корреляционный анализ количесвенных признаков по пирсону
sns.heatmap(test_features.drop(['id'], axis=1).corr(), annot=True, fmt='.2f',  cmap='coolwarm', center=0);
plt.suptitle("Корреляционный анализ количесвенных признаков по Пирсону из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[105]:


# корреляционный анализ количесвенных признаков по spearman
corr_spearman = test_features.corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='coolwarm', center=0);
plt.suptitle("Корреляционный анализ количесвенных признаков по Спирмену из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[106]:


# корреляционный анализ количественных  и качественных признаков
test_features.drop(['id'], axis=1).phik_matrix(interval_cols = interval_cols)


# **Выводы 3.1**
#  
# - графики распредения и ящики с усами построены
# - целевой признак распределн одинаково в тренировочной и тестовой выборке   
# - построены коррялиционные матрицы по Пирсону, Спирмену и Фи:
#     - выявлена высокая корр. "+" связь между
#         - salary и 	level
#         - salary и 	workload
#         - employment_years и level

# ### Задача 2 3.2. Портрет «уволившегося сотрудника»

# In[107]:


#портрет «уволившегося сотрудника»

# диаграммы рассеяния 
sns.scatterplot(
    data=train_quit
    , x="dept"
    , y="salary" 
    ,  hue="quit"
    , style="quit"
);
plt.suptitle("Диаграмма рассеяния департаментов и зарплаты", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[108]:


# диаграммы рассеяния 
sns.scatterplot(
    data=train_quit
    , x="workload"
    , y="salary" 
    ,  hue="quit"
    , style="quit"
);
plt.suptitle("Диаграмма рассеяния загрузки и зарплаты", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# In[109]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit = train_quit.pivot_table(index=['dept','quit'], values='id', aggfunc='count')
pivot_train_quit.columns = ['count']
pivot_train_quit['ratio']= pivot_train_quit['count']/train_quit['id'].shape[0]
pivot_train_quit.sort_values(by='ratio', ascending=False)


# In[110]:


df1=train_quit.query('quit =="no"')
df1['salary'].mean()


# In[111]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit = train_quit.pivot_table(index='quit', values='salary', aggfunc='mean')
pivot_train_quit.columns = ['mean']

df1=train_quit.query('quit =="no"')
mean_no=df1['salary'].mean()


pivot_train_quit['growth_rate']= (pivot_train_quit['mean']/mean_no - 1)
pivot_train_quit.sort_values(by='mean', ascending=False)


# In[112]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit = train_quit.pivot_table(index=['dept','quit'], values='salary', aggfunc='mean')
pivot_train_quit.columns = ['mean']
pivot_train_quit.sort_values(by='mean', ascending=False)


# In[113]:


#фрейм только по уволившимся 
train_quit_yes = train_quit.query('quit=="yes"')


# In[114]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='dept', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)


# In[115]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='level', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)


# In[116]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='workload', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)


# In[117]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='last_year_promo', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)


# In[118]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='last_year_violations', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)


# In[119]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='employment_years', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)


# In[120]:


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='supervisor_evaluation', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)


# **Выводы 3.2**
#  
# - портрет «уволившегося сотрудника».
#     - департамента sales и ниже зарплата => выше вероятность уволиться
#     - выше уровень загруженности и ниже зарплата => выше вероятность уволиться 
#     - среднее значение зарплаты ушедших сотрудников ниже на 37% по сравнению с теми, кто остался в компании   
# - портрет «уволившегося сотрудника» в:
#     - 36%  из sales департамента
#     - 89% junior уровня
#     - 46% с низкой загрузкой
#     - 99.9% без повышения в прошлом году
#     - 80% без нарушений
#     - 53% работают 1 год
#     - 46% оценка "3"

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> 
#     
# 1. Анализ должен быть выполнен по всем признакам таблицы.
#     
# 2. Анализ категориальных/дискретных признаков нужно сделать с помощью относительных величин - оперировать абсолютными величинами не всегда удобно, так как в зависимости от масштаба исследования (компания на 100 человек и компания на 100 000 человек, например) результаты могут быть более или менее интерпретируемыми. Для категориальных/дискретных признаков мы можем использовать `value_counts(normalize=True)`.
#     
# 3. При этом обрати внимание, что относительные величины нужно считать именно по срезу `quit='yes'`, а не делать анализ в разрезе `quit`: в случае, если у нас будет сильный дисбаланс целевого признака (например, 99/1), текущий формат анализа будет неинформативен, так как все доли оттока будут в где-то на дне графика.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v.2 <a class="tocSkip"></h2>
#   
# выполнен выше
#     
# <b> 

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Нет оценки долей по дискретным признака.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v.3 <a class="tocSkip"></h2>
#   
# выполнен выше
#     

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.4 <a class="tocSkip"> </h2>
#  
# <b>Все отлично!👍:</b> Учтено.

# ### Задача 2 3.3. Распределения признака job_satisfaction_rate для ушедших и оставшихся сотрудников.

# In[121]:


#объединение таблиц  train_quit и satisfaction_rate

df_sr = test_target_job_satisfaction_rate.loc[:, ['id' , 'job_satisfaction_rate']]


data_quit_sr = pd.merge(test_target_quit, df_sr, on='id', how='left')

data_quit_sr_no = data_quit_sr.query('quit == "no"')
data_quit_sr_yes = data_quit_sr.query('quit == "yes"')


# In[122]:


## визуализация распределения признака
# частотная гистограмма распределения признака с bins=50
bins = plt.hist(data_quit_sr_no['job_satisfaction_rate'], bins=50)
plt.vlines(x=data_quit_sr_no['job_satisfaction_rate'].mean(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), label='Среднее')
plt.vlines(x=data_quit_sr_no['job_satisfaction_rate'].median(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), linestyles='--', label='Медиана')
plt.title('Гистограмма распределения признака job_satisfaction_rate из data_quit_sr_no')
plt.xlabel('Уровень удовлетворенности работой')
plt.ylabel('Частота')
plt.legend()
plt.show()

# ящик с усами с горизонтальной ориентацией
plt.title('График ящик с усами для признака job_satisfaction_rate из data_quit_sr_no')
plt.boxplot(data_quit_sr_no['job_satisfaction_rate'], vert=False)
plt.xlabel('Уровень удовлетворенности работой')
plt.show()

data_quit_sr_no_mean = data_quit_sr_no['job_satisfaction_rate'].mean()
data_quit_sr_no_median = data_quit_sr_no['job_satisfaction_rate'].median()

print('Медиана признака job_satisfaction_rate тестовой выборки для оставшихся сотрудников', data_quit_sr_no_median)
print('Среднее значение признака job_satisfaction_rate тестовой выборки для оставшихся сотрудников', data_quit_sr_no_mean)


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Не забывай про подписи осей на гистограммах.

# <div class="alert alert-info">
# <h2> Комментарий студента v2 <a class="tocSkip"> </h2>
# 
#  сделаны выше     
# </div>

# <div class="alert alert-success">
#     
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[123]:


## визуализация распределения признака
# частотная гистограмма распределения признака с bins=50
bins = plt.hist(data_quit_sr_yes['job_satisfaction_rate'], bins=50)
plt.vlines(x=data_quit_sr_yes['job_satisfaction_rate'].mean(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), label='Среднее')
plt.vlines(x=data_quit_sr_yes['job_satisfaction_rate'].median(), colors='red', ymin=bins[0].min(), ymax=bins[0].max(), linestyles='--', label='Медиана')
plt.title('Гистограмма распределения признака job_satisfaction_rate из data_quit_sr_yes')
plt.xlabel('Уровень удовлетворенности работой')
plt.ylabel('Частота')
plt.legend()
plt.show()

# ящик с усами с горизонтальной ориентацией
plt.title('График ящик с усами для признака job_satisfaction_rate из data_quit_sr_yes')
plt.boxplot(data_quit_sr_yes['job_satisfaction_rate'], vert=False)
plt.xlabel('Уровень удовлетворенности работой')
plt.show()

data_quit_sr_yes_mean = data_quit_sr_yes['job_satisfaction_rate'].mean()
data_quit_sr_yes_median = data_quit_sr_yes['job_satisfaction_rate'].median()

print('Медиана признака job_satisfaction_rate тестовой выборки для уволившихся сотрудников', data_quit_sr_yes_median)
print('Среднее значение признака job_satisfaction_rate тестовой выборки для уволившихся сотрудников', data_quit_sr_yes_mean)


# **Выводы 3.3**
#   
# - Уровень удовлетворённости сотрудника работой в компании влияет на то, уволится ли сотрудник. Данные визуализированы: 
#     - Медиана признака job_satisfaction_rate тестовой выборки для уволившихся сотрудников 0.37
#     - Среднее значение признака job_satisfaction_rate тестовой выборки для уволившихся сотрудников 0.38771276595744686
#     - Медиана признака job_satisfaction_rate тестовой выборки для оставшихся сотрудников 0.66
#     - Среднее значение признака job_satisfaction_rate тестовой выборки для оставшихся сотрудников 0.6120403899721448

# **Выводы**
#  
# - графики распредения и ящики с усами построены
# - целевой признак распределн одинаково в тренировочной и тестовой выборке
# - портрет «уволившегося сотрудника».
#     - департамента sales и ниже зарплата => выше вероятность уволиться
#     - выше уровень загруженности и ниже зарплата => выше вероятность уволиться 
#     - среднее значение зарплаты ушедших сотрудников ниже на 37% по сравнению с теми, кто остался в компании   
# - портрет «уволившегося сотрудника» в:
#     - 36%  из sales департамента
#     - 89% junior уровня
#     - 46% с низкой загрузкой
#     - 99.9% без повышения в прошлом году
#     - 80% без нарушений
#     - 53% работают 1 год
#     - 46% оценка "3"
#     
#     
# - Уровень удовлетворённости сотрудника работой в компании влияет на то, уволится ли сотрудник. Данные визуализированы: 
#     - Медиана признака job_satisfaction_rate тестовой выборки для уволившихся сотрудников 0.37
#     - Среднее значение признака job_satisfaction_rate тестовой выборки для уволившихся сотрудников 0.38771276595744686
#     - Медиана признака job_satisfaction_rate тестовой выборки для оставшихся сотрудников 0.66
#     - Среднее значение признака job_satisfaction_rate тестовой выборки для оставшихся сотрудников 0.6120403899721448
#     
# - построены коррялиционные матрицы по Пирсону, Спирмену и Фи:
#     - выявлена высокая корр. "+" связь между
#         - salary и 	level
#         - salary и 	workload
#         - employment_years и level

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Этот раздел нужно разделить на задания 3.1, 3.2 и 3.3 согласно целевой структуре проекта. Сейчас все активности собраны в одном блоке и сделаны без каких-либо разделений, что делает невозможным проверку этой части проекта, так как нет понимания, какая активность к кому заданию относится.
#     
# На каждое задание не забудь сформулирвоать вывод.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# - раздел  разделен на задания 3.1, 3.2 и 3.3
# - выводы сделаны выше     
# </div>
# 

# <div class="alert alert-success">
#     
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# ## Задача 2 Шаг 4. Добавление нового входного признака

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Тут аналогично не нужно вертикальное соединение данных и сплитование.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v.2 <a class="tocSkip"></h2>
# 
# исправлено ниже
#     
# <b> 

# <div class="alert alert-success">
#     
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[124]:


# Добавляем предсказания модели в исходный датасет
train_quit['predicted_job_satisfaction'] = randomized_search.predict(train_quit.drop(['quit'], axis=1))
train_quit.info()


# In[125]:


test_features.info()


# In[126]:


# Добавляем предсказания модели в исходный датасет
test_features_3 = test_features.drop(['id'], axis=1)
test_features_3['predicted_job_satisfaction'] = randomized_search.predict(test_features_3)
test_features_3


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Обрати внимание, что `predicted_job_satisfaction` не может быть равен нулю. Также с помощью предсказаний нужно получить этот признак для `train`.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# - predicted_job_satisfaction исправлен, теперь не равен нулю, скрипт выше
#     
#     
# </div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Предсказания нужно сделать отдельно для `train` и отдельно для `test` - сейчас они делаются для совмещённой выборки `df2`.

# <div class="alert alert-info">
# <h2> Комментарий студента v2 <a class="tocSkip"> </h2>
# 
# выполнено выше
#     
#     
# </div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Предсказания для `test` закомментированы и не выполняются.

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.4 <a class="tocSkip"> </h2>
#  
# <b>Все отлично!👍:</b> Учтено.

# In[127]:


# количество строк-дубликатов в данных (должно быть ноль)
train_quit_no_id = train_quit.drop(['id'], axis=1)
train_quit_no_id.duplicated().sum()


# In[128]:


# удаление строк-дубликатов в данных (сохранить те же индексы строк, что были у них до удаления дубликатов)
train_quit_no_id = train_quit_no_id.drop_duplicates()
train_quit_no_id.duplicated().sum()


# In[129]:


#interval_cols
interval_cols2 = ['salary'
                 , 'predicted_job_satisfaction']


# In[130]:


#корреляционный анализ количественных и качественных признаков
corr_phik=test_features_3.phik_matrix(interval_cols=interval_cols2)
corr_phik


# In[131]:


#корреляционный анализ количественных и качественных признаков
corr_phik=train_quit_no_id.phik_matrix(interval_cols=interval_cols2)
corr_phik


# **Выводы**
#  
# - job_satisfaction_rate, предсказанный лучшей моделью первой задачи, добавлен к входным признакам второй задачи в df_2
# - из датафрейма для будущего обучения удален id, удалены дубликаты и проведен анализ корреляции
# - новый признак не создаёт мультиколлинеарности с уже существующими признаками.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> После добавления новых признаков нужно повторно провести анализ корреляции выборок, чтобы убедиться, что новый признак не создаёт мультиколлинеарности с уже существующими признаками.
#     

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# анализ корреляции добавлен выше
# </div>
# 

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Тоже не настроен `interval_cols`.

# <div class="alert alert-info">
# <h2> Комментарий студента v2
#    <a class="tocSkip"> </h2>
# 
# добавлено выше
# </div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> См.ранее по `interval_cols`.

# <div class="alert alert-info">
# <h2> Комментарий студента v3
#    <a class="tocSkip"> </h2>
# 
# добавлено выше
# </div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.4 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> В текущем `interval_cols` должен быть `predicted_job_satisfaction`, а не `job_satisfaction_rate`, как сейчас.

# <div class="alert alert-info">
# <h2> Комментарий студента v4
#    <a class="tocSkip"> </h2>
# 
# добавлено выше
# </div>

# ## Задача 2 Шаг 5. Подготовка данных

# In[132]:


# Добавляем предсказания модели в исходный датасет
test_features_4 = test_features
test_features_4['predicted_job_satisfaction'] = randomized_search.predict(test_features)
test_features_4


# In[133]:


# данные для теста
test_df2 = pd.merge(test_features,test_target_quit, on='id', how='left')
test_df2.info()


# In[134]:


# пайплайн, который выберет лучшую комбинацию модели и гиперпараметров. 

RANDOM_STATE = 42


# загружаем данные
X_train = train_quit_no_id.drop(['quit' 
                                ], axis=1)

X_test = test_df2.drop(['quit'], axis=1)
y_train = train_quit_no_id['quit']
y_test = test_df2['quit']


X_train.shape, X_test.shape,y_train.shape, y_test.shape


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Кодирование нельзя делать руками:
# - при появлении новых значений категориальных признаков, не предусмотренных твоим решением, модель или упадёт, или будет генерировать ошибки;
# - такое решение не сделаешь частью `Pipeline`, а все современные решения базируются на них.
#     
# Поэтому кодирование нужно делать только с применением обучаемых трансформеров. Для кодирования целевого признака нужно использовать `LabelEncoder` (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# - LabelEncoder создан ниже
#     
#     
# </div>
#     

# <div class="alert alert-success">
#     
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[135]:


# создаём списки с названиями признаков

ohe_columns = ['dept']

ord_columns = ['level'
                 , 'workload'
                , 'last_year_promo'
                , 'last_year_violations' 
              ] 

num_columns = [  'employment_years'
               , 'supervisor_evaluation'
               , 'salary'
              ]

# создаём пайплайн для подготовки признаков из списка ohe_columns: заполнение пропусков и OHE-кодирование
# SimpleImputer + OHE
ohe_pipe = Pipeline(
    [('simpleImputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
     ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))  # , sparse=False - убираем (если библитоека новая !pip install -U scikit-learn )handle_unknown='error' - при появлении новых категорий в тестовых данных пайплайн упадёт с ошибкой.
    ]
    )


# создаём пайплайн для подготовки признаков из списка ord_columns: заполнение пропусков и Ordinal-кодирование
# SimpleImputer + OE
ord_pipe = Pipeline(
    [('simpleImputer_before_ord', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
     ('ord',  OrdinalEncoder(
                categories= [
                    ['junior', 'middle', 'sinior'],
                    ['medium', 'high', 'low'] ,                   
                     ['no', 'yes'] ,                   
                     ['no', 'yes']                                        
                ], 
                handle_unknown='use_encoded_value', unknown_value=np.nan
           )
        ),
     ('simpleImputer_after_ord', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))
    ]
)


# создаём пайплайн для подготовки признаков из списка num_columns: заполнение пропусков 

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), # полиномиальные признаки для числовых колонок
    ('scaler', StandardScaler())
])

# создаём общий пайплайн для подготовки данных
data_preprocessor = ColumnTransformer(
    [
     ('ohe', ohe_pipe, ohe_columns),
     ('ord', ord_pipe, ord_columns),
     ('num', MinMaxScaler(), num_columns)
    ], 
    remainder='passthrough'
)


# Кодирование целевой переменной LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)  # yes -> 1, no -> 0
y_test_encoded = le.transform(y_test)


# создаём итоговый пайплайн: подготовка данных и модель
pipe_final = Pipeline([
    ('preprocessor', data_preprocessor),
    ('models', DecisionTreeClassifier())  # Регрессор по умолчанию
]
)



param_grid = [
  
    
    # словарь для модели DecisionTreeClassifier()
    {
        'models': [DecisionTreeClassifier(random_state=RANDOM_STATE)],
        'models__max_depth': range(2, 20),
        'models__min_samples_split': range(2, 20),
        'models__min_samples_leaf': range(2, 25),
        'models__max_features': ['sqrt', 'log2', None],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough']
    }
     ,

    
    # словарь для модели KNeighborsClassifier() 
     {
           'models': [KNeighborsClassifier()]
          ,'models__n_neighbors': range(2, 50)
          ,'models__weights': ['uniform', 'distance']       
#uniform – если все соседи примерно равноценны (например, данные нормализованы, и расстояния не сильно различаются).
#distance – если ближайшие объекты важнее дальних (например, в данных есть выбросы или шум).
          , 'preprocessor__num': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough']   
      },

    # словарь для модели LogisticRegression()
     {
         'models': [LogisticRegression(
               random_state=RANDOM_STATE,
               max_iter=500   # увеличиваем количество итераций
         )],
         'models__penalty': ['l1', 'l2'],
         'models__solver': ['saga' ,  'liblinear'],
         'models__C': [0.001, 0.01, 0.1, 1, 10, 100],
         'preprocessor__num': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough'] 
     }
      ,
    
    # словарь для модели SVC() #'models__gamma': [0.1,1,10]        'models__C': [0.1,1,10,100] 
     {
        'models': [SVC(
             random_state=RANDOM_STATE
             , probability=True # Если probability=False, SVC не сможет вернуть вероятности, и вызов predict_proba() приведёт к ошибке
         )]
         ,'models__kernel': ['rbf','linear'] # исключены 'sigmoid','poly' для более быстрой отработки запроса
         ,'preprocessor__num': [StandardScaler(), MinMaxScaler(), RobustScaler(), 'passthrough'] 
     }
     #,

    # RandomForest
 #    {
 #        'models': [RandomForestClassifier(random_state=RANDOM_STATE)],
 #        'models__n_estimators': [50, 100, 200],
 #        'models__max_depth': [None, 10, 20, 30],
  #       'models__min_samples_split': [2, 5, 10],
  #       'preprocessor__num': [StandardScaler(), None]
 #    },
    
    # GradientBoosting
 #    {
 #        'models': [GradientBoostingClassifier(random_state=RANDOM_STATE)],
 #        'models__n_estimators': [50, 100],
 #        'models__learning_rate': [0.01, 0.1],
 #        'models__max_depth': [3, 5, 7],
 #        'preprocessor__num': [None]
 #    }    
        
]


# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Нужно исследовать не менее 3 моделей - сейчас активны только 2.

# <div class="alert alert-block alert-info">
# <h2> Комментарий студента v2 <a class="tocSkip"></h2>
# 
#     
#  работают 4 модели <b>

# <div class="alert alert-success">
#     
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Учтено.

# In[136]:


randomized_search = RandomizedSearchCV(
    pipe_final, 
    param_grid, 
    cv=5,
    scoring='roc_auc',  
    n_iter=200,  # количество итераций
    random_state=RANDOM_STATE,
    n_jobs=-1,
    #error_score='raise'
)


# ##   Задача 2 Шаг 6. Обучение модели

# In[137]:



# Обучаем модель
randomized_search.fit(X_train, y_train_encoded) 

# Получаем предсказания
train_predictions = randomized_search.predict(X_train)
test_predictions = randomized_search.predict(X_test)

# вероятности предсказаний (не для линейной регресии)
train_predictions_proba = randomized_search.predict_proba(X_train)[:, 1]
test_predictions_proba  = randomized_search.predict_proba(X_test)[:, 1]


print('Лучшая модель и её параметры:\n\n', randomized_search.best_estimator_)
print ('Метрика лучшей модели при кросс-валидации (ROC-AUC):', randomized_search.best_score_)
print ('ROC-AUC (тренировочные данные):', roc_auc_score(y_train, train_predictions_proba))
print("ROC-AUC (тестовые данные):", roc_auc_score(y_test, test_predictions_proba))


# **Комментарий**
# - Лучшая модель и её параметры: DecisionTreeClassifier(max_depth=8, max_features='sqrt', min_samples_leaf=12, min_samples_split=3, random_state=42) и passthrough
# - ROC-AUC (тренировочные данные): 0.9392185450350071
# - ROC-AUC (тестовые данные): 0.9161548776151247
# 

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Тут аналогично не достигнуто требуемое качество модели - ошибка та же, что и в первой задаче.

# <div class="alert alert-info">
# <h2> Комментарий студента <a class="tocSkip"> </h2>
# 
# достигнут ROC-AUC 0.93 на тестовой выборке
# </div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.2 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Текущая оценка аналогично получена на неправильных данных.

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.3 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Тут та же проблема - проверь подготовку `test` выборки.

# <div class="alert alert-info">
# <h2> Комментарий студента v3
#    <a class="tocSkip"> </h2>
# 
# напишите точнее, что вы имеете ввиду- я не понимаю вас.
# все комментарии выше учтены, в т.ч. комментарий про предсказания
# </div>

# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера v.4 <a class="tocSkip"></h2>
# 
#     
# <b>На доработку❌:</b> Здесь, как писал ранее, та же проблема соответствия строк - убедись, что предсказания и реальные ответы сравниваются по одному и тому же работнику. Сейчас оценка на `test` закомментирована и не выполняется.

# <div class="alert alert-info">
# <h2> Комментарий студента v4
#    <a class="tocSkip"> </h2>
#     
# - тестовый фрейм поправлен
# - оценка на `test` выведена
# - достигнут ROC-AUC (тестовые данные): 0.9161548776151247
#     
# </div>

# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации⚠️:</b> Аналогично стоит выполнить оценку модели на адеватность - в случае задачи классификации мы можем использовать  `DummyClassifier` (https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier).
#         

# In[138]:


#вывод всех результатов моделей в датафрейм
#pd.set_option('display.max_columns', None)

# все результаты кросс-валидации
results_df = pd.DataFrame(randomized_search.cv_results_)

# Выбор и сортировка нужных колонок
final_results = results_df[
    ['rank_test_score', 'param_models', 'mean_test_score', 'std_test_score', 'mean_score_time', 'params']
].sort_values('rank_test_score')

# отображение
pd.set_option('display.max_colwidth', 200)
final_results.head(3)


# In[139]:


# Вычисляем важность признаков
result = permutation_importance(
    randomized_search.best_estimator_, 
    X_test, 
    y_test_encoded,
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Создаем DataFrame
feature_importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=True)
feature_importance
# Создаем график
sns.set_style('white')
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6));


# In[140]:


# важность признаков
feature_importance


# **Выводы**
#  
#  - Были построен  1 пайплайн:
#     - Целевой признак (quit) 
#         - DecisionTreeClassifier  
#         - LogisticRegression
#         - KNeighborsClassifier
#         - SVC
# 
# - Лучшая модель и её параметры: DecisionTreeClassifier(max_depth=8, max_features='sqrt', min_samples_leaf=12, min_samples_split=3, random_state=42) и passthrough
# - ROC-AUC (тренировочные данные): 0.9392185450350071
# - ROC-AUC (тестовые данные): 0.9161548776151247
# 
# - Для лучшей модели самые важные параметры: predicted_job_satisfaction
# 

# ## Задача 2 Шаг 7. Общий вывод

# **Выводы**
# 
# - 3 таблицы train_quit, test_features, test_target_quit были пролиты
# - есть пропуски (значения nan), обработка пропусков выполнена на этапе pipeline: пропущенные значения в категориальных признаках заполняются самым частотным значением ('most_frequent'), а числовые - меданным значениями ('median')
# - названия в данных корректны
# - типы даных корректны
# - нет строк-дубликатов в данных в каждой таблице
# - графики распредения и ящики с усами построены
# - целевой признак распределн одинаково в тренировочной и тестовой выборке
# - портрет «уволившегося сотрудника».
#     - департамента sales и ниже зарплата => выше вероятность уволиться
#     - выше уровень загруженности и ниже зарплата => выше вероятность уволиться
#     - среднее значение зарплаты ушедших сотрудников ниже на 37% по сравнению с теми, кто остался в компании
# - уровень удовлетворённости сотрудника работой в компании влияет на то, уволится ли сотрудник. Данные визуализированы:
#     - 	Медиана признака job_satisfaction_rate тестовой выборки для уволившихся сотрудников 0.37
#     - 	Среднее значение признака job_satisfaction_rate тестовой выборки для уволившихся сотрудников 0.38771276595744686
#     - 	Медиана признака job_satisfaction_rate тестовой выборки для оставшихся сотрудников 0.66
#     - 	Среднее значение признака job_satisfaction_rate тестовой выборки для оставшихся сотрудников 0.6120403899721448
# - построены коррялиционные матрицы по Пирсону, Спирмену и Фи:
#     - 	выявлена высокая корр. "+" связь между
#         - 	salary и level
#         - 	salary и workload
#         - 	employment_years и quit
# -	Был построен 1 пайплайн:
#     -	Целевой признак (quit)
#         - DecisionTreeClassifier  
#         - LogisticRegression
#         - KNeighborsClassifier
#         - SVC
# 
# - Лучшая модель и её параметры: DecisionTreeClassifier(max_depth=8, max_features='sqrt', min_samples_leaf=12, min_samples_split=3, random_state=42) и passthrough
# - ROC-AUC (тренировочные данные): 0.9392185450350071
# - ROC-AUC (тестовые данные): 0.9161548776151247
# 
# - Для лучшей модели самые важные параметры: predicted_job_satisfaction
# 
# **Рекомендация для заказчика**
# - С помощью построенной модели компания сможет предсказать то, что сотрудник уволится из компании.
# - Более всего на увольнение влияет зарплата и стаж. Чем ниже зарплата и стаж, тем вероятнее уволится сотрудник.
# - Компания должна принять меры и предовращать отток сотнудник, увеличивая мотивацию как материальную, так и нематериальную не увольняться из компании.

# ## Общий вывод

# 
#  **Описание проекта**
#  
#  - Заказчик: HR-аналитики компании «Работа с заботой»
#  - Цель анализа:  оптимизировать управление персоналом: как избежать финансовых потерь и оттока сотрудников
#  - Образ результата: ML-модель, которая: 
#      - предсказывает уровень удовлетворённости сотрудника на основе данных заказчика. 
#      - предсказывает вероятность, что сотрудник уволится из компании
# 
#  - Входные данные задачи 1 (уровень удовлетворённости сотрудника) :
#  
#      - Тренировочная выборка:train_job_satisfaction_rate.csv
#      - Входные признаки тестовой выборки: test_features.csv
#      - Целевой признак тестовой выборки:test_target_job_satisfaction_rate.csv
# 
#  - Входные данные задачи 2 (вероятность, что сотрудник уволится из компании) :
#     - Тренировочная выборка: train_quit.csv
#     - Входные признаки тестовой выборки те же, что и в прошлой задаче: test_features.csv
#     - Целевой признак тестовой выборки: test_target_quit.csv
# 
#  
#  **Ход исследования**
#  
# уровень удовлетворённости сотрудника:    
#     
#  - Шаг 1. Загрузка данных
#  - Шаг 2. Предобработка данных
#  - Шаг 3. Исследовательский анализ данных
#  - Шаг 4. Подготовка данных
#  - Шаг 5. Обучение моделей
#  - Шаг 6. Общий вывод
# 
# вероятность, что сотрудник уволится из компании:
# 
#  - Шаг 1. Загрузка данных
#  - Шаг 2. Предобработка данных
#  - Шаг 3. Исследовательский анализ данных
#  - Шаг 4. Добавление нового входного признака
#  - Шаг 5. Подготовка данных
#  - Шаг 6. Обучение моделей
#  - Шаг 7. Общий вывод
#  
#  
# 
#  **Рекомендация для заказчика**
#  
# 
# - С помощью построенной модели 1  компания сможет предсказать уровень удовлетворённости сотрудника на основе данных и предпринять меры по повышению показателя. 
# - Более всего повышает уровень удовлетворённости зарплата и уровень нагрузки. 
# - Компания должна принять меры к сотрудникам с понижающимся уровнем удовлетворенности. Если компания не предпримет изменений, сотрудники будут увольняться или хуже работать.
# 
# - С помощью построенной модели 2  компания сможет предсказать то, что сотрудник уволится из компании.
# - Более всего на увольнение влияет зарплата и стаж. Чем ниже зарплата и стаж, тем вероятнее уволится сотрудник.
# - Компания должна принять меры и предовращать отток сотнудник, увеличивая мотивацию как материальную, так и нематериальную не увольняться из компании.
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> Есть итоговый вывод и рекомендации.

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# # Комментарий ревьюера: общий вывод по проекту.
# 
# Татьяна, проект получился на довольно хорошем уровне - отличная работа над проектом, молодец!
# 
# Мне нравится твой аналитический подход к выполнению проекта, ты соблюдаешь структуру работы, выполняешь её последовательно - это очень хорошо! Шаги проекта выполнены по порядку согласно плану проекта, нет смысловых и структурных ям. Важно, что не забываешь про выводы.
#     
# Над проектом ещё стоит поработать - есть рекомендации по дополнению некоторых твоих шагов проекта. Такие рекомендации я отметил жёлтыми комментариями. Будет здорово, если ты учтёшь их - так проект станет структурно и содержательно более совершенным.
#     
# Также в работе есть критические замечания. К этим замечаниям я оставил пояснительные комментарии красного цвета, в которых перечислил возможные варианты дальнейших действий. Уверен, ты быстро с этим управишься:)
#     
# Если о том, что нужно сделать в рамках комментариев, будут возникать вопросы - оставь их, пожалуйста, в комментариях, и я отвечу на них во время следующего ревью.
#     
# Также буду рад ответить на любые твои вопросы по проекту или на какие-либо другие, если они у тебя имеются - оставь их в комментариях, и я постараюсь ответить:)
#     
# Жду твой проект на повторном ревью. До встречи:)
