
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

# добавлю комментарий , чтобы сохранить коммит2

# ## Импорт библиотек

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


# ## Задача 1 Шаг 1. Загрузка данных

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
      
# вывод пяти случайных строк
train_job_satisfaction_rate.sample(5) 

# вывод пяти случайных строк
test_features.sample(5) 


# вывод пяти случайных строк
test_target_job_satisfaction_rate.sample(5) 


#размер таблиц
         
print(f'Размер таблицы train_job_satisfaction_rate: { train_job_satisfaction_rate.shape}')  
print(f'Размер таблицы test_features: { test_features.shape}')  
print(f'Размер таблицы test_target_job_satisfaction_rate: { test_target_job_satisfaction_rate.shape}')  


#   **Выводы**
#   
#  - 3 таблицы train_job_satisfaction_rate,  test_features,  test_target_job_satisfaction_rate были пролиты

# ## Задача 1 Шаг 2. Предобработка данных


#основная информация о датафрейме
train_job_satisfaction_rate.info()

#основная информация о датафрейме
test_features.info()


#основная информация о датафрейме
test_target_job_satisfaction_rate.info()

#анализ
train_job_satisfaction_rate.isna().sum()

#анализ
test_features.isna().sum()

#анализ
test_target_job_satisfaction_rate.isna().sum()


# **Комментарий**
# 
# Обработка пропусков будет на этапе pipeline: пропущенные значения сначала заполняются nan, а затем самым частотным значением ('most_frequent').


# количество строк-дубликатов в данных (должно быть ноль)
train_job_satisfaction_rate.duplicated().sum()


# количество строк-дубликатов в данных (должно быть ноль)
test_target_job_satisfaction_rate.duplicated().sum()



# количество строк-дубликатов в данных (должно быть ноль)
test_features.duplicated().sum()



#уникальные значения
train_job_satisfaction_rate['dept'].unique()



#уникальные значения
train_job_satisfaction_rate['level'].unique()


#уникальные значения
train_job_satisfaction_rate['workload'].unique()


# **Выводы**
#  
# - есть пропуски (значения nan), обработка пропусков будет на этапе pipeline: пропущенные значения заполняются самым частотным значением ('most_frequent')
# - названия в данных корректны
# - типы даных корректны
# - нет строк-дубликатов в данных в каждой таблице

#исправить грамматическую ошибку в sinior.
train_job_satisfaction_rate.replace("sinior", "senior", inplace=True)
test_features.replace("sinior", "senior", inplace=True)
test_target_job_satisfaction_rate.replace("sinior", "senior", inplace=True)

#анализ
test_features['dept'].unique()

# заменим пустые строки на nan перед обработкой
test_features['dept'] = test_features['dept'].replace(' ', np.nan)


#анализ
test_features['dept'].unique()


#анализ
test_features['level'].unique()



#анализ
test_features['workload'].unique()

# заменим пустые строки на nan перед обработкой
test_features['workload'] = test_features['workload'].replace(' ', np.nan)



#анализ
test_features['workload'].unique()



#анализ
test_features['employment_years'].unique()




#анализ
test_features['last_year_promo'].unique()


#анализ
test_features['last_year_violations'].unique()

#анализ
test_features['supervisor_evaluation'].unique()


# ## Задача 1 Шаг 3. Исследовательский анализ данных

# ### Задача 1 Шаг 3.1 Train


#разброс значений
pd.set_option('display.max_columns', None)
train_job_satisfaction_rate.describe()

#построение гистограмм

train_job_satisfaction_rate.hist(figsize=(10, 10)
        , bins = 100
       );

plt.suptitle("Гистограмма распределения признаков train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


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

#countplot
sns.countplot(train_job_satisfaction_rate['employment_years']
             );

plt.suptitle("Распределение employment_years из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

#countplot
sns.countplot(train_job_satisfaction_rate['supervisor_evaluation']
             );
plt.suptitle("Распределение supervisor_evaluation из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


#countplot
sns.countplot(train_job_satisfaction_rate['last_year_promo']
             );
plt.suptitle("Распределение last_year_promo из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

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


#countplot
sns.countplot(train_job_satisfaction_rate['last_year_violations']
             );
plt.suptitle("Распределение last_year_violations из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# корреляционный анализ количесвенных признаков по пирсону

sns.heatmap(train_job_satisfaction_rate.drop(['id'], axis=1).corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0);

plt.suptitle("Корреляционный анализ количесвенных признаков по Пирсону из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# корреляционный анализ количесвенных признаков по spearman
corr_spearman = train_job_satisfaction_rate.drop(['id'], axis=1).corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='coolwarm', center=0);

plt.suptitle("Корреляционный анализ количесвенных признаков по Спирмену из train_job_satisfaction_rate", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# корреляционный анализ количественных  и качественных признаков
interval_cols = ['salary'
                 , 'job_satisfaction_rate']
corr_phik = train_job_satisfaction_rate.drop(['id'], axis=1).phik_matrix(interval_cols = interval_cols)


sns.heatmap(corr_phik, annot=True, fmt='.2f', cmap='coolwarm', center=0);

plt.suptitle("Корреляционный анализ  количественных  и качественных признаков по Фи из train_job_satisfaction_rate", fontsize=16, y=1.02);
plt.tight_layout()
plt.show();

#разброс значений
#pd.set_option('display.max_columns', None)
test_features.describe()


#разброс значений
#pd.set_option('display.max_columns', None)
test_target_job_satisfaction_rate.describe()


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


#построение гистограмм


test_features.hist(figsize=(10, 10)
        , bins = 100
       );

plt.suptitle("Гистограмма распределения признаков test_features", fontsize=16, y=1.02)

test_target_job_satisfaction_rate.hist(figsize=(10, 10)
        , bins = 100
       );

plt.suptitle("Гистограмма распределния признаков test_target_job_satisfaction_rate", fontsize=16, y=1.02);

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

#countplot
sns.countplot(test_features['supervisor_evaluation']
             );

plt.suptitle("Распределение supervisor_evaluation из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


#countplot
sns.countplot(test_features['employment_years']
             );

plt.suptitle("Распределение employment_years из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


#корреляционный анализ количесвенных признаков по пирсону

sns.heatmap((test_features.drop(['id'], axis=1)).corr(),annot=True,fmt='.2f', cmap='coolwarm',center=0);
plt.suptitle("Корреляционный анализ  количественных признаков по Пирсону из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# корреляционный анализ количесвенных признаков по spearman
corr_spearman=test_features.drop(['id'], axis=1).corr(method='spearman')
sns.heatmap(corr_spearman,annot=True,fmt='.2f',cmap='coolwarm',center=0);

plt.suptitle("Корреляционный анализ  количественных признаков по Спирмену из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

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

# количество строк-дубликатов в данных (должно быть ноль)
train_job_satisfaction_rate_no_id = train_job_satisfaction_rate.drop(['id'], axis=1)
train_job_satisfaction_rate_no_id.duplicated().sum()



# удаление строк-дубликатов в данных (сохранить те же индексы строк, что были у них до удаления дубликатов)
train_job_satisfaction_rate_no_id = train_job_satisfaction_rate_no_id.drop_duplicates()
train_job_satisfaction_rate_no_id.duplicated().sum()



#корреляционный анализ количественных и качественных признаков
corr_phik=train_job_satisfaction_rate_no_id.phik_matrix(interval_cols=interval_cols)
corr_phik
#sns.heatmap(corr_phik,annot=True,fmt='.2f',cmap='coolwarm',center=0);


# данные для теста
test_df = pd.merge(test_features,test_target_job_satisfaction_rate, on='id', how='left')
test_df.info() 



# проверка на дубли
test_df['id'].duplicated().sum()



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
    ,

# Добавлен RandomForest
#    {
#        'models': [RandomForestRegressor(random_state=RANDOM_STATE)],
#        'models__n_estimators': [50, 100, 200],
#        'models__max_depth': [None, 5, 10],
#        'models__min_samples_leaf': [1, 2, 4]
#    },
# Добавьте GradientBoosting  
    {
        'models': [GradientBoostingRegressor(random_state=RANDOM_STATE)],
        'models__n_estimators': [50, 100],
        'models__learning_rate': [0.01, 0.1],
        'models__max_depth': [3, 5]
    }


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


#

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









# **Выводы**
# - Лучшая модель и её параметры: DecisionTreeRegressor(max_depth=18, min_samples_split=15, random_state=42) и MinMaxScaler()           
# - SMAPE (тренировочные данные): 11.399947748557818
# - SMAPE (тестовые данные): 14.006414402065234
# 



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




# вывод пяти случайных строк
train_quit.sample(5)





# вывод пяти случайных строк
test_features.sample(5)




# вывод пяти случайных строк
test_target_quit.sample(5) 





#размер таблиц
         
print(f'Размер таблицы train_quit: { train_quit.shape}')  
print(f'Размер таблицы test_features: { test_features.shape}')  
print(f'Размер таблицы test_target_quit: { test_target_quit.shape}')  


#   **Выводы**
#   
#  - 3 таблицы train_quit,  test_features, test_target_quit были пролиты

# ## Задача 2 Шаг 2. Предобработка данных




#основная информация о датафрейме
train_quit.info()





#основная информация о датафрейме
test_features.info()





#основная информация о датафрейме
test_target_quit.info()





#анализ
train_quit.isna().sum()





#анализ
test_features.isna().sum()




#уникальные значения
test_features['dept'].unique()




# заменим пустые строки на nan перед обработкой
test_features['dept'] = test_features['dept'].replace(' ', np.nan)




#уникальные значения
test_features['dept'].unique()




#уникальные значения
test_features['level'].unique()





#анализ
test_target_quit.isna().sum()




#исправить грамматическую ошибку в sinior.
test_features.replace("sinior", "senior", inplace=True)
train_quit.replace("sinior", "senior", inplace=True)
test_target_quit.replace("sinior", "senior", inplace=True)


# **Комментарий**
# 
# - есть пропуски (значения nan, " "), обработка пропусков выполнена на этапе pipeline: пропущенные значения в категориальных признаках заполняются самым частотным значением ('most_frequent'), а числовые - меданным значениями ('median')



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



# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_quit_train = train_quit.pivot_table(index='quit', values='id', aggfunc='count')
pivot_quit_train.columns = ['count']
pivot_quit_train['ratio']= pivot_quit_train['count']/train_quit['id'].shape[0]
pivot_quit_train.sort_values(by='ratio', ascending=False)


#countplot
sns.countplot(train_quit['employment_years']
             );
plt.suptitle("Распределение employment_years из train_quit ", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()



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



# корреляционный анализ количесвенных признаков по пирсону
sns.heatmap(train_quit.corr(), annot=True, fmt='.2f',  cmap='coolwarm', center=0);
plt.suptitle("Корреляционный анализ количесвенных признаков по Пирсону из train_quit", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()



# корреляционный анализ количественных признаков по spearman
corr_spearman = train_quit.drop(['id'], axis=1).corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='coolwarm', center=0);
plt.suptitle("Корреляционный анализ количесвенных признаков по Спирмену из train_quit", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# корреляционный анализ количественных  и качественных признаков
train_quit.drop(['id'], axis=1).phik_matrix(interval_cols=interval_cols)


# #### Задача 2 Шаг 3.1 Test



#разброс значений
#pd.set_option('display.max_columns', None)

test_features.drop(['id'], axis=1).describe()




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


#countplot
sns.countplot(test_target_quit['quit']
                           );
plt.suptitle("Распределение quit из test_target_quit", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()



# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_quit_test = test_target_quit.pivot_table(index='quit', values='id', aggfunc='count')
pivot_quit_test.columns = ['count']
pivot_quit_test['ratio']= pivot_quit_test['count']/test_target_quit['id'].shape[0]
pivot_quit_test.sort_values(by='ratio', ascending=False)



# корреляционный анализ количесвенных признаков по пирсону
sns.heatmap(test_features.drop(['id'], axis=1).corr(), annot=True, fmt='.2f',  cmap='coolwarm', center=0);
plt.suptitle("Корреляционный анализ количесвенных признаков по Пирсону из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()




# корреляционный анализ количесвенных признаков по spearman
corr_spearman = test_features.corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='coolwarm', center=0);
plt.suptitle("Корреляционный анализ количесвенных признаков по Спирмену из test_features", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()



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



# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit = train_quit.pivot_table(index=['dept','quit'], values='id', aggfunc='count')
pivot_train_quit.columns = ['count']
pivot_train_quit['ratio']= pivot_train_quit['count']/train_quit['id'].shape[0]
pivot_train_quit.sort_values(by='ratio', ascending=False)



df1=train_quit.query('quit =="no"')
df1['salary'].mean()



# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit = train_quit.pivot_table(index='quit', values='salary', aggfunc='mean')
pivot_train_quit.columns = ['mean']

df1=train_quit.query('quit =="no"')
mean_no=df1['salary'].mean()


pivot_train_quit['growth_rate']= (pivot_train_quit['mean']/mean_no - 1)
pivot_train_quit.sort_values(by='mean', ascending=False)



# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit = train_quit.pivot_table(index=['dept','quit'], values='salary', aggfunc='mean')
pivot_train_quit.columns = ['mean']
pivot_train_quit.sort_values(by='mean', ascending=False)



#фрейм только по уволившимся 
train_quit_yes = train_quit.query('quit=="yes"')


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='dept', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)



# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='level', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)



# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='workload', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)



# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='last_year_promo', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)


# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='last_year_violations', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)



# анализ значений данных
#pd.set_option('display.max_rows', None)
pivot_train_quit_yes = train_quit_yes.pivot_table(index='employment_years', values='quit', aggfunc='count')
pivot_train_quit_yes.columns = ['count']
pivot_train_quit_yes['ratio']= pivot_train_quit_yes['count']/train_quit_yes['id'].shape[0]
pivot_train_quit_yes.sort_values(by='ratio', ascending=False)



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



# ### Задача 2 3.3. Распределения признака job_satisfaction_rate для ушедших и оставшихся сотрудников.



#объединение таблиц  train_quit и satisfaction_rate

df_sr = test_target_job_satisfaction_rate.loc[:, ['id' , 'job_satisfaction_rate']]


data_quit_sr = pd.merge(test_target_quit, df_sr, on='id', how='left')

data_quit_sr_no = data_quit_sr.query('quit == "no"')
data_quit_sr_yes = data_quit_sr.query('quit == "yes"')




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



# Добавляем предсказания модели в исходный датасет
train_quit['predicted_job_satisfaction'] = randomized_search.predict(train_quit.drop(['quit'], axis=1))
train_quit.info()




test_features.info()



# Добавляем предсказания модели в исходный датасет
test_features_3 = test_features.drop(['id'], axis=1)
test_features_3['predicted_job_satisfaction'] = randomized_search.predict(test_features_3)
test_features_3



# количество строк-дубликатов в данных (должно быть ноль)
train_quit_no_id = train_quit.drop(['id'], axis=1)
train_quit_no_id.duplicated().sum()




# удаление строк-дубликатов в данных (сохранить те же индексы строк, что были у них до удаления дубликатов)
train_quit_no_id = train_quit_no_id.drop_duplicates()
train_quit_no_id.duplicated().sum()

:


#interval_cols
interval_cols2 = ['salary'
                 , 'predicted_job_satisfaction']




#корреляционный анализ количественных и качественных признаков
corr_phik=test_features_3.phik_matrix(interval_cols=interval_cols2)
corr_phik



#корреляционный анализ количественных и качественных признаков
corr_phik=train_quit_no_id.phik_matrix(interval_cols=interval_cols2)
corr_phik


# **Выводы**
#  
# - job_satisfaction_rate, предсказанный лучшей моделью первой задачи, добавлен к входным признакам второй задачи в df_2
# - из датафрейма для будущего обучения удален id, удалены дубликаты и проведен анализ корреляции
# - новый признак не создаёт мультиколлинеарности с уже существующими признаками.

# ## Задача 2 Шаг 5. Подготовка данных


# Добавляем предсказания модели в исходный датасет
test_features_4 = test_features
test_features_4['predicted_job_satisfaction'] = randomized_search.predict(test_features)
test_features_4



# данные для теста
test_df2 = pd.merge(test_features,test_target_quit, on='id', how='left')
test_df2.info()




# пайплайн, который выберет лучшую комбинацию модели и гиперпараметров. 

RANDOM_STATE = 42


# загружаем данные
X_train = train_quit_no_id.drop(['quit' 
                                ], axis=1)

X_test = test_df2.drop(['quit'], axis=1)
y_train = train_quit_no_id['quit']
y_test = test_df2['quit']


X_train.shape, X_test.shape,y_train.shape, y_test.shape


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


