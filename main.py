import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

data = pd.read_csv('Data/salary-train.csv')
test = pd.read_csv('Data/salary-test-mini.csv')
data.head()
test

#Проверка чтения
print(' Data length: ', len(data), '\n Test length', len(test))

#Переработка данных
def transform_data(data):
    data.FullDescription = data.FullDescription.replace('[^a-zA-Z0-9]', ' ', regex=True).str.lower()
    return data

data = transform_data(data)
test = transform_data(test)

data.head()

vectorized = TfidfVectorizer(min_df=5)
tf_idf = vectorized.fit_transform(data.FullDescription)

print(tf_idf[:5])
print('\n\n', tf_idf.shape)

data['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)

data.head()

one_hot = DictVectorizer()
oh = one_hot.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))

print(oh[:5])
print('\n\n', oh.shape)

X = hstack([tf_idf, oh])
y = data['SalaryNormalized']

print(X.shape)
print('\n\n', y[:5])