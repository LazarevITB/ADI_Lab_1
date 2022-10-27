import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

data = pd.read_csv('Data/salary-train.csv')
test = pd.read_csv('Data/salary-test-mini.csv')
data.head()
test
print(' Data length: ', len(data), '\n Test length', len(test))
