
from palmerpenguins import penguins
from pandas import get_dummies
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

## Get Data


df = penguins.load_penguins().dropna()

df.head(3)



X = get_dummies(df[['bill_length_mm', 'species', 'sex']], drop_first = True)
y = df['body_mass_g']

model = LinearRegression().fit(X, y)


## Chapter 2 lab 

from vetiver import VetiverModel, vetiver_pin_write, VetiverAPI
v = VetiverModel(model, model_name='penguin_model', prototype_data=X)

import pins

b = pins.board_folder('data/model', allow_pickle_read=True)

# VetiverModel 객체를 pins 보드에 저장
vetiver_pin_write(b, v)
del v

v = VetiverModel.from_pin(b, 'penguin_model')


app = VetiverAPI(v, check_prototype=True)
app.run(port = 8080)
