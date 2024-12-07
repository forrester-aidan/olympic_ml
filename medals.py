import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
teams = pd.read_csv("teams.csv")
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

teams = teams.dropna()

# split data into sets
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
reg.fit(train[predictors], train["medals"])

predictions = reg.predict(test[predictors])
print(predictions.shape)