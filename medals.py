import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
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
test["predictions"] = predictions
test.loc[test["predictions"] < 0, "predictions"] = 0
test["predictions"] = test["predictions"].round()

error = mean_absolute_error(test["medals"], test["predictions"])
test["predictions"] = predictions

sns.lmplot(x="athletes", y="predictions", data=test, fit_reg=True, ci=None)
plt.show()
