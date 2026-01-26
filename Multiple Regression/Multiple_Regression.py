import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Mul_Reg.csv")


X = df[["x1", "x2"]]   
y = df["y"]            


model = LinearRegression()
model.fit(X, y)


print("Intercept (b0):", model.intercept_)
print("Coefficients (b1 and b2):", model.coef_)


x1_new = 3
x2_new = 2

y_pred = model.predict([[x1_new, x2_new]])

print("Predicted y:", y_pred[0])
