import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Load CSV
df = pd.read_csv("Mul_Reg.csv")

# Step 2: Select features (X) and target (y)
X = df[["x1", "x2"]]   # independent variables
y = df["y"]            # dependent variable

# Step 3: Create and train model
model = LinearRegression()
model.fit(X, y)

# Show learned parameters
print("Intercept (b0):", model.intercept_)
print("Coefficients (b1 and b2):", model.coef_)

# Step 4: Predict y for a new (x1, x2)
x1_new = 3
x2_new = 2

y_pred = model.predict([[x1_new, x2_new]])

print("Predicted y:", y_pred[0])
