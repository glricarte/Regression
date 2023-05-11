
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import pandas as pd

df = pd.read_csv("Housing.csv")
Y = df['price']
X = df['lotsize']
X = X.values.reshape(len(X),1)
Y = Y.values.reshape(len(Y),1)
print(df.head(10))


X_train = X[:-250]
X_test = X[-250:]
Y_train = Y[:-250]
Y_test = Y[-250:]

plt.scatter(X_test, Y_test,  color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())


regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3)
plt.show()
plt.show()