import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
trainset = pd.read_csv("C:/Users/Parth/Desktop/Github Project/train.csv")
testset = pd.read_csv("C:/Users/Parth/Desktop/Github Project/test.csv")
sns.pairplot(trainset)
y_train = trainset['price']
X_train=trainset[['calculated_host_listings_count','reviews_per_month','number_of_reviews','minimum_nights']]
lm = LinearRegression()
lm.fit(X_train,y_train)
X_test = testset[['calculated_host_listings_count','reviews_per_month','number_of_reviews','minimum_nights']]
predictions = lm.predict(X_test)
output=pd.DataFrame(data={"id":testset["id"],"price":predictions})
output.to_csv("C:/Users/Parth/Desktop/Github Project/prediction.csv",index=False,quoting=3,sep=',')
