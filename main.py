import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, linear_model, model_selection, metrics

print("----------------------------------------")

df = pd.read_csv("Life_Expectancy_Data.csv")
print(df)

print("----------------------------------------")

print(df.info())

print("----------------------------------------")

print(df.describe(include='all'))

print("----------------------------------------")

df['Population'] = df['Population']/1000000
print(df['Population'])

print("----------------------------------------")

df = df.dropna(axis=0)
df.info()

print("----------------------------------------")

Turkey =df[df['Country'] == 'Turkey']
print(Turkey)

print("----------------------------------------")

plt.plot(Turkey['Year'], Turkey['Life expectancy '], color='red')
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title("Turkey")
plt.show()

print("----------------------------------------")

print(df['Status'].value_counts())

print("----------------------------------------")

plt.boxplot([df[df['Status']=='Developing']['Life expectancy '],
             df[df['Status']=='Developed']['Life expectancy ']],
             labels=['Developing','Developed'])
plt.ylabel('Life expectancy')
plt.xlabel('Countries')
plt.show()

print("----------------------------------------")

print(df.sort_values("Life expectancy "))

print("----------------------------------------")

plt.plot(df['GDP'], df['Life expectancy '])
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')
plt.show()

print("----------------------------------------")

l_encoder= preprocessing.LabelEncoder()
df.loc[:,'Status'] = l_encoder.fit_transform(df.loc[:,'Status'])
print(df['Status'].value_counts())

print("----------------------------------------")

df.loc[:, 'Country'] = l_encoder.fit_transform(df.loc[:, 'Country'])
print(df['Country'].value_counts())

print("----------------------------------------")

fig, ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(df.corr(), annot=True, ax=ax)
plt.show()

print("----------------------------------------")

df =df.drop(labels=['Income composition of resources', ' thinness 5-9 years', 
                'infant deaths', 'percentage expenditure'], axis=1)
print(df)

print("----------------------------------------")

y = df['Life expectancy ']
df = df.drop(labels='Life expectancy ', axis=1)
print(df)

print("----------------------------------------")

y = y.to_numpy(dtype='float64')
print(y)

print("----------------------------------------")

x_train, x_test, y_train, y_test = model_selection.train_test_split(df, y, test_size=0.2, random_state=42)
x_valid, x_test, y_valid, y_test = model_selection.train_test_split(x_test, y_test, test_size=0.5, random_state=42)
print(f'X_train shape -->{x_train.shape}')
print(f'X_valid shape -->{x_valid.shape}')
print(f'X_test shape -->{x_test.shape}')
print(f'y_train shape -->{y_train.shape}')
print(f'y_valid shape -->{y_valid.shape}')
print(f'y_test shape -->{y_test.shape}')

print("----------------------------------------")

scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)
print(x_train)

print("----------------------------------------")

model = linear_model.LinearRegression()
model.fit(X=x_train, y=y_train)
val_hat = model.predict(x_valid)
print(metrics.mean_squared_error(y_valid, val_hat))

print("----------------------------------------")

poi_test = model.predict(x_test)
print(metrics.r2_score(y_test, poi_test))

print("----------------------------------------")