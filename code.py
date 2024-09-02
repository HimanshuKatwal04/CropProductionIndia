'''Based on the Information the ultimate goal would be to predict crop production and find important insights highlighting key indicators and metrics that influence crop production.'''

#Load the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset
d=pd.read_csv("/content/Crop Production.csv")
df=pd.DataFrame(d)
df.head(6)
df.info()
df.describe()

#Check and clean the dataset 
df.isnull().sum()

'''The dataset contains 3,730 missing values in the Production column, which accounts for approximately 1.52% of the total entries. Since the percentage of missing data is relatively low, we can consider 
   a few strategies to handle it:
   Remove the missing entries: This is a straightforward approach but could result in the loss of some data.
   Impute missing values: We can fill in missing production values using statistical methods (e.g., mean, median) or more advanced techniques like regression or K-Nearest Neighbors (KNN) imputation.'''

#Let us check the distribution of Production values
sns.histplot(df['Production'].dropna(), kde=True, bins=50, color='skyblue')
plt.title('Distribution of Production Values')
plt.xlabel('Production (in metric tons)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#The graph is a right skewed graph hence now lets find the correlation of Area and Production
selected_columns=['Area','Production']
correlation_matrix = df[selected_columns].corr()
# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix for Production and Area')
plt.show()
#We see that Area and Production are positively correlated 

#So instead of dropping the Null values of Production we can imputate them using Linear Regression as the correlation is strong
train_data=df[df['Production'].notnull()]
test_data=df[df['Production'].isnull()]
x_train=train_data[['Area']]
y_train=train_data[['Production']]

model = LinearRegression()
model.fit(x_train, y_train)

X_test = test_data[['Area']]
df.loc[df['Production'].isnull(), 'Production'] = model.predict(X_test)


