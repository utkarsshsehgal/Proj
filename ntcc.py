import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error




df = pd.read_excel("C:\\Users\\utkarssh sehgal\\Desktop\\New folder (2)\\Sales Operations for Economic Cities Properties 2016-2022.xlsx")

print(df.head())
print(df.info())
print(df.describe())
print(df.shape)


# Average contract value by district
average_contract_value = df.groupby('District Name')['Contract value'].mean()
print("Average Contract Value by District:")
print(average_contract_value)
print()

# Total contract value by economic city
total_contract_value_by_city = df.groupby('Economic city')['Contract value'].sum()
print("Total Contract Value by Economic City:")
print(total_contract_value_by_city)
print()

# Average contract value by land use
average_contract_value_by_land_use = df.groupby('Land use')['Contract value'].mean()
print("Average Contract Value by Land Use:")
print(average_contract_value_by_land_use)
print()

# Comparing contract values between different economic cities
plt.figure(figsize=(10, 6))
sns.barplot(x='Economic city', y='Contract value', data=df)
plt.xlabel('Economic City')
plt.ylabel('Contract Value')
plt.title('Contract Value by Economic City')
plt.xticks(rotation=45)
plt.show()

# Comparing contract values between different land use categories
plt.figure(figsize=(10, 6))
sns.barplot(x='Land use', y='Contract value', data=df)
plt.xlabel('Land Use')
plt.ylabel('Contract Value')
plt.title('Contract Value by Land Use')
plt.xticks(rotation=45)
plt.show()

# Average contract value by district
plt.figure(figsize=(10, 6))
sns.barplot(x=average_contract_value.index, y=average_contract_value.values)
plt.xlabel('District Name')
plt.ylabel('Average Contract Value')
plt.title('Average Contract Value by District')
plt.xticks(rotation=45)
plt.show()

#  Price Prediction
# Select the relevant features for price prediction
X = df[['Last activity date in years', 'Quarter', 'Property area', 'District Name']]
y = df['Contract value']

# Convert categorical variables to numeric using one-hot encoding
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualizing the Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Property area'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Property area'], y_pred, color='red', label='Predicted')
plt.xlabel('Property Area')
plt.ylabel('Contract Value')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Property area'], residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Property Area')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Distribution of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=20, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

# Actual vs. Predicted Prices by District
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['Property area'], y=y_test, hue=y_test, palette='Set1', label='Actual')
sns.scatterplot(x=X_test['Property area'], y=y_pred, hue=y_test, palette='Set2', label='Predicted')
plt.xlabel('Property Area')
plt.ylabel('Contract Value')
plt.title('Actual vs Predicted Prices by District')
plt.legend()
plt.show()

# Boxplot of Residuals by District
plt.figure(figsize=(10, 6))
sns.boxplot(x=y_test, y=residuals)
plt.xlabel('Contract Value')
plt.ylabel('Residuals')
plt.title('Residuals by District')
plt.xticks(rotation=45)
plt.show()



# Residuals vs. Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()


# Pairwise Relationships of Selected Features
selected_features = ['Last activity date in years', 'Quarter', 'Property area', 'District Name']
sns.pairplot(df[selected_features])
plt.title('Pairwise Relationships of Selected Features')
plt.show()

# Scatter plot of Actual vs. Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Property area'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Property area'], y_pred, color='red', label='Predicted')
plt.xlabel('Property Area')
plt.ylabel('Contract Value')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

