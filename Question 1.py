import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: Load the Data
file_path = r"C:/Users/Swartza12/Downloads/Restaurant Revenue.xlsx"
data = pd.read_excel(file_path)

# Step 2: Explore the Data
print(data.head())  # Display the first few rows of the data
print(data.info())  # Summary of the data, including data types and missing values

# Step 3: Prepare the Data
X = data[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 'Average_Customer_Spending', 'Promotions', 'Reviews']]
y = data['Monthly_Revenue']

# If 'Promotions' is categorical, encode it into numerical values
# If 'Promotions' is already numerical, skip this step

# Add constant term for intercept
X = sm.add_constant(X)

# Step 4: Build the Model
model = sm.OLS(y, X).fit()

# Step 5: Predict Monthly Revenue for Rows Before Row 10
predictions_before_row_10 = model.predict(X.iloc[:9])  # Predictions for the first 9 rows

# Step 6: Evaluate Accuracy Using Rows 10 Onwards
actual_revenue_from_row_10 = y.iloc[9:]  # Actual revenue from row 10 onwards

# Calculate Mean Absolute Percentage Error (MAPE) for accuracy evaluation
mape = np.mean(np.abs((actual_revenue_from_row_10 - predictions_before_row_10) / actual_revenue_from_row_10)) * 100
accuracy = 100 - mape

print(f"Accuracy of the model: {accuracy:.2f}%")

# Step 7: Plot the Findings
plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), predictions_before_row_10, marker='o', linestyle='-', color='b', label='Predicted Revenue')
plt.plot(range(1, 10), y.iloc[:9], marker='o', linestyle='-', color='g', label='Actual Revenue')
plt.title('Predicted vs Actual Monthly Revenue')
plt.xlabel('Month')
plt.ylabel('Monthly Revenue')
plt.legend()
plt.grid(True)
plt.show()

# Step 8: Output Multiple Regression Statistics
print(model.summary())
print("Hello World")
