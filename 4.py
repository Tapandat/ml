import pandas as pd
from statistics import mean, variance
import matplotlib.pyplot as plt

# Load the Data
df = pd.read_excel('Lab Session Data (1).xlsx', sheet_name='IRCTC Stock Price')

# Calculate Mean and Variance of Price
price_mean = mean(df['Price'])
price_variance = variance(df['Price'])

print(f"Mean of Price: {price_mean}")  
print(f"Variance of Price: {price_variance}")

# Select Wednesday Prices and Calculate Sample Mean
wednesday_prices = df.loc[df['Day'] == 'Wednesday', 'Price']
wednesday_mean = mean(wednesday_prices

def mean(x):
     return sum(x) / len(x)
 # Sample Mean of April Prices
 april_prices = df.loc[df['Month'] == 'Apr', 'Price']
 april_mean = mean(april_prices)

 print(f"April Prices' Sample Mean:  {april_mean}")
 print(f"Difference from Population Mean:  {april_mean - price_mean}")

 # Calculate Probability of making a loss
 loss_probability = sum(df['Chg%'] < 0) / len(df)
 print(f"Probability of making a loss:  {loss_probability:.2%}")

 # Probability of profit when it is Wednesday
 wednesday

# Calculate Conditional Probability of Profit Given Wednesday
conditional_profit_probability = wednesday_profit_probability
print(f"Conditional probability of profit given Wednesday: {conditional_profit_probability:.2%}")

# Create Scatter Plot of Chg% vs Day
plt.figure(figsize=(10, 6))
plt.title('Scatter Plot of Chg% vs Day')
plt.show()
