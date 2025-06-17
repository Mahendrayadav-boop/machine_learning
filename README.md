# machine_learning
Machine learning 
# pricing_strategy_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load sample data (create your own or use real retail sales data)
df = pd.read_csv('data/pricing_data.csv')  # Columns: ['price', 'units_sold', 'day_of_week', 'discount', 'season']

# Feature engineering
df['revenue'] = df['price'] * df['units_sold']

# Define features and target
X = df[['price', 'day_of_week', 'discount', 'season']]
y = df['units_sold']  # Target: how many units sold at a given price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")

# Price optimization simulation
price_range = np.linspace(5, 50, 100)
sim_data = pd.DataFrame({
    'price': price_range,
    'day_of_week': 3,  # Assume Wednesday
    'discount': 0,
    'season': 1        # Assume in-season
})
sim_units = model.predict(sim_data)
sim_revenue = sim_data['price'] * sim_units

# Plot price vs predicted revenue
plt.figure(figsize=(10,6))
plt.plot(price_range, sim_revenue, label='Predicted Revenue', color='green')
plt.title('Optimal Pricing Strategy')
plt.xlabel('Price')
plt.ylabel('Predicted Revenue')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pricing_optimization_plot.png")
plt.show()

# Optimal price
optimal_price = price_range[np.argmax(sim_revenue)]
print(f" Optimal Price to Maximize Revenue: ${optimal_price:.2f}")

