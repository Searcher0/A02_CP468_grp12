import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('WeatherData_Q3.csv')

# Separate data based on the rain condition
no_rain = df[df['y'] == 0]
rain = df[df['y'] == 1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(no_rain['x₁'], no_rain['x₂'], color='blue', marker='s', s=100, label='No Rain (y=0)')
plt.scatter(rain['x₁'], rain['x₂'], color='red', marker='o', s=100, label='Rain (y=1)')

# Add labels, title, and legend
plt.xlabel('Temperature (scaled 0-1)')
plt.ylabel('Humidity (scaled 0-1)')
plt.title('Weather Data: Rain Prediction')
plt.legend()
plt.grid(True)
plt.show()