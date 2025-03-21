import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('WeatherData_Q3.csv')

# Separate data based on the "rain" column
no_rain = df[df['rain'] == 0]
rain = df[df['rain'] == 1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(rain['temp'], rain['humid'],
            color='red', marker='o', s=100, label='Rain')
plt.scatter(no_rain['temp'], no_rain['humid'],
            color='blue', marker='s', s=100, label='No Rain')

# Add labels, title, and legend
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Rain Data')
plt.legend()
plt.grid(True)
plt.show()
