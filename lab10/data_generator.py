# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters for data generation
num_records = 250000
start_date = datetime(2023, 1, 1, 0, 0, 0)
pressure_min = 980.0  # Minimum pressure value
pressure_max = 1050.0  # Maximum pressure value

# Generate a list of dates and times in increments of one minute
dates = [start_date + timedelta(minutes=i) for i in range(num_records)]
dates_formatted = [date.strftime('%Y-%m-%d') for date in dates]
times_formatted = [date.strftime('%H:%M:%S') for date in dates]

# Generate random pressure values within the specified range
pressures = np.random.uniform(pressure_min, pressure_max, num_records)

# Create a DataFrame with the specified order of columns
data = {
    'date': dates_formatted,
    'time': times_formatted,
    'pressure': pressures
}
df = pd.DataFrame(data, columns=['date', 'time', 'pressure'])  # Set column order explicitly

# Save to a CSV file
df.to_csv('pressure_data.csv', index=False)
