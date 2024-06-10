import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate random data for 1 year
start_date = datetime(2023, 1, 1)
end_date = start_date + timedelta(days=365)
dates = pd.date_range(start_date, end_date, freq='h')

# Create the DataFrame
df = pd.DataFrame({
    'date': dates,
    'temperature': np.random.uniform(10, 30, len(dates)),
    'humidity': np.random.uniform(30, 90, len(dates)),
    'wind_speed': np.random.uniform(5, 20, len(dates)),
    'weather': np.random.choice(['rain', 'sunny', 'windy'], len(dates))
})

# Save the data to an Excel file
df.to_excel('sample_weather_data.xlsx', index=False)