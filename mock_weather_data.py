import pandas as pd
from datetime import datetime, timedelta
import random

# Definiamo il range: il mese di maggio 2025
start_date = datetime(2025, 5, 1, 0)
end_date = datetime(2025, 6, 1, 0)
data_full_month = []
conditions = ['clear', 'cloud', 'rain']
current_day = start_date

while current_day < end_date:
    day_start = current_day
    # 50% chance per giornata fissa o mista
    if random.random() < 0.5:
        # Giornata con condizione fissa
        condition = random.choice(conditions)
        rain = 1 if condition == "rain" else 0
        for h in range(24):
            hour_time = day_start + timedelta(hours=h)
            data_full_month.append((hour_time.strftime("%Y-%m-%d %H:%M:%S"), rain, condition))
    else:
        # Giornata mista: ogni ora viene assegnata casualmente
        for h in range(24):
            hour_time = day_start + timedelta(hours=h)
            condition = random.choice(conditions)
            rain = 1 if condition == "rain" else 0
            data_full_month.append((hour_time.strftime("%Y-%m-%d %H:%M:%S"), rain, condition))
    current_day += timedelta(days=1)

df_full_month = pd.DataFrame(data_full_month, columns=["datetime", "rain", "weather_condition"])

# Salva il file in una directory a cui hai accesso, per esempio nella directory corrente
output_path = "data/weather.csv"
df_full_month.to_csv(output_path, index=False)
print(f"File salvato come {output_path}")
