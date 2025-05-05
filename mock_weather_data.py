import pandas as pd
from datetime import datetime, timedelta
import random

# Monthly weather probabilities in Turin (approximate)
monthly_prob = {
    1: {"clear": 0.2, "cloud": 0.4, "rain": 0.4},
    2: {"clear": 0.3, "cloud": 0.4, "rain": 0.3},
    3: {"clear": 0.4, "cloud": 0.4, "rain": 0.2},
    4: {"clear": 0.5, "cloud": 0.3, "rain": 0.2},
    5: {"clear": 0.6, "cloud": 0.3, "rain": 0.1},
    6: {"clear": 0.7, "cloud": 0.2, "rain": 0.1},
    7: {"clear": 0.8, "cloud": 0.15, "rain": 0.05},
    8: {"clear": 0.8, "cloud": 0.15, "rain": 0.05},
    9: {"clear": 0.6, "cloud": 0.3, "rain": 0.1},
    10: {"clear": 0.4, "cloud": 0.4, "rain": 0.2},
    11: {"clear": 0.3, "cloud": 0.4, "rain": 0.3},
    12: {"clear": 0.2, "cloud": 0.4, "rain": 0.4},
}


def weighted_choice(prob_dict):
    conditions = list(prob_dict.keys())
    weights = list(prob_dict.values())
    return random.choices(conditions, weights=weights, k=1)[0]


def simulate_weather(start_date: datetime, end_date: datetime, output_path: str):
    data = []
    current_day = start_date

    while current_day < end_date:
        month = current_day.month
        probs = monthly_prob[month]

        if random.random() < 0.5:
            # Fixed day
            cond = weighted_choice(probs)
            rain = 1 if cond == "rain" else 0
            for h in range(24):
                ts = current_day + timedelta(hours=h)
                data.append((ts.strftime("%Y-%m-%d %H:%M:%S"), rain, cond))
        else:
            # Mixed day: blocks of 6-12h with same condition
            remaining_hours = 24
            ts = current_day
            while remaining_hours > 0:
                block_len = min(random.randint(3, 8), remaining_hours)
                cond = weighted_choice(probs)
                rain = 1 if cond == "rain" else 0
                for _ in range(block_len):
                    data.append((ts.strftime("%Y-%m-%d %H:%M:%S"), rain, cond))
                    ts += timedelta(hours=1)
                remaining_hours -= block_len

        current_day += timedelta(days=1)

    df = pd.DataFrame(data, columns=["datetime", "rain", "weather_condition"])
    df.to_csv(output_path, index=False)
    print(f"File meteo salvato in: {output_path}")


# Run
simulate_weather(
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2026, 1, 1),
    output_path="data/weather.csv"
)
