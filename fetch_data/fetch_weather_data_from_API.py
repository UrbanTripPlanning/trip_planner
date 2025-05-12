import requests
from datetime import datetime, timedelta
import pandas as pd
import ast

API_KEY = 'db90f1ec7bfc4f4b90f1ec7bfcef4b8c'
HISTORY_DAILY_URL = 'https://api.weather.com/v2/pws/history/all?'
STATION_ID = 'ITURIN3276'


def fetch_and_clean_csv(start_date, end_date, csv_filename):
    # 1) scarico tutti i record in una lista
    all_rows = []
    current = start_date
    delta = timedelta(days=1)

    while current <= end_date:
        ymd = current.strftime('%Y%m%d')
        params = {
            'apiKey': API_KEY,
            'format': 'json',
            'stationId': STATION_ID,
            'date': int(ymd),
            'units': 'm'
        }
        r = requests.get(HISTORY_DAILY_URL, params=params)
        if r.status_code != 200:
            print(f"Errore {r.status_code} per {ymd}")
            current += delta
            continue

        data = r.json()
        rows = data.get('history', {}).get('dailysummary') or data.get('observations', [])
        for rec in rows:
            rec['date'] = current.strftime('%Y-%m-%d')
        all_rows.extend(rows)

        print(f"{ymd}: {len(rows)} record")
        current += delta

    # 2) trasformo la lista in DataFrame
    df = pd.DataFrame(all_rows)

    # 3) esplodo metric (se è stringa, prima literal_eval)
    if df['metric'].dtype == object:
        df['metric'] = df['metric'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df = df.join(pd.json_normalize(df.pop('metric')))

    # 4) rinomino, imposto indice e tolgo nome indice
    df = (
        df
        .rename(columns={'obsTimeLocal': 'timestamp'})
        .set_index('timestamp')
        .rename_axis(None)
    )

    # 5) seleziono solo le colonne che mi interessano
    cols = [
        'solarRadiationHigh', 'humidityAvg',
        'tempAvg', 'windspeedAvg',
        'precipRate', 'precipTotal'
    ]
    df_final = df[cols]

    # 6) salvo il CSV “ripulito”
    df_final.to_csv(csv_filename, encoding='utf-8')
    print(f"CSV finale salvato in: {csv_filename}")


if __name__ == "__main__":
    start = datetime(2024, 1, 1).date()
    end = datetime(2024, 12, 31).date()
    fetch_and_clean_csv(start, end, 'weather_data.csv')
