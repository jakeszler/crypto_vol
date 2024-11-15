from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json

app = FastAPI(title="Crypto Volatility API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VolatilityResponse(BaseModel):
    symbol: str
    average_vol: float
    max_vol: float
    min_vol: float
    current_vol: float
    timestamps: list
    rolling_vols: list
    prices: list

@app.get("/")
async def root():
    return {"message": "Crypto Volatility API. Use /volatility/{symbol} to get volatility metrics."}

@app.get("/volatility/{symbol}")
async def get_volatility(
    symbol: str, 
    lookback_hours: Optional[int] = 24, 
    interval: Optional[str] = '5m'
):
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        limit = 1000
        all_data = []
        current_start_ms = start_ms

        while current_start_ms < end_ms:
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'startTime': current_start_ms,
                'endTime': end_ms,
                'limit': limit
            }
            url = 'https://api.binance.com/api/v3/klines'
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            current_start_ms = int(data[-1][0]) + 300000

        if not all_data:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
            
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        df['returns'] = np.log(df['close']/df['close'].shift(1))
        
        minutes_per_interval = 5
        intervals_per_hour = 60 // minutes_per_interval
        window_size = intervals_per_hour

        df['vol_1h_bps'] = (
            df['returns'].rolling(window_size).std() * 
            np.sqrt(intervals_per_hour) * 
            10000
        )
        
        overall_vol = df['returns'].std() * np.sqrt(intervals_per_hour) * 10000

        return VolatilityResponse(
            symbol=symbol,
            average_vol=float(overall_vol),
            max_vol=float(df['vol_1h_bps'].max()),
            min_vol=float(df['vol_1h_bps'].min()),
            current_vol=float(df['vol_1h_bps'].iloc[-1]),
            timestamps=[ts.isoformat() for ts in df['timestamp']],
            rolling_vols=df['vol_1h_bps'].fillna(0).tolist(),
            prices=df['close'].tolist()
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from Binance: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))