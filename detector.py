import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic order book snapshots (20,000 timesteps)
np.random.seed(42)
timestamps = pd.date_range("2023-01-01", periods=20000, freq="10ms")

def generate_order_book(n=20000):
    # Base values
    mid_price = 100.0
    spreads = np.abs(np.random.normal(0.1, 0.02, n))
    
    # Bid/Ask levels (5 levels deep)
    bids = pd.DataFrame({
        f"bid_{i}": np.round(mid_price - np.cumsum(spreads) - i*0.05, 2)
        for i in range(1, 6)
    })
    asks = pd.DataFrame({
        f"ask_{i}": np.round(mid_price + np.cumsum(spreads) + i*0.05, 2)
        for i in range(1, 6)
    })
    
    # Volume with random noise + occasional spoofing
    volumes = np.abs(np.random.lognormal(mean=3, sigma=0.5, size=(n, 10)))
    volumes[::500, 0] *= 5  # Inject spoofing (sudden bid volume spikes)
    
    return pd.concat([bids, asks], axis=1), volumes

order_book, volumes = generate_order_book()

def flag_liquidity_anomalies(order_book, volumes, window=100):
    df = order_book.copy()
    df["spread"] = df["ask_1"] - df["bid_1"]
    
    # Volume dispersion (Gini coefficient)
    gini = lambda x: 1 - 2 * (x/x.sum()).cumsum().mean()
    df["volume_gini"] = pd.Series(volumes).apply(gini, axis=1)
    
    # Spread anomalies (Z-score)
    df["spread_z"] = (df["spread"] - df["spread"].rolling(window).mean()) / df["spread"].rolling(window).std()
    
    # Flags
    df["low_liquidity_flag"] = (df["volume_gini"] > 0.7) | (df["spread_z"] > 2)
    return df

liquidity_flags = flag_liquidity_anomalies(order_book, volumes)

def detect_spoofing(volumes, threshold=4.0):
    # Check for sudden volume spikes followed by rapid withdrawal
    bid_volume_changes = np.diff(volumes[:, 0], axis=0)
    spoof_signal = (bid_volume_changes > threshold) & (np.roll(bid_volume_changes, -1) < -threshold )
    return np.where(spoof_signal)[0]

spoofing_indices = detect_spoofing(volumes)
print(f"Detected spoofing at timesteps: {spoofing_indices}")

def simulate_slippage(order_book, volumes, order_size=1000):
    # Simulate market order execution
    executed_prices = []
    for i in range(len(order_book)):
        remaining = order_size
        executed = []
        for level in range(5):
            price = order_book.iloc[i][f"ask_{level+1}"]
            available_vol = volumes[i, level+5]
            fill = min(remaining, available_vol)
            executed.append((fill, price))
            remaining -= fill
            if remaining <= 0: break
        
        avg_price = sum(f*p for f,p in executed) / order_size
        slippage = (avg_price / order_book.iloc[i]["ask_1"] - 1) * 10000  # in bps
        executed_prices.append(slippage)
    
    return executed_prices

slippage = simulate_slippage(order_book, volumes)
print(f"Max slippage: {max(slippage):.2f} bps")

plt.figure(figsize=(12, 6))
plt.plot(liquidity_flags["spread_z"], label="Spread Z-Score")
plt.scatter(spoofing_indices, liquidity_flags["spread_z"].iloc[spoofing_indices], 
            c='red', label="Spoofing Detected")
plt.axhline(y=2, color='r', linestyle='--', label="Anomaly Threshold")
plt.legend()
plt.title("Liquidity Anomalies & Spoofing Detection")
plt.show()

import websockets
import json
import asyncio

async def stream_order_book(symbol='BTCUSDT', depth=5):
    uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth{depth}@100ms"
    async with websockets.connect(uri) as ws:
        while True:
            data = json.loads(await ws.recv())
            bids = pd.DataFrame(data['bids'], columns=['price', 'quantity'], dtype=float)
            asks = pd.DataFrame(data['asks'], columns=['price', 'quantity'], dtype=float)
            yield bids, asks  # Real-time bid/ask DataFrames

# Usage:
async def process_book():
    async for bids, asks in stream_order_book():
        liquidity_metrics = calculate_liquidity(bids, asks)  # See Step 2


def calculate_liquidity(bids, asks):
    metrics = {
        'spread': asks['price'].iloc[0] - bids['price'].iloc[0],
        'mid_price': (asks['price'].iloc[0] + bids['price'].iloc[0]) / 2,
        'order_book_imbalance': (
            (bids['quantity'].sum() - asks['quantity'].sum()) / 
            (bids['quantity'].sum() + asks['quantity'].sum())
        ),
        'vwap_bid': (bids['price'] * bids['quantity']).sum() / bids['quantity'].sum(),
        'vwap_ask': (asks['price'] * asks['quantity']).sum() / asks['quantity'].sum(),
        'liquidity_ratio': bids['quantity'].iloc[0] / asks['quantity'].iloc[0]
    }
    return metrics

from pykalman import KalmanFilter

def kalman_tracking(observed_values):
    kf = KalmanFilter(
        initial_state_mean=observed_values[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01
    )
    state_means, _ = kf.filter(observed_values)
    return state_means

# Apply to spread monitoring:
spread_series = liquidity_flags["spread"].values
dynamic_threshold = kalman_tracking(spread_series) + 2 * np.std(spread_series)

from datetime import datetime

def align_microsecond_events(book_data, trade_data):
    # Use precise timestamps (nanosecond precision)
    book_data['timestamp'] = pd.to_datetime(book_data['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    trade_data['timestamp'] = pd.to_datetime(trade_data['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    
    # Merge within 100μs windows
    merged = pd.merge_asof(
        trade_data.sort_values('timestamp'),
        book_data.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('100us')
    )
    return merged

def slippage_optimized_execution(bids, asks, order_size, strategy='TWAP'):
    executed_prices = []
    remaining = order_size
    
    if strategy == 'TWAP':
        # Time-Weighted Average Price (slices order over time)
        slices = np.linspace(0, order_size, 5)
        for slice_size in np.diff(slices):
            price = (asks['price'].iloc[0] + bids['price'].iloc[0]) / 2
            executed_prices.append((slice_size, price))
    
    elif strategy == 'VWAP':
        # Volume-Weighted Execution
        for _, row in asks.iterrows():
            fill = min(remaining, row['quantity'])
            executed_prices.append((fill, row['price']))
            remaining -= fill
            if remaining <= 0: break
    
    avg_price = sum(f*p for f,p in executed_prices) / order_size
    slippage_bps = (avg_price / asks['price'].iloc[0] - 1) * 10000
    return slippage_bps

import time

class LatencyMonitor:
    def __init__(self):
        self.log = []
    
    def ping(self, event_id):
        self.log.append((event_id, time.time_ns()))
    
    def report(self):
        latencies = np.diff([t for _, t in self.log])
        print(f"P99 Latency: {np.percentile(latencies, 99)/1e6:.2f}ms")

# Usage:
monitor = LatencyMonitor()
monitor.ping("order_received")
# ... processing ...
monitor.ping("order_executed")
monitor.report()

