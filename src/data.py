"""
Data Engine for Crypto Filter
Handles fetching from Binance Futures and local caching via Parquet.
"""
import requests
import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Union, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import from config
from src.config import MANDATORY_CRYPTO, BENCHMARK_SYMBOL, TIMEZONE_OFFSET, IGNORED_CRYPTO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceFuturesFetcher:
    """Fetches data from Binance Perpetual Futures (USDT-M)"""
    
    BASE_URL = "https://fapi.binance.com"
    TICKER_24H = "/fapi/v1/ticker/24hr"
    KLINES = "/fapi/v1/klines"
    EXCHANGE_INFO = "/fapi/v1/exchangeInfo"
    ORDERBOOK = "/fapi/v1/depth"
    
    def __init__(self, rate_limit_delay: float = 0.05):
        self.session = requests.Session()
        self.rate_limit_delay = rate_limit_delay

    def _request(self, endpoint: str, params: dict = None) -> dict:
        try:
            url = f"{self.BASE_URL}{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Request failed {endpoint}: {e}")
            return {}

    def get_orderbooks(self, symbol: str) -> pd.DataFrame:
        """Get orderbook for a single symbol"""
        data = self._request(self.ORDERBOOK, {'symbol': symbol, 'limit': 1000})
        if not data: return pd.DataFrame()
        
        bids = pd.DataFrame(data.get('bids', []), columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(data.get('asks', []), columns=['price', 'qty'], dtype=float)
        
        bids['side'] = 'bid'
        asks['side'] = 'ask'
        return pd.concat([bids, asks])

    def get_books_status(self, symbol: str, impact_usd: float = 1000, imbalance_pct: float = 0.05):
        """
        Calculate impact spread and orderbook imbalance.
        impact_usd: USD size to calculate average execution price.
        imbalance_pct: range around mid-price to calculate volume imbalance.
        """
        df = self.get_orderbooks(symbol)
        if df.empty: return {}

        bids = df[df["side"] == "bid"].sort_values("price", ascending=False)
        asks = df[df["side"] == "ask"].sort_values("price", ascending=True)
        if bids.empty or asks.empty: return {}

        mid = (bids.iloc[0]["price"] + asks.iloc[0]["price"]) / 2

        def calc_impact_price(book, target_usd):
            notional = 0.0
            total_qty = 0.0
            for _, row in book.iterrows():
                p, q = row["price"], row["qty"]
                v = p * q
                if notional + v >= target_usd:
                    needed_v = target_usd - notional
                    total_qty += needed_v / p
                    notional = target_usd
                    break
                total_qty += q
                notional += v
            return target_usd / total_qty if notional >= target_usd and total_qty > 0 else None

        avg_ask = calc_impact_price(asks, impact_usd)
        avg_bid = calc_impact_price(bids, impact_usd)
        
        impact_spread = (avg_ask - avg_bid) / mid if (avg_ask and avg_bid) else None
        
        # Imbalance
        lower = mid * (1 - imbalance_pct)
        upper = mid * (1 + imbalance_pct)
        bid_liq = bids[bids.price >= lower]["qty"].sum() * mid
        ask_liq = asks[asks.price <= upper]["qty"].sum() * mid
        total_liq = bid_liq + ask_liq
        imbalance = (bid_liq - ask_liq) / total_liq if total_liq > 0 else None

        return {
            "mid_price": mid,
            "impact_spread": impact_spread,
            "orderbook_imbalance": imbalance,
            "bid_liquidity": bid_liq,
            "ask_liquidity": ask_liq
        }

    def get_top_volume_symbols(self, top_n: int = 50, exclude: List[str] = None) -> List[str]:
        """Get top N symbols by 24h Quote (USDT) Volume"""
        if exclude is None:
            exclude = ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT', 'USTUSDT', 'FDUSDUSDT']
            
        # Add IGNORED_CRYPTO to exclude list
        if IGNORED_CRYPTO:
            exclude = list(set(exclude + IGNORED_CRYPTO))
            
        data = self._request(self.TICKER_24H)
        if not data: return []
        
        df = pd.DataFrame(data)
        df['quoteVolume'] = pd.to_numeric(df['quoteVolume'])
        df = df.sort_values('quoteVolume', ascending=False)
        
        top_symbols = []
        for _, row in df.iterrows():
            sym = row['symbol']
            if sym.endswith('USDT') and sym not in exclude:
                top_symbols.append(sym)
                if len(top_symbols) >= int(top_n):
                    break
        return top_symbols

    def get_all_symbols(self) -> List[str]:
        """Get all valid USDT perpetual symbols"""
        info = self._request(self.EXCHANGE_INFO)
        if not info or 'symbols' not in info:
            return []
            
        symbols = []
        for s in info['symbols']:
            sym = s['symbol']
            if sym.endswith('USDT') and s['status'] == 'TRADING':
                if sym not in IGNORED_CRYPTO:
                    symbols.append(sym)
        return sorted(symbols)

    def fetch_klines(self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = None) -> pd.DataFrame:
        """Fetch klines for a single symbol"""
        params = {'symbol': symbol, 'interval': interval}
        
        if start_time: params['startTime'] = start_time
        if end_time: params['endTime'] = end_time
        if limit: params['limit'] = min(limit, 1500)
        
        raw = self._request(self.KLINES, params)
        if not raw: return pd.DataFrame()
        
        # standard binance kline columns
        cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore']
        
        df = pd.DataFrame(raw, columns=cols)
        
        # Optimize types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        for c in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(np.float64)
            
        df = df.sort_values('open_time')
        return df[['open_time', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']]

    def fetch_history(self, 
                      symbol: str, 
                      interval: str, 
                      years: float = 0,
                      start_time: Union[str, int, datetime, None] = None, 
                      end_time: Union[str, int, datetime, None] = None, 
                      limit: int = 1500) -> pd.DataFrame:
        """
        Fetch full history by chunking with retries.
        Supports flexible time parameters similar to fetch_binance_data.
        """
        # Calculate start_ts and end_ts in milliseconds
        if start_time is None and end_time is None:
            # use years if provided, else default to some recent period
            days = int(365 * years) if years > 0 else 30
            start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            end_ts = int(datetime.now().timestamp() * 1000)
        else:
            # end_time defaults to now if not provided
            if end_time is None:
                end_ts = int(datetime.now().timestamp() * 1000)
            elif isinstance(end_time, (str, datetime)):
                end_ts = int(pd.to_datetime(end_time).timestamp() * 1000)
            else:
                end_ts = int(end_time)

            # start_time calculation
            if isinstance(start_time, (str, datetime)):
                start_ts = int(pd.to_datetime(start_time).timestamp() * 1000)
            elif start_time is not None:
                start_ts = int(start_time)
            else:
                # fall back to years if start_time is None
                days = int(365 * years) if years > 0 else 30
                start_ts = int((pd.to_datetime(end_ts, unit='ms') - timedelta(days=days)).timestamp() * 1000)

        all_dfs = []
        current_start = start_ts
        max_retries = 3
        
        while current_start < end_ts:
            df = pd.DataFrame()
            for attempt in range(max_retries):
                try:
                    df = self.fetch_klines(symbol, interval, start_time=current_start, end_time=end_ts, limit=limit)
                    if not df.empty:
                        break
                except Exception as e:
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {symbol} due to: {e}")
                    time.sleep(1)
            
            if df.empty:
                break
                
            all_dfs.append(df)
            last_ts = int(df.iloc[-1]['open_time'].timestamp() * 1000)
            
            if last_ts >= end_ts or len(df) < limit:
                break
            
            current_start = last_ts + 1  # ensure forward progress
            time.sleep(self.rate_limit_delay)
            
        if not all_dfs:
            return pd.DataFrame()
            
        final_df = pd.concat(all_dfs).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
        return final_df

    def fetch_candles(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch exactly N candles by chunking backwards from current time.
        """
        all_dfs = []
        current_end = int(datetime.now().timestamp() * 1000)
        remaining = limit
        
        while remaining > 0:
            fetch_limit = min(remaining, 1500)
            df = self.fetch_klines(symbol, interval, end_time=current_end, limit=fetch_limit)
            
            if df.empty:
                break
            
            all_dfs.append(df)
            remaining -= len(df)
            
            # Move current_end to the oldest candle's open time - 1ms
            current_end = int(df['open_time'].min().timestamp() * 1000) - 1
            
            if len(df) < fetch_limit:
                break
                
            time.sleep(self.rate_limit_delay)
            
        if not all_dfs:
            return pd.DataFrame()
            
        final_df = pd.concat(all_dfs).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
        return final_df.tail(limit)


class DataManager:
    """Manages data storage and caching"""
    
    def __init__(self, data_dir: str = "data_cache", cache_size: int = 50):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._cache = {}
        self._cache_size = cache_size
        self.fetcher = BinanceFuturesFetcher()
        self._last_sync = {}
        self._universe_cache = None
        self._last_universe_fetch = 0
        
    def get_universe(self, top_n: int = 1000) -> List[str]:
        """Returns top N symbols by volume, cached for 10 minutes session-wide."""
        now = time.time()
        if self._universe_cache is not None and (now - self._last_universe_fetch < 600):
            return self._universe_cache
            
        try:
            # Fetch absolute all or up to 1000
            syms = self.fetcher.get_top_volume_symbols(top_n=top_n)
            self._universe_cache = syms
            self._last_universe_fetch = now
            return syms
        except Exception as e:
            logger.error(f"Failed to fetch universe: {e}")
            return MANDATORY_CRYPTO # Fallback
        
    def _get_path(self, symbol: str, interval: str) -> str:
        return os.path.join(self.data_dir, f"{symbol}_{interval}.parquet")
    
    def save_data(self, data: pd.DataFrame, symbol: str, interval: str):
        if data.empty: return
        path = self._get_path(symbol, interval)
        data.to_parquet(path, index=False)
        # Invalidate cache on save
        self.clear_cache(symbol, interval)
        
    def append_data(self, symbol: str, interval: str, new_data: pd.DataFrame):
        """Merges new data with existing cache, removes duplicates and saves."""
        if new_data.empty: return
        
        path = self._get_path(symbol, interval)
        if not os.path.exists(path):
            self.save_data(new_data, symbol, interval)
            return
            
        existing = pd.read_parquet(path)
            
        # Merge and deduplicate
        combined = pd.concat([existing, new_data])
        combined = combined.drop_duplicates(subset=['open_time'], keep='last')
        combined = combined.sort_values('open_time').reset_index(drop=True)
        
        self.save_data(combined, symbol, interval)

    def get_cache_range(self, symbol: str, interval: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Returns (first_ts, last_ts) from the cached file without loading the full set."""
        path = self._get_path(symbol, interval)
        if not os.path.exists(path):
            return None, None
        try:
            # Efficiently read just the open_time column
            df = pd.read_parquet(path, columns=['open_time'])
            if df.empty: return None, None
            return df['open_time'].min(), df['open_time'].max()
        except Exception as e:
            logger.error(f"Error reading cache range for {symbol}_{interval}: {e}")
            return None, None

    def _sync_data(self, symbol: str, interval: str):
        """Automatically fetch and append missing data up to the current time."""
        if not interval or interval == 'None':
            # Skip if interval is missing or a string 'None'
            return

        cache_key = f"{symbol}_{interval}"
        now = time.time()
        
        path = self._get_path(symbol, interval)
        if hasattr(self, '_last_sync') and cache_key in self._last_sync:
            if os.path.exists(path) and (now - self._last_sync[cache_key] < 60):
                # Debounced
                return
                
        if not hasattr(self, '_last_sync'):
            self._last_sync = {}

        try:
            first_ts, last_ts = self.get_cache_range(symbol, interval)
            end_t = datetime.now()
            
            if not first_ts:
                # No data: fetch 30 days
                logger.info(f"Auto-sync: Fetching initial 30 days for {symbol} ({interval})")
                requested_start = end_t - timedelta(days=30)
                df = self.fetcher.fetch_history(symbol, interval, start_time=requested_start, end_time=end_t)
                if not df.empty:
                    self.save_data(df, symbol, interval)
                    self._last_sync[cache_key] = now
            else:
                # Incremental forward gap (start from last_ts to update the last candle and fetch new ones)
                start_ts_ms = int(pd.Timestamp(last_ts).tz_localize('UTC').timestamp() * 1000)
                df_fwd = self.fetcher.fetch_history(symbol, interval, start_time=start_ts_ms, end_time=end_t)
                if not df_fwd.empty and len(df_fwd) > 1:
                    logger.info(f"Auto-sync: Fetched {len(df_fwd)} new candles for {symbol} ({interval})")
                    self.append_data(symbol, interval, df_fwd)
                    self._last_sync[cache_key] = now
        except Exception as e:
            logger.error(f"Error auto-syncing {symbol}_{interval}: {e}")

    def load_data(self, symbol: str, interval: str, auto_sync: bool = True) -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{interval}"
        path = self._get_path(symbol, interval)
        
        if auto_sync:
            self._sync_data(symbol, interval)
            
        # Check if file exists first
        if not os.path.exists(path):
            return None
            
        # Check cache
        if cache_key in self._cache:
            # Check if file has been modified since cached
            mtime = os.path.getmtime(path)
            if self._cache[cache_key]['mtime'] >= mtime:
                return self._cache[cache_key]['data']
        
        # Load from disk
        try:
            df = pd.read_parquet(path)
            if not df.empty and 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time']) + timedelta(hours=TIMEZONE_OFFSET)
                if 'close_time' in df.columns:
                    df['close_time'] = pd.to_datetime(df['close_time']) + timedelta(hours=TIMEZONE_OFFSET)
            # Simple LRU: if full, remove oldest (first inserted in dict)
            if len(self._cache) >= self._cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                
            self._cache[cache_key] = {
                'data': df,
                'mtime': os.path.getmtime(path)
            }
            return df
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    def clear_cache(self, symbol: Optional[str] = None, interval: Optional[str] = None):
        if symbol and interval:
            cache_key = f"{symbol}_{interval}"
            self._cache.pop(cache_key, None)
        else:
            self._cache.clear()
            
    def delete_data(self, interval: str = "ALL"):
        """Deletes cached files for a specific interval or all data."""
        files = os.listdir(self.data_dir)
        deleted_count = 0
        for f in files:
            if not f.endswith(".parquet"):
                continue
            
            should_delete = False
            if interval == "ALL":
                should_delete = True
            else:
                # format: SYMBOL_INTERVAL.parquet
                name = f.replace(".parquet", "")
                parts = name.split('_')
                if len(parts) >= 2 and parts[-1] == interval:
                    should_delete = True
            
            if should_delete:
                try:
                    os.remove(os.path.join(self.data_dir, f))
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting {f}: {e}")
        
        self.clear_cache()
        return deleted_count
        
    def get_existing_symbols(self) -> List[str]:
        # Legacy support or simple list
        return list(self.get_inventory().keys())

    def get_inventory(self) -> Dict[str, List[str]]:
        """Returns dictionary of {symbol: [interval, interval...]}"""
        files = os.listdir(self.data_dir)
        inventory = {}
        for f in files:
            if f.endswith(".parquet"):
                # format: SYMBOL_INTERVAL.parquet
                # Note: symbol might contain underscores? content usually doesn't.
                # safely split from right?
                # standard is SYMBOL_INTERVAL.parquet
                name = f.replace(".parquet", "")
                parts = name.split('_')
                if len(parts) >= 2:
                    interval = parts[-1]
                    symbol = "_".join(parts[:-1]) # join rest in case symbol has _
                    
                if symbol not in inventory:
                    inventory[symbol] = []
                inventory[symbol].append(interval)
        return inventory

    def get_cache_metadata(self) -> List[Dict]:
        """Returns detailed metadata for all cached files"""
        files = os.listdir(self.data_dir)
        metadata = []
        for f in files:
            if f.endswith(".parquet"):
                try:
                    path = os.path.join(self.data_dir, f)
                    # Parse name
                    name = f.replace(".parquet", "")
                    parts = name.split('_')
                    if len(parts) >= 2:
                        interval = parts[-1]
                        symbol = "_".join(parts[:-1])
                        
                        # Read minimal data (just open_time)
                        df = pd.read_parquet(path, columns=['open_time'])
                        if not df.empty:
                            start = df['open_time'].min()
                            end = df['open_time'].max()
                            count = len(df)
                            metadata.append({
                                'Symbol': symbol,
                                'Interval': interval,
                                'Start Date': start,
                                'End Date': end,
                                'Count': count
                            })
                except Exception as e:
                    logger.error(f"Error reading metadata for {f}: {e}")
        return metadata
