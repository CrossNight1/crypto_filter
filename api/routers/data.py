from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.data import BinanceFuturesFetcher, DataManager
from src.config import AVAILABLE_INTERVALS, MANDATORY_CRYPTO, IGNORED_CRYPTO
from datetime import datetime
import pandas as pd
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Singletons equivalent for the router
fetcher = BinanceFuturesFetcher()
manager = DataManager()

class FetchRequest(BaseModel):
    symbols: List[str]
    intervals: List[str]
    mode: str = "Range" # "Range" or "Limit"
    days_back: int = 30
    limit: int = 1000

@router.get("/universe")
def get_universe(top_n: int = 50):
    """Get top N symbols combining mandatory and top volume from Binance."""
    try:
        new_syms = fetcher.get_top_volume_symbols(top_n=top_n)
        combined = set(MANDATORY_CRYPTO).union(new_syms)
        filtered = {s for s in combined if s not in IGNORED_CRYPTO}
        return {"symbols": sorted(list(filtered))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def background_fetch_task(req: FetchRequest):
    """Background task to fetch and cache data."""
    for sym in req.symbols:
        for inter in req.intervals:
            try:
                first_ts, last_ts = manager.get_cache_range(sym, inter)
                if req.mode == "Range":
                    end_t = datetime.now()
                    requested_start = end_t - pd.Timedelta(days=req.days_back)
                    
                    end_t_utc = pd.to_datetime(end_t.timestamp(), unit='s')
                    req_start_utc = pd.to_datetime(requested_start.timestamp(), unit='s')
                    
                    dfs_to_fetch = []
                    
                    if not first_ts:
                        df_full = fetcher.fetch_history(sym, inter, start_time=requested_start, end_time=end_t)
                        if not df_full.empty:
                            dfs_to_fetch.append(df_full)
                    else:
                        # Forward Map
                        if last_ts < end_t_utc - pd.Timedelta(minutes=5):
                            start_ts_ms = int(pd.Timestamp(last_ts).tz_localize('UTC').timestamp() * 1000) + 1
                            df_fwd = fetcher.fetch_history(sym, inter, start_time=start_ts_ms, end_time=end_t)
                            if not df_fwd.empty:
                                dfs_to_fetch.append(df_fwd)
                                
                        # Backward Map
                        if first_ts > req_start_utc + pd.Timedelta(minutes=5):
                            end_ts_ms = int(pd.Timestamp(first_ts).tz_localize('UTC').timestamp() * 1000) - 1
                            df_bwd = fetcher.fetch_history(sym, inter, start_time=requested_start, end_time=end_ts_ms)
                            if not df_bwd.empty:
                                dfs_to_fetch.append(df_bwd)
                                
                    if dfs_to_fetch:
                        final_df = pd.concat(dfs_to_fetch)
                        manager.append_data(sym, inter, final_df)
                else:
                    # Limit
                    df = fetcher.fetch_candles(sym, inter, limit=req.limit)
                    if not df.empty:
                        manager.append_data(sym, inter, df)
            except Exception as e:
                logger.error(f"Error fetching {sym} {inter}: {str(e)}")

@router.post("/fetch")
def start_fetch(req: FetchRequest, background_tasks: BackgroundTasks):
    """Trigger background fetching of data"""
    background_tasks.add_task(background_fetch_task, req)
    return {"message": "Data fetching started in background"}

@router.delete("/cache/{interval}")
def delete_cache(interval: str):
    """Delete cached data for interval"""
    try:
        count = manager.delete_data(interval)
        return {"message": f"Successfully deleted {count} files for {interval}", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metadata")
def get_metadata():
    """Get metadata for cached files"""
    try:
        metadata = manager.get_cache_metadata()
        
        return {"metadata": metadata}
            
        return {"metadata": metadata}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
