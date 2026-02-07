
import os
import pandas as pd

def verify_all_variance():
    cache_dir = 'data_cache'
    files = [f for f in os.listdir(cache_dir) if f.endswith('.parquet')]
    
    print(f"Verifying {len(files)} files...")
    flat_count = 0
    results = []
    
    for f in files:
        df = pd.read_parquet(os.path.join(cache_dir, f))
        std = df['close'].std()
        unique = len(df['close'].unique())
        
        if std == 0 or unique == 1:
            status = "FLAT"
            flat_count += 1
        else:
            status = "OK"
            
        results.append({
            'file': f,
            'std': std,
            'unique': unique,
            'status': status
        })
    
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print(res_df.sort_values('status', ascending=False).to_string())
    
    print(f"\nFinal Summary: {flat_count} flat out of {len(files)} files.")

if __name__ == "__main__":
    verify_all_variance()
