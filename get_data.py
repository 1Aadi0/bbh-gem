import h5py
import numpy as np
import os
import glob

# --- CONFIGURATION ---
# Path to your data (adjust if needed, but this matches your previous 'ls')
SEARCH_DIR = "./output-0000/binary_final"
OUTPUT_FILE = "final_data.npz"

def find_best_iteration(file_path):
    """Scans a file to find the iteration with the highest Refinement Level (rl)."""
    with h5py.File(file_path, 'r') as f:
        # Get all keys that look like data sets
        keys = [k for k in f.keys() if " it=" in k and " rl=" in k]
        if not keys: return None, None
        
        # Parse into (rl, iteration, key_name)
        parsed = []
        for k in keys:
            parts = k.split()
            rl = -1
            it = -1
            for p in parts:
                if p.startswith('rl='): rl = int(p.split('=')[1])
                if p.startswith('it='): it = int(p.split('=')[1])
            parsed.append((rl, it, k))
            
        # Sort by RL (descending) then Iteration (descending)
        # We want the finest grid at the latest time
        parsed.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return parsed[0] # Returns (best_rl, best_it, best_key)

def extract_all():
    print("🚀 STARTING FRESH DATA EXTRACTION...")
    
    if not os.path.exists(SEARCH_DIR):
        print(f"❌ Error: Could not find directory {SEARCH_DIR}")
        return

    # Files we need
    file_map = {
        'alp':   'admbase-lapse.h5',
        'shift': 'admbase-shift.h5',
        'metric': 'admbase-metric.h5'
    }
    
    # 1. Find the target iteration from the Lapse file
    lapse_path = os.path.join(SEARCH_DIR, file_map['alp'])
    print(f"--> Scanning {file_map['alp']} for best data...")
    best_rl, best_it, best_key = find_best_iteration(lapse_path)
    
    if best_key is None:
        print("❌ Could not find valid data in lapse file.")
        return
        
    print(f"   ✅ Target found: Iteration {best_it} (Refinement Level {best_rl})")

    data_store = {}
    
    # 2. Extract Lapse
    print("--> Extracting Lapse (Alpha)...")
    with h5py.File(lapse_path, 'r') as f:
        dset = f[best_key]
        vol = dset[:]
        # Save middle slice (Z=0)
        data_store['alp'] = vol[vol.shape[0]//2, :, :]
        
        # Get Coordinates
        try:
            origin = dset.attrs['origin']
            delta = dset.attrs['delta']
            nx = vol.shape[2]; ny = vol.shape[1]
            data_store['x'] = origin[0] + np.arange(nx) * delta[0]
            data_store['y'] = origin[1] + np.arange(ny) * delta[1]
        except:
            print("   ⚠️  Warning: Coordinate metadata missing. Using indices.")
            data_store['x'] = np.arange(vol.shape[2])
            data_store['y'] = np.arange(vol.shape[1])

    # 3. Extract Shift (Beta)
    print("--> Extracting Shift (Beta)...")
    shift_path = os.path.join(SEARCH_DIR, file_map['shift'])
    with h5py.File(shift_path, 'r') as f:
        for comp in ['betax', 'betay', 'betaz']:
            # Find key matching our target iteration and RL
            keys = [k for k in f.keys() if f" {comp} " in k or f"::{comp} " in k]
            target_key = next((k for k in keys if f"it={best_it}" in k and f"rl={best_rl}" in k), None)
            
            if target_key:
                vol = f[target_key][:]
                data_store[comp] = vol[vol.shape[0]//2, :, :]
            else:
                print(f"   ⚠️  Missing {comp}, filling with zeros.")
                data_store[comp] = np.zeros_like(data_store['alp'])

    # 4. Extract Metric (Gamma)
    print("--> Extracting Spatial Metric (Gamma)...")
    metric_path = os.path.join(SEARCH_DIR, file_map['metric'])
    with h5py.File(metric_path, 'r') as f:
        # Note: Depending on output, 'gzz' etc might be missing in 2D output, but we try all
        for comp in ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']:
            keys = [k for k in f.keys() if f" {comp} " in k or f"::{comp} " in k]
            target_key = next((k for k in keys if f"it={best_it}" in k and f"rl={best_rl}" in k), None)
            
            if target_key:
                vol = f[target_key][:]
                data_store[comp] = vol[vol.shape[0]//2, :, :]
            else:
                # Defaults: diagonal=1, off-diagonal=0
                default_val = 1.0 if comp in ['gxx', 'gyy', 'gzz'] else 0.0
                data_store[comp] = np.full_like(data_store['alp'], default_val)

    # 5. Save
    np.savez(OUTPUT_FILE, **data_store)
    print(f"\n🎉 DONE. Data saved to: {OUTPUT_FILE}")
    print("   Ready for Step 2.")

if __name__ == "__main__":
    extract_all()