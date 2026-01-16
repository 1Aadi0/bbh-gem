import h5py
import numpy as np
import os
import glob

# --- CONFIGURATION ---
SEARCH_DIR = "./output-0000/binary_final"
OUTPUT_FILE = "data.npz"

# --- PHYSICS TARGETS ---
# Based on mp_Psi4 data:
# Targets the specific physical time to extract data from.
TARGET_TIME = 130.0  

def find_closest_iteration_to_time(file_path, target_t):
    """Scans a file to find the iteration corresponding to the physical time t."""
    best_it = -1
    best_diff = 1e9
    best_key = None
    best_rl = -1

    with h5py.File(file_path, 'r') as f:
        # We only care about Refinement Level 8 (Fine Grid) or close to it
        # Adjust 'rl=...' filter if your fine grid is different (e.g., rl=7 or rl=6)
        # Using rl=2 or 3 is usually good for visualizing fields, rl=8 is too small (puncture only)
        target_vis_rl = 3 
        
        keys = [k for k in f.keys() if f" rl={target_vis_rl}" in k]
        
        if not keys:
            print(f"   ⚠️  Warning: RL={target_vis_rl} not found. Falling back to any RL.")
            keys = list(f.keys())

        for k in keys:
            # ETK HDF5 datasets usually have a 'time' attribute
            try:
                t_val = f[k].attrs['time']
                diff = abs(t_val - target_t)
                
                if diff < best_diff:
                    best_diff = diff
                    best_key = k
                    
                    # Extract iteration from string " ... it=25600 ..."
                    # Format is usually "name it=123 rl=2 ..."
                    parts = k.split()
                    for p in parts:
                        if p.startswith('it='): best_it = int(p.split('=')[1])
                        if p.startswith('rl='): best_rl = int(p.split('=')[1])
            except:
                continue
                
    return best_rl, best_it, best_key, best_diff

def extract_all():
    print(f"🚀 STARTING EXTRACTION FOR TIME t ≈ {TARGET_TIME} ...")
    
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
    print(f"--> Scanning {file_map['alp']} for time t={TARGET_TIME}...")
    
    best_rl, best_it, best_key, time_diff = find_closest_iteration_to_time(lapse_path, TARGET_TIME)
    
    if best_key is None:
        print("❌ Could not find valid data in lapse file.")
        return
        
    print(f"   ✅ Target found:")
    print(f"      - Iteration: {best_it}")
    print(f"      - Refinement Level: {best_rl}")
    print(f"      - Time Mismatch: {time_diff:.4f} M")

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
            # Construct a regex-like search for the same iteration and RL
            # We assume shift has same structure as lapse
            keys = [k for k in f.keys() if (f" {comp} " in k or f"::{comp} " in k)]
            
            # Strict match on iteration and RL to ensure sync
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
        for comp in ['gxx', 'gxy', 'gxz', 'gyy', 'gyz', 'gzz']:
            keys = [k for k in f.keys() if (f" {comp} " in k or f"::{comp} " in k)]
            target_key = next((k for k in keys if f"it={best_it}" in k and f"rl={best_rl}" in k), None)
            
            if target_key:
                vol = f[target_key][:]
                data_store[comp] = vol[vol.shape[0]//2, :, :]
            else:
                # Defaults: diagonal=1, off-diagonal=0
                default_val = 1.0 if comp in ['gxx', 'gyy', 'gzz'] else 0.0
                data_store[comp] = np.full_like(data_store['alp'], default_val)

    # 5. Save
    # CHANGE THIS LINE:
    np.savez(OUTPUT_FILE, time=TARGET_TIME, **data_store) 
    print(f"\n🎉 DONE. Data saved to: {OUTPUT_FILE}")
    print(f"   Snapshot Time: {TARGET_TIME}")

if __name__ == "__main__":
    extract_all()