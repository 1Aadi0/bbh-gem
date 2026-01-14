import h5py
import numpy as np
import os

# --- CONFIGURATION ---
SEARCH_DIR = "./output-0000/binary_final"
OUTPUT_FILE = "final_data.npz"

# Target Refinement Level (RL). 
# RL=0 is the biggest box (whole system). RL=7 is the smallest (inside BH).
# We want something in the middle/coarse end to see the binary.
TARGET_RL = 2 

def extract_binary_grid():
    print("🚀 EXTRACTING BINARY VIEW (Robust Mode)...")
    
    file_map = {'alp': 'admbase-lapse.h5', 'shift': 'admbase-shift.h5', 'metric': 'admbase-metric.h5'}
    lapse_path = os.path.join(SEARCH_DIR, file_map['alp'])
    
    if not os.path.exists(lapse_path):
        print(f"❌ Error: {lapse_path} not found.")
        return

    # 1. Find the Best Available Grid
    target_key = None
    target_it = None
    selected_rl = -1
    
    with h5py.File(lapse_path, 'r') as f:
        # Get keys with iteration and refinement level info
        all_keys = [k for k in f.keys() if " it=" in k and " rl=" in k]
        
        if not all_keys:
            print("❌ No valid datasets found in lapse file.")
            return

        # extract iterations
        # format usually: "name it=12345 ..."
        try:
            iterations = sorted(list(set([int(k.split(' it=')[1].split()[0]) for k in all_keys])))
            target_it = iterations[-1] # Use last iteration
            print(f"   -> Latest Iteration: {target_it}")
        except Exception as e:
            print(f"❌ Error parsing iterations: {e}")
            return

        # Filter keys for this iteration
        it_keys = [k for k in all_keys if f" it={target_it} " in k]
        
        # Parse RLs for this iteration
        # list of tuples: (rl_value, key_string)
        available_grids = []
        for k in it_keys:
            try:
                # robust parsing of " rl=X "
                rl_str = k.split(' rl=')[1].split()[0]
                rl_val = int(rl_str)
                available_grids.append((rl_val, k))
            except:
                continue
        
        if not available_grids:
            print("❌ Could not parse Refinement Levels (RL) for the latest iteration.")
            return

        # Sort by RL
        available_grids.sort(key=lambda x: x[0])
        print(f"   -> Available RLs: {[x[0] for x in available_grids]}")

        # Logic: Find RL closest to TARGET_RL
        # If TARGET_RL is 2, and we have [0, 1, 6, 7], we pick 1.
        best_match = min(available_grids, key=lambda x: abs(x[0] - TARGET_RL))
        
        selected_rl = best_match[0]
        target_key = best_match[1]
        
        print(f"   ✅ Selected Grid: RL={selected_rl} (Closest to Target {TARGET_RL})")
        print(f"      Key: {target_key}")

    # 2. Extract Data (Lapse)
    data_store = {}
    print("   -> Extracting Lapse...")
    with h5py.File(lapse_path, 'r') as f:
        dset = f[target_key]
        vol = dset[:]
        data_store['alp'] = vol[vol.shape[0]//2, :, :] # Z=0 slice
        
        # Coords
        try:
            origin = dset.attrs['origin']
            delta = dset.attrs['delta']
            nx = vol.shape[2]; ny = vol.shape[1]
            data_store['x'] = origin[0] + np.arange(nx) * delta[0]
            data_store['y'] = origin[1] + np.arange(ny) * delta[1]
            print(f"      Domain: X [{data_store['x'][0]:.1f}, {data_store['x'][-1]:.1f}]")
        except:
            data_store['x'] = np.arange(vol.shape[2])
            data_store['y'] = np.arange(vol.shape[1])

    # 3. Extract Shift & Metric (using robust matching)
    # We need to find keys in other files that match 'it={target_it}' and 'rl={selected_rl}'
    
    for ftype, fname in [('shift', file_map['shift']), ('metric', file_map['metric'])]:
        fpath = os.path.join(SEARCH_DIR, fname)
        print(f"   -> Extracting {ftype}...")
        
        with h5py.File(fpath, 'r') as f:
            for comp in ['betax', 'betay', 'betaz', 'gxx', 'gxy', 'gyy', 'gxz', 'gyz', 'gzz']:
                # Filter irrelevants
                if ftype == 'metric' and comp.startswith('beta'): continue
                if ftype == 'shift' and comp.startswith('g'): continue
                
                # Search for matching key
                # We need a key that has the variable name, the iteration, AND the RL
                keys = [k for k in f.keys() if (f" {comp} " in k or f"::{comp} " in k)]
                
                # Strict match on IT and RL
                match_key = None
                for k in keys:
                    if f" it={target_it} " in k and f" rl={selected_rl} " in k:
                        match_key = k
                        break
                
                if match_key:
                    vol = f[match_key][:]
                    data_store[comp] = vol[vol.shape[0]//2, :, :]
                else:
                    # Fill default
                    val = 1.0 if comp in ['gxx','gyy','gzz'] else 0.0
                    data_store[comp] = np.full_like(data_store['alp'], val)

    np.savez(OUTPUT_FILE, **data_store)
    print(f"\n🎉 DONE. Binary grid saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_binary_grid()