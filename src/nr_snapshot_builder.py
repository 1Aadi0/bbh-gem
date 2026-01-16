import h5py
import numpy as np
import os

# --- CONFIGURATION ---
SEARCH_DIR = "./output-0000/binary_final"
OUTPUT_FILE = "data.npz"

# --- PHYSICS CONTROLS ---
# Based on mp_Psi4 data:
# Targets the specific physical time to extract data from.
TARGET_TIME = 80.0  

# Refinement Level to extract (2 is usually good for visualization)
TARGET_RL = 3 

def get_anchor_iteration(h5_file, target_time, target_rl):
    """
    Scans a reference file (Lapse) to find the Iteration Number 
    that corresponds to the Physical Time closest to TARGET_TIME.
    """
    best_it = -1
    min_diff = 1e9
    best_key = None

    if not os.path.exists(h5_file):
        return None, None, None

    with h5py.File(h5_file, 'r') as f:
        # Filter for the correct Refinement Level first
        keys = [k for k in f.keys() if f" rl={target_rl}" in k]
        
        if not keys:
            print(f"   ⚠️  RL={target_rl} not found in anchor file. Checking all RLs...")
            keys = list(f.keys())

        for k in keys:
            # ETK files usually store physical time in attributes
            try:
                t_val = f[k].attrs['time']
                diff = abs(t_val - target_time)
                
                if diff < min_diff:
                    min_diff = diff
                    best_key = k
                    # Parse Iteration from key string "name it=123 rl=..."
                    parts = k.split()
                    for p in parts:
                        if p.startswith('it='):
                            best_it = int(p.split('=')[1])
            except:
                continue

    return best_it, min_diff, best_key

def find_closest_dataset(h5_file, target_it, target_rl, var_name):
    """
    Finds a dataset matching RL and closest possible Iteration to target_it.
    Useful because Shift/Metric might save at slightly different steps than Lapse.
    """
    with h5py.File(h5_file, 'r') as f:
        # Filter keys that have the variable name AND correct RL
        candidates = []
        for k in f.keys():
            # Check variable name match (e.g. " betax " or "::betax ")
            if (f" {var_name} " in k or f"::{var_name} " in k) and f" rl={target_rl}" in k:
                try:
                    # Extract iteration
                    it = int(k.split(' it=')[1].split()[0])
                    candidates.append((it, k))
                except: continue
        
        if not candidates: return None
        
        # Sort by distance to target_it
        candidates.sort(key=lambda x: abs(x[0] - target_it))
        best_it, best_key = candidates[0]
        
        # Warn if the mismatch is huge (> 500 iterations is suspicious)
        if abs(best_it - target_it) > 500:
            print(f"   ⚠️  Large sync gap for {var_name}: Wanted it={target_it}, Found it={best_it}")
            
        return best_key

def extract_snapshot():
    print(f"🚀 EXTRACTING SNAPSHOT FOR TIME t ≈ {TARGET_TIME} ...")
    
    file_map = {
        'alp': 'admbase-lapse.h5', 
        'shift': 'admbase-shift.h5', 
        'metric': 'admbase-metric.h5'
    }
    
    lapse_path = os.path.join(SEARCH_DIR, file_map['alp'])
    if not os.path.exists(lapse_path):
        print(f"❌ Error: {lapse_path} not found.")
        return

    # 1. Find the Anchor Iteration (The moment in time we want)
    print(f"--> Scanning Lapse to map Time {TARGET_TIME} -> Iteration...")
    target_it, time_diff, ref_key = get_anchor_iteration(lapse_path, TARGET_TIME, TARGET_RL)

    if target_it == -1 or ref_key is None:
        print("❌ Could not find valid data for that time.")
        return

    print(f"   ✅ Anchor Found:")
    print(f"      - Iteration: {target_it}")
    print(f"      - Time Error: {time_diff:.5f} M")
    print(f"      - Refinement Level: {TARGET_RL}")

    data_store = {}

    # 2. Extract Lapse (The Anchor)
    with h5py.File(lapse_path, 'r') as f:
        dset = f[ref_key]
        vol = dset[:]
        # Save Z=0 slice (Middle of the box)
        data_store['alp'] = vol[vol.shape[0]//2, :, :]
        
        # Extract Coordinates
        try:
            origin = dset.attrs['origin']
            delta = dset.attrs['delta']
            nx = vol.shape[2]; ny = vol.shape[1]
            data_store['x'] = origin[0] + np.arange(nx) * delta[0]
            data_store['y'] = origin[1] + np.arange(ny) * delta[1]
        except:
            print("   ⚠️  No coordinate metadata. Generating index grid.")
            data_store['x'] = np.arange(vol.shape[2])
            data_store['y'] = np.arange(vol.shape[1])

    # 3. Extract Shift and Metric (Using Fuzzy Matching to target_it)
    for ftype, fname in [('shift', file_map['shift']), ('metric', file_map['metric'])]:
        fpath = os.path.join(SEARCH_DIR, fname)
        if not os.path.exists(fpath): 
            print(f"   ⚠️  Skipping {fname} (not found)")
            continue
        
        print(f"   -> Scanning {ftype} for it={target_it}...")
        
        vars_to_get = []
        if ftype == 'shift': vars_to_get = ['betax', 'betay', 'betaz']
        if ftype == 'metric': vars_to_get = ['gxx', 'gxy', 'gyy', 'gxz', 'gyz', 'gzz']
        
        for comp in vars_to_get:
            best_key = find_closest_dataset(fpath, target_it, TARGET_RL, comp)
            
            if best_key:
                with h5py.File(fpath, 'r') as f:
                    vol = f[best_key][:]
                    data_store[comp] = vol[vol.shape[0]//2, :, :]
            else:
                print(f"      ⚠️  Missing {comp}. Filling with Identity/Zero.")
                # Fallback: Metric diagonal=1, others=0
                val = 1.0 if comp in ['gxx','gyy','gzz'] else 0.0
                data_store[comp] = np.full_like(data_store['alp'], val)

    # 4. Save to Disk
    # CHANGE THIS LINE:
    np.savez(OUTPUT_FILE, time=TARGET_TIME, **data_store) 
    # Using TARGET_TIME ensures consistency with the user-defined target 
    # rather than the exact physical time from the dataset.
    print(f"\n🎉 SNAPSHOT SAVED: {OUTPUT_FILE}")
    print(f"   Physical Time: {TARGET_TIME}")
    print(f"   Iteration: {target_it}")

if __name__ == "__main__":
    extract_snapshot()