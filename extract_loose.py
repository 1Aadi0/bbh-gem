import h5py
import numpy as np
import os

# --- CONFIGURATION ---
SEARCH_DIR = "./output-0000/binary_final"
OUTPUT_FILE = "final_data.npz"
TARGET_RL = 2  # The refinement level we want (Coarse enough to see binary)

def find_closest_dataset(h5_file, target_it, target_rl, var_name):
    """Finds a dataset matching RL and closest possible Iteration."""
    with h5py.File(h5_file, 'r') as f:
        # Filter keys that have the variable and correct RL
        candidates = []
        for k in f.keys():
            if (f" {var_name} " in k or f"::{var_name} " in k) and f" rl={target_rl}" in k:
                try:
                    it = int(k.split(' it=')[1].split()[0])
                    candidates.append((it, k))
                except: continue
        
        if not candidates: return None
        
        # Sort by distance to target_it
        candidates.sort(key=lambda x: abs(x[0] - target_it))
        best_it, best_key = candidates[0]
        
        # Warn if the mismatch is huge (more than 100 iterations)
        if abs(best_it - target_it) > 100:
            print(f"   ⚠️  Warning: Large time gap for {var_name} (Target It: {target_it}, Found: {best_it})")
            
        return best_key

def extract_loose():
    print("🚀 EXTRACTING DATA (Loose Match Mode)...")
    
    file_map = {'alp': 'admbase-lapse.h5', 'shift': 'admbase-shift.h5', 'metric': 'admbase-metric.h5'}
    lapse_path = os.path.join(SEARCH_DIR, file_map['alp'])
    
    if not os.path.exists(lapse_path):
        print("❌ Data directory not found.")
        return

    # 1. Get Reference Grid (Lapse)
    data_store = {}
    target_it = 0
    
    with h5py.File(lapse_path, 'r') as f:
        # Find latest iteration for TARGET_RL
        keys = [k for k in f.keys() if f" rl={TARGET_RL}" in k]
        if not keys:
            print(f"❌ RL={TARGET_RL} not found in Lapse. Try a different RL (e.g., 0 or 1).")
            return
            
        iterations = sorted([int(k.split(' it=')[1].split()[0]) for k in keys])
        target_it = iterations[-1]
        
        # Find exact key for this iteration
        ref_key = next(k for k in keys if f" it={target_it} " in k)
        
        print(f"   ✅ Reference: Iteration {target_it}, RL={TARGET_RL}")
        print(f"      Key: {ref_key}")
        
        # Extract Lapse
        dset = f[ref_key]
        vol = dset[:]
        data_store['alp'] = vol[vol.shape[0]//2, :, :]
        
        # Extract Coordinates
        try:
            origin = dset.attrs['origin']
            delta = dset.attrs['delta']
            nx = vol.shape[2]; ny = vol.shape[1]
            data_store['x'] = origin[0] + np.arange(nx) * delta[0]
            data_store['y'] = origin[1] + np.arange(ny) * delta[1]
        except:
            data_store['x'] = np.arange(vol.shape[2])
            data_store['y'] = np.arange(vol.shape[1])
            
    # 2. Extract Other Variables (Fuzzy Match)
    for ftype, fname in [('shift', file_map['shift']), ('metric', file_map['metric'])]:
        fpath = os.path.join(SEARCH_DIR, fname)
        if not os.path.exists(fpath): continue
        
        print(f"   -> Scanning {ftype}...")
        
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
                print(f"      ⚠️  Missing {comp}. Filling with default.")
                val = 1.0 if comp in ['gxx','gyy','gzz'] else 0.0
                data_store[comp] = np.full_like(data_store['alp'], val)

    np.savez(OUTPUT_FILE, **data_store)
    print(f"\n🎉 DONE. Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_loose()