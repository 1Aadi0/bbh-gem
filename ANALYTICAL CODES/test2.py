import sympy as sp

def calculate_automated_fields():
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M = sp.symbols('M', real=True, positive=True)
    rs = 2 * M
    
    # 1. Define Metric and Tetrad
    alpha_func = sp.sqrt(1 - rs/r)
    g_cov = sp.diag(-(alpha_func**2), 1/(alpha_func**2), r**2, r**2 * sp.sin(theta)**2)
    g_inv = g_cov.inv()
    sqrt_det_g = sp.simplify(sp.sqrt(-g_cov.det()))
    
    # Non-Rotated Tetrad Legs A_mu^(hat_a)
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],           # t-leg
        [0, 1/alpha_func, 0, 0],         # r-leg
        [0, 0, r, 0],                    # theta-leg
        [0, 0, 0, r*sp.sin(theta)]       # phi-leg
    ])

    # 2. Define Normal Observer n_lambda = (-alpha, 0, 0, 0)
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    coords = [t, r, theta, phi]

    def get_fields(alpha_idx):
        A_mu = Tetrad[alpha_idx, :]
        
        # Calculate Covariant Field Strength F_uv
        F_cov = sp.zeros(4, 4)
        for u in range(4):
            for v in range(4):
                F_uv = sp.diff(A_mu[v], coords[u]) - sp.diff(A_mu[u], coords[v])
                F_cov[u, v] = sp.simplify(F_uv)
        
        # Raise indices: F^uv = g^ua g^vb F_ab
        F_inv = sp.simplify(g_inv * F_cov * g_inv.T)
        
        # A. Calculate Electric Field Vector: E^u = n_v * F^uv
        # Dynamically contracts n_covariant with F_contravariant
        E_vec = [sp.simplify(sum(n_cov[v] * F_inv[u, v] for v in range(4))) for u in range(4)]
        
        # B. Calculate Hodge Dual (*F)^uv = (1/2) * (1/sqrt(-g)) * epsilon^uvsr * F_sr
        dual_F = sp.zeros(4, 4)
        for u in range(4):
            for v in range(4):
                if u == v: continue
                val = 0
                for s in range(4):
                    for r_idx in range(4):
                        eps = sp.LeviCivita(u, v, s, r_idx)
                        if eps != 0:
                            val += eps * F_cov[s, r_idx]
                dual_F[u, v] = sp.simplify(val / (2 * sqrt_det_g))
        
        # C. Calculate Magnetic Field Vector: B^u = n_v * (*F)^uv
        B_vec = [sp.simplify(sum(n_cov[v] * dual_F[u, v] for v in range(4))) for u in range(4)]
        
        return E_vec, B_vec

    # 3. Execution and Display
    labels = ['t', 'r', 'theta', 'phi']
    print("=== DYNAMICAL FIELD CALCULATION (NON-ROTATED) ===")
    for i in range(4):
        E, B = get_fields(i)
        print(f"\n--- Tetrad Leg: {labels[i]} ---")
        for mu in range(4):
            if E[mu] != 0: print(f"E^({labels[i]})[{coords[mu]}] = {E[mu]}")
            if B[mu] != 0: print(f"B^({labels[i]})[{coords[mu]}] = {B[mu]}")

if __name__ == "__main__":
    calculate_automated_fields()