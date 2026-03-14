# SymbolicEngine.py
import sympy as sp

def calculate_automated_fields():
    # --- 1. Variables and Metric Setup ---
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M = sp.symbols('M', real=True, positive=True)
    R_far = sp.symbols('R', real=True, positive=True)
    rs = 2 * M
    
    half = sp.Rational(1, 2)
    alpha_func = sp.sqrt(1 - rs/r)
    
    g_cov = sp.diag(-(alpha_func**2), 1/(alpha_func**2), r**2, r**2 * sp.sin(theta)**2)
    g_inv = g_cov.inv()
    
    # Fix the Absolute Value Branch Cut
    sqrt_det_g = sp.simplify(sp.sqrt(-g_cov.det())).replace(sp.Abs(sp.sin(theta)), sp.sin(theta))
    
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],
        [0, 1/alpha_func, 0, 0],
        [0, 0, r, 0],
        [0, 0, 0, r*sp.sin(theta)]
    ])
    # # #Test case
    # alpha_func = 1
    # # # Covariant Metric g_uv
    # g_cov = sp.diag(-1, 1, r**2, r**2 * sp.sin(theta)**2)
    # g_inv = g_cov.inv()
    # sqrt_det_g = sp.simplify(sp.sqrt(-g_cov.det())).replace(sp.Abs(sp.sin(theta)), sp.sin(theta))
    # # Non-Rotated Tetrad Legs A_mu^(hat_a)
    # Tetrad = sp.Matrix([
    #     [1, 0, 0, 0],           # t-leg (index 0)
    #     [0, 1, 0, 0],         # r-leg (index 1)
    #     [0, 0, r, 0],                    # theta-leg (index 2)
    #     [0, 0, 0, r*sp.sin(theta)]       # phi-leg (index 3)
    # ]) 
    # print("\n=== 1. NON-ROTATED TETRAD MATRIX (Diagonal / Spherical) ===")
    # print("Rows: [t, r, theta, phi]_hat | Cols: [dt, dr, dtheta, dphi]")
    # sp.pprint(Tetrad)

    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    coords = [t, r, theta, phi]

    def get_dynamical_fields(alpha_idx):
        A_mu = Tetrad[alpha_idx, :]
        F_cov = sp.zeros(4, 4)
        for u in range(4):
            for v in range(4):
                F_cov[u, v] = sp.simplify(sp.diff(A_mu[v], coords[u]) - sp.diff(A_mu[u], coords[v]))
        
        F_up = sp.simplify(g_inv * F_cov * g_inv.T)
        E_vec = [sp.simplify(sum(n_cov[v] * F_up[u, v] for v in range(4))) for u in range(4)]
        
        dual_F = sp.zeros(4, 4)
        for u in range(4):
            for v in range(4):
                if u == v: continue
                hodge_val = 0
                for s in range(4):
                    for r_idx in range(4):
                        eps = sp.LeviCivita(u, v, s, r_idx)
                        if eps != 0:
                            hodge_val += -eps * F_cov[s, r_idx]
                dual_F[u, v] = sp.simplify(hodge_val / (2 * sqrt_det_g))
        
        B_vec = [sp.simplify(sum(n_cov[v] * dual_F[u, v] for v in range(4))) for u in range(4)]
        return E_vec, B_vec
    
    def constitutive_relations(E_Results, B_Results, A_mu):
        E_hat = sp.zeros(4, 4)
        B_hat = sp.zeros(4, 4)
        for a in range(4):
            for i_hat in range(4):
                E_hat[a, i_hat] = sp.simplify(sum(E_Results[a][mu] * A_mu[i_hat, mu] for mu in range(4)))
                B_hat[a, i_hat] = sp.simplify(sum(B_Results[a][mu] * A_mu[i_hat, mu] for mu in range(4)))

        D_hat = sp.zeros(4, 4)
        H_hat = sp.zeros(4, 4)
        tr_E = sum(E_hat[l, l] for l in range(1, 4))
        tr_B = sum(B_hat[l, l] for l in range(1, 4))

        for i in range(1, 4):
            eps_sum_D = sum(sum(sp.LeviCivita(i, j, k) * B_hat[j, k] for k in range(1, 4)) for j in range(1, 4))
            D_hat[0, i] = sp.simplify(-eps_sum_D)
            eps_sum_H = sum(sum(sp.LeviCivita(i, j, k) * E_hat[j, k] for k in range(1, 4)) for j in range(1, 4))
            H_hat[0, i] = sp.simplify(half * eps_sum_H)

        for k in range(1, 4):
            for i in range(1, 4):
                delta_ki = 1 if k == i else 0
                D_hat[k, i] = sp.simplify(-half * (E_hat[k, i] + E_hat[i, k]) + delta_ki * tr_E)
                eps_term = sum(sp.LeviCivita(k, i, j) * E_hat[0, j] for j in range(1, 4))
                H_hat[k, i] = sp.simplify(-B_hat[i, k] + half * delta_ki * tr_B + eps_term)

        return E_hat, B_hat, D_hat, H_hat
    
    def calculate_charges_and_currents(E_hat, B_hat, D_hat, H_hat):
        rho = sp.zeros(4, 1) 
        s = sp.zeros(4, 4)   
        def contract_alpha(V, W, col_V, col_W):
            return sum(V[alpha, col_V] * W[alpha, col_W] for alpha in range(4))

        ED_scalar = sp.simplify(sum(contract_alpha(E_hat, D_hat, k, k) for k in range(1, 4)))
        BH_scalar = sp.simplify(sum(contract_alpha(B_hat, H_hat, k, k) for k in range(1, 4)))
        
        rho[0] = sp.simplify(-half * (ED_scalar + BH_scalar))
        for i in range(1, 4):
            val = sum(sp.LeviCivita(i, j, k) * contract_alpha(B_hat, D_hat, j, k) 
                      for j in range(1, 4) for k in range(1, 4) if sp.LeviCivita(i, j, k) != 0)
            rho[i] = sp.simplify(-val)
            
        for i in range(1, 4):
            val = sum(sp.LeviCivita(i, j, k) * contract_alpha(E_hat, H_hat, j, k) 
                      for j in range(1, 4) for k in range(1, 4) if sp.LeviCivita(i, j, k) != 0)
            s[0, i] = sp.simplify(-val)
            
        for j in range(1, 4):
            for i in range(1, 4):
                delta_ij = 1 if i == j else 0
                s[j, i] = sp.simplify(contract_alpha(E_hat, D_hat, j, i) + contract_alpha(B_hat, H_hat, i, j) - half * delta_ij * (ED_scalar + BH_scalar))
        return rho, s
    
    def calculate_PRL_charges_and_currents(E_hat, B_hat, D_hat, H_hat, sqrt_det_g, alpha_func):
        sqrt_gamma = sp.simplify(sqrt_det_g / alpha_func)
        q = sp.zeros(4, 1) 
        j = sp.zeros(4, 4) 
        def contract_alpha(V, W, col_V, col_W):
            return sum(V[alpha, col_V] * W[alpha, col_W] for alpha in range(4))

        ED_scalar = sp.simplify(sum(contract_alpha(E_hat, D_hat, k, k) for k in range(1, 4)))
        BH_scalar = sp.simplify(sum(contract_alpha(B_hat, H_hat, k, k) for k in range(1, 4)))
        
        q[0] = sp.simplify(-half * sqrt_gamma * (ED_scalar + BH_scalar))
        for i in range(1, 4):
            val = sum(sp.LeviCivita(i, k, l) * contract_alpha(B_hat, D_hat, k, l) 
                      for k in range(1, 4) for l in range(1, 4) if sp.LeviCivita(i, k, l) != 0)
            q[i] = sp.simplify(-sqrt_gamma * val)
            
        for i in range(1, 4):
            val = sum(sp.LeviCivita(i, k, l) * contract_alpha(E_hat, H_hat, k, l) 
                      for k in range(1, 4) for l in range(1, 4) if sp.LeviCivita(i, k, l) != 0)
            j[0, i] = sp.simplify(-sqrt_gamma * val)
            
        for leg in range(1, 4): 
            for comp in range(1, 4):
                delta_leg_comp = 1 if comp == leg else 0
                j[leg, comp] = sp.simplify(sqrt_gamma * (contract_alpha(E_hat, D_hat, leg, comp) + contract_alpha(B_hat, H_hat, comp, leg) - half * delta_leg_comp * (ED_scalar + BH_scalar)))
        return q, j
    
    print("Executing Symbolic Tensor Engine...")
    E_Results, B_Results = [], []
    for i in range(4):
        E, B = get_dynamical_fields(i)
        E_Results.append(E)
        B_Results.append(B)

    E_hat, B_hat, D_hat, H_hat = constitutive_relations(E_Results, B_Results, Tetrad)
    rho_hat, s_hat = calculate_charges_and_currents(E_hat, B_hat, D_hat, H_hat)
    q_hat, j_hat = calculate_PRL_charges_and_currents(E_hat, B_hat, D_hat, H_hat, sqrt_det_g, alpha_func)
    
    print("Calculations Complete.")
    
    # Return everything needed for plotting or further analysis
    return {
        'E_hat': E_hat, 'B_hat': B_hat, 
        'D_hat': D_hat, 'H_hat': H_hat,
        'rho_hat': rho_hat, 's_hat': s_hat,
        'q_hat': q_hat, 'j_hat': j_hat,
        'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M}
    }