import sympy as sp

# ==========================================
# 1. SPACETIME METRIC DEFINITIONS
# ==========================================

# ==========================================
# 1. SPACETIME METRIC DEFINITIONS (OPTIMIZED AST)
# ==========================================

def get_schwarzschild_spherical():
    """Defines the Schwarzschild Metric in Spherical Coordinates."""
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M = sp.symbols('M', real=True, positive=True)
    rs = 2 * M
    
    alpha_func = sp.sqrt(1 - rs/r)
    
    g_cov = sp.diag(-(alpha_func**2), 1/(alpha_func**2), r**2, r**2 * sp.sin(theta)**2)
    g_inv = sp.diag(-1/(alpha_func**2), alpha_func**2, 1/r**2, 1/(r**2 * sp.sin(theta)**2))
    
    # Exact known volume element (No .det() or .simplify() needed!)
    sqrt_det_g = r**2 * sp.sin(theta)
    
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],
        [0, 1/alpha_func, 0, 0],
        [0, 0, r, 0],
        [0, 0, 0, r*sp.sin(theta)]
    ])
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    
    return {
        'coords': [t, r, theta, phi], 'alpha_func': alpha_func, 'g_inv': g_inv,
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov,
        'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M}
    }

def get_schwarzschild_isotropic():
    """Defines the Schwarzschild Metric in Isotropic Cartesian Coordinates."""
    t, x, y, z = sp.symbols('t x y z', real=True)
    M = sp.symbols('M', real=True, positive=True)
    
    r_bar = sp.sqrt(x**2 + y**2 + z**2)
    psi = 1 + M / (2 * r_bar)
    alpha_func = (1 - M / (2 * r_bar)) / psi
    
    g_cov = sp.diag(-(alpha_func**2), psi**4, psi**4, psi**4)
    g_inv = sp.diag(-1/(alpha_func**2), 1/psi**4, 1/psi**4, 1/psi**4)
    
    sqrt_det_g = alpha_func * psi**6
    
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],
        [0, psi**2, 0, 0],
        [0, 0, psi**2, 0],
        [0, 0, 0, psi**2]
    ])
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    
    return {
        'coords': [t, x, y, z], 'alpha_func': alpha_func, 'g_inv': g_inv,
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov,
        'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'M': M}
    }

def get_reissner_nordstrom_spherical():
    """Reissner-Nordström Metric (Spherical Coordinates)"""
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M, Q = sp.symbols('M Q', real=True, positive=True)
    
    f = 1 - (2*M)/r + (Q**2)/r**2
    alpha_func = sp.sqrt(f)
    
    g_cov = sp.diag(-(alpha_func**2), 1/(alpha_func**2), r**2, r**2 * sp.sin(theta)**2)
    g_inv = sp.diag(-1/(alpha_func**2), alpha_func**2, 1/r**2, 1/(r**2 * sp.sin(theta)**2))
    
    sqrt_det_g = r**2 * sp.sin(theta)
    
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],
        [0, 1/alpha_func, 0, 0],
        [0, 0, r, 0],
        [0, 0, 0, r*sp.sin(theta)]
    ])
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    
    return {
        'coords': [t, r, theta, phi], 'alpha_func': alpha_func, 'g_inv': g_inv, 
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov, 
        'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M, 'Q': Q}
    }

def get_reissner_nordstrom_isotropic():
    """Reissner-Nordström Metric (Isotropic Cartesian Coordinates)"""
    t, x, y, z = sp.symbols('t x y z', real=True)
    M, Q = sp.symbols('M Q', real=True, positive=True)
    
    r_bar = sp.sqrt(x**2 + y**2 + z**2)
    H = 1 + M/r_bar + (M**2 - Q**2)/(4 * r_bar**2)
    alpha_func = (1 - (M**2 - Q**2)/(4 * r_bar**2)) / H
    
    g_cov = sp.diag(-(alpha_func**2), H**2, H**2, H**2)
    g_inv = sp.diag(-1/(alpha_func**2), 1/H**2, 1/H**2, 1/H**2)
    
    sqrt_det_g = alpha_func * H**3
    
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],
        [0, H, 0, 0],
        [0, 0, H, 0],
        [0, 0, 0, H]
    ])
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    
    return {
        'coords': [t, x, y, z], 'alpha_func': alpha_func, 'g_inv': g_inv, 
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov, 
        'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'M': M, 'Q': Q}
    }

def get_flrw_cartesian():
    """Flat FLRW Cosmological Metric (Cartesian Coordinates)"""
    t, x, y, z = sp.symbols('t x y z', real=True)
    a_t = sp.Function('a')(t) 
    alpha_func = sp.sympify(1)
    
    g_cov = sp.diag(-1, a_t**2, a_t**2, a_t**2)
    g_inv = sp.diag(-1, 1/a_t**2, 1/a_t**2, 1/a_t**2)
    
    sqrt_det_g = a_t**3
    
    Tetrad = sp.Matrix([
        [1, 0, 0, 0],
        [0, a_t, 0, 0],
        [0, 0, a_t, 0],
        [0, 0, 0, a_t]
    ])
    n_cov = sp.Matrix([-1, 0, 0, 0])
    
    return {
        'coords': [t, x, y, z], 'alpha_func': alpha_func, 'g_inv': g_inv, 
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov, 
        'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'a_t': a_t}
    }

def get_bardeen_spherical():
    """Bardeen Regular Black Hole (Spherical Coordinates)"""
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M, Qm = sp.symbols('M Qm', real=True, positive=True) 
    
    f = 1 - (2 * M * r**2) / (r**2 + Qm**2)**(sp.Rational(3, 2))
    alpha_func = sp.sqrt(f)
    
    g_cov = sp.diag(-(alpha_func**2), 1/(alpha_func**2), r**2, r**2 * sp.sin(theta)**2)
    g_inv = sp.diag(-1/(alpha_func**2), alpha_func**2, 1/r**2, 1/(r**2 * sp.sin(theta)**2))
    
    sqrt_det_g = r**2 * sp.sin(theta)
    
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],
        [0, 1/alpha_func, 0, 0],
        [0, 0, r, 0],
        [0, 0, 0, r*sp.sin(theta)]
    ])
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    
    return {
        'coords': [t, r, theta, phi], 'alpha_func': alpha_func, 'g_inv': g_inv, 
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov, 
        'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M, 'Qm': Qm}
    }

def get_kerr_boyer_lindquist():
    """Kerr Metric (Exact Boyer-Lindquist Spherical Coordinates)"""
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M, a = sp.symbols('M a', real=True, positive=True)

    Sigma = r**2 + a**2 * sp.cos(theta)**2
    Delta = r**2 - 2*M*r + a**2
    A_term = (r**2 + a**2)**2 - a**2 * Delta * sp.sin(theta)**2

    alpha_sq = Sigma * Delta / A_term
    alpha_func = sp.sqrt(alpha_sq)
    beta_phi = -2 * M * a * r / A_term

    gamma_rr = Sigma / Delta
    gamma_ttheta = Sigma
    gamma_pp = (A_term / Sigma) * sp.sin(theta)**2

    g_cov = sp.Matrix([
        [-alpha_sq + gamma_pp * beta_phi**2, 0, 0, gamma_pp * beta_phi],
        [0, gamma_rr, 0, 0],
        [0, 0, gamma_ttheta, 0],
        [gamma_pp * beta_phi, 0, 0, gamma_pp]
    ])
    
    g_inv = sp.Matrix([
        [-1/alpha_sq, 0, 0, beta_phi/alpha_sq],
        [0, 1/gamma_rr, 0, 0],
        [0, 0, 1/gamma_ttheta, 0],
        [beta_phi/alpha_sq, 0, 0, 1/gamma_pp - (beta_phi**2)/alpha_sq]
    ])
    
    sqrt_det_g = alpha_func * sp.sqrt(gamma_rr * gamma_ttheta * gamma_pp)
    sqrt_det_g = sqrt_det_g.replace(sp.Abs(sp.sin(theta)), sp.sin(theta))

    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],
        [0, sp.sqrt(gamma_rr), 0, 0],
        [0, 0, sp.sqrt(gamma_ttheta), 0],
        [beta_phi * sp.sqrt(gamma_pp), 0, 0, sp.sqrt(gamma_pp)]
    ])
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])

    return {
        'coords': [t, r, theta, phi], 'alpha_func': alpha_func, 'g_inv': g_inv,
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov,
        'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M, 'a': a}
    }

def get_kerr_quasi_isotropic_spherical():
    """Kerr Metric (Quasi-Isotropic Spherical Coordinates)"""
    t, r_bar, theta, phi = sp.symbols('t r_bar theta phi', real=True)
    M, a = sp.symbols('M a', real=True, positive=True)
    
    r = r_bar + M + (M**2 - a**2) / (4 * r_bar)
    Sigma = r**2 + a**2 * sp.cos(theta)**2
    Delta = r**2 - 2*M*r + a**2
    A_term = (r**2 + a**2)**2 - a**2 * Delta * sp.sin(theta)**2
    
    alpha_sq = Sigma * Delta / A_term
    alpha_func = sp.sqrt(alpha_sq)
    beta_phi = -2 * M * a * r / A_term
    
    gamma_rr = Sigma / r_bar**2
    gamma_ttheta = Sigma
    gamma_pp = (A_term / Sigma) * sp.sin(theta)**2
    
    g_cov = sp.Matrix([
        [-alpha_sq + gamma_pp * beta_phi**2, 0, 0, gamma_pp * beta_phi],
        [0, gamma_rr, 0, 0],
        [0, 0, gamma_ttheta, 0],
        [gamma_pp * beta_phi, 0, 0, gamma_pp]
    ])
    
    g_inv = sp.Matrix([
        [-1/alpha_sq, 0, 0, beta_phi/alpha_sq],
        [0, 1/gamma_rr, 0, 0],
        [0, 0, 1/gamma_ttheta, 0],
        [beta_phi/alpha_sq, 0, 0, 1/gamma_pp - (beta_phi**2)/alpha_sq]
    ])
    
    sqrt_det_g = alpha_func * sp.sqrt(gamma_rr * gamma_ttheta * gamma_pp)
    sqrt_det_g = sqrt_det_g.replace(sp.Abs(sp.sin(theta)), sp.sin(theta))
    
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],
        [0, sp.sqrt(gamma_rr), 0, 0],
        [0, 0, sp.sqrt(gamma_ttheta), 0],
        [-beta_phi * sp.sqrt(gamma_pp), 0, 0, sp.sqrt(gamma_pp)]
    ])
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    
    return {
        'coords': [t, r_bar, theta, phi], 'alpha_func': alpha_func, 'g_inv': g_inv, 
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov, 
        'symbols': {'t': t, 'r_bar': r_bar, 'theta': theta, 'phi': phi, 'M': M, 'a': a}
    }

def get_kerr_schild_cartesian():
    """Kerr-Schild Metric (Horizon-Penetrating Exact Cartesian Coordinates)"""
    t, x, y, z = sp.symbols('t x y z', real=True)
    M, a = sp.symbols('M a', real=True, positive=True)

    R2 = x**2 + y**2 + z**2
    r2 = (R2 - a**2)/2 + sp.sqrt((R2 - a**2)**2 / 4 + a**2 * z**2)
    r_val = sp.sqrt(r2)

    f = (2 * M * r_val**3) / (r2**2 + a**2 * z**2)

    l_x = (r_val * x + a * y) / (r2 + a**2)
    l_y = (r_val * y - a * x) / (r2 + a**2)
    l_z = z / r_val
    l_vec = [l_x, l_y, l_z]

    alpha_func = 1 / sp.sqrt(1 + f)

    Tetrad = sp.zeros(4, 4)
    Tetrad[0, 0] = alpha_func
    for k in range(3):
        Tetrad[k+1, 0] = (f / sp.sqrt(1 + f)) * l_vec[k]
        for i in range(3):
            delta_ki = 1 if k == i else 0
            Tetrad[k+1, i+1] = delta_ki + (sp.sqrt(1 + f) - 1) * l_vec[k] * l_vec[i]

    g_cov = sp.zeros(4, 4)
    l_mu = [1, l_x, l_y, l_z]
    eta = sp.diag(-1, 1, 1, 1)

    for mu in range(4):
        for nu in range(4):
            g_cov[mu, nu] = eta[mu, nu] + f * l_mu[mu] * l_mu[nu]

    g_inv = sp.zeros(4, 4)
    l_up = [-1, l_x, l_y, l_z] 
    for mu in range(4):
        for nu in range(4):
            g_inv[mu, nu] = eta[mu, nu] - f * l_up[mu] * l_up[nu]

    sqrt_det_g = sp.sympify(1)
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])

    return {
        'coords': [t, x, y, z], 'alpha_func': alpha_func, 'g_inv': g_inv,
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov,
        'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'M': M, 'a': a}
    }
# ==========================================
# 2. THE CORE TENSOR ENGINE (RAW AST EVALUATION)
# ==========================================

def calculate_automated_fields(metric_data):
    """
    Takes a standardized metric dictionary and calculates all DGREM fields.
    Zero algebraic simplification. Builds the raw computational graph for NumPy.
    Executes instantly.
    """
    coords = metric_data['coords']
    alpha_func = metric_data['alpha_func']
    g_inv = metric_data['g_inv']
    sqrt_det_g = metric_data['sqrt_det_g']
    Tetrad = metric_data['Tetrad']
    n_cov = metric_data['n_cov']
    
    half = sp.Rational(1, 2)

    def get_dynamical_fields(alpha_idx):
        A_mu = Tetrad[alpha_idx, :]
        F_cov = sp.zeros(4, 4)
        for u in range(4):
            for v in range(4):
                # PURE DERIVATIVE, NO SIMPLIFICATION
                F_cov[u, v] = sp.diff(A_mu[v], coords[u]) - sp.diff(A_mu[u], coords[v])
        
        # PURE MATRIX MULTIPLICATION
        F_up = g_inv * F_cov * g_inv.T
        
        E_vec = [sum(n_cov[v] * F_up[u, v] for v in range(4)) for u in range(4)]
        
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
                dual_F[u, v] = hodge_val / (2 * sqrt_det_g)
        
        B_vec = [sum(n_cov[v] * dual_F[u, v] for v in range(4)) for u in range(4)]
        return E_vec, B_vec
    
    def constitutive_relations(E_Results, B_Results, A_mu):
        E_hat = sp.zeros(4, 4)
        B_hat = sp.zeros(4, 4)
        for a in range(4):
            for i_hat in range(4):
                E_hat[a, i_hat] = sum(E_Results[a][mu] * A_mu[i_hat, mu] for mu in range(4))
                B_hat[a, i_hat] = sum(B_Results[a][mu] * A_mu[i_hat, mu] for mu in range(4))

        D_hat = sp.zeros(4, 4)
        H_hat = sp.zeros(4, 4)
        tr_E = sum(E_hat[l, l] for l in range(1, 4))
        tr_B = sum(B_hat[l, l] for l in range(1, 4))

        for i in range(1, 4):
            eps_sum_D = sum(sum(sp.LeviCivita(i, j, k) * B_hat[j, k] for k in range(1, 4)) for j in range(1, 4))
            D_hat[0, i] = -eps_sum_D
            eps_sum_H = sum(sum(sp.LeviCivita(i, j, k) * E_hat[j, k] for k in range(1, 4)) for j in range(1, 4))
            H_hat[0, i] = half * eps_sum_H

        for k in range(1, 4):
            for i in range(1, 4):
                delta_ki = 1 if k == i else 0
                D_hat[k, i] = -half * (E_hat[k, i] + E_hat[i, k]) + delta_ki * tr_E
                eps_term = sum(sp.LeviCivita(k, i, j) * E_hat[0, j] for j in range(1, 4))
                H_hat[k, i] = -B_hat[i, k] + half * delta_ki * tr_B + eps_term
        return E_hat, B_hat, D_hat, H_hat
    
    def calculate_charges_and_currents(E_hat, B_hat, D_hat, H_hat):
        rho = sp.zeros(4, 1) 
        s = sp.zeros(4, 4)   
        def contract_alpha(V, W, col_V, col_W):
            return sum(V[alpha, col_V] * W[alpha, col_W] for alpha in range(4))

        ED_scalar = sum(contract_alpha(E_hat, D_hat, k, k) for k in range(1, 4))
        BH_scalar = sum(contract_alpha(B_hat, H_hat, k, k) for k in range(1, 4))
        
        rho[0] = -half * (ED_scalar + BH_scalar)
        for i in range(1, 4):
            val = sum(sp.LeviCivita(i, j, k) * contract_alpha(B_hat, D_hat, j, k) 
                      for j in range(1, 4) for k in range(1, 4) if sp.LeviCivita(i, j, k) != 0)
            rho[i] = -val
            
        for i in range(1, 4):
            val = sum(sp.LeviCivita(i, j, k) * contract_alpha(E_hat, H_hat, j, k) 
                      for j in range(1, 4) for k in range(1, 4) if sp.LeviCivita(i, j, k) != 0)
            s[0, i] = -val
            
        for j in range(1, 4):
            for i in range(1, 4):
                delta_ij = 1 if i == j else 0
                s[j, i] = contract_alpha(E_hat, D_hat, j, i) + contract_alpha(B_hat, H_hat, i, j) - half * delta_ij * (ED_scalar + BH_scalar)
        return rho, s
    
    def calculate_PRL_charges_and_currents(E_hat, B_hat, D_hat, H_hat, sqrt_det_g, alpha_func):
        sqrt_gamma = sqrt_det_g / alpha_func
        q = sp.zeros(4, 1) 
        j = sp.zeros(4, 4) 
        def contract_alpha(V, W, col_V, col_W):
            return sum(V[alpha, col_V] * W[alpha, col_W] for alpha in range(4))

        ED_scalar = sum(contract_alpha(E_hat, D_hat, k, k) for k in range(1, 4))
        BH_scalar = sum(contract_alpha(B_hat, H_hat, k, k) for k in range(1, 4))
        
        q[0] = -half * sqrt_gamma * (ED_scalar + BH_scalar)
        for i in range(1, 4):
            val = sum(sp.LeviCivita(i, k, l) * contract_alpha(B_hat, D_hat, k, l) 
                      for k in range(1, 4) for l in range(1, 4) if sp.LeviCivita(i, k, l) != 0)
            q[i] = -sqrt_gamma * val
            
        for i in range(1, 4):
            val = sum(sp.LeviCivita(i, k, l) * contract_alpha(E_hat, H_hat, k, l) 
                      for k in range(1, 4) for l in range(1, 4) if sp.LeviCivita(i, k, l) != 0)
            j[0, i] = -sqrt_gamma * val
            
        for leg in range(1, 4): 
            for comp in range(1, 4):
                delta_leg_comp = 1 if comp == leg else 0
                j[leg, comp] = sqrt_gamma * (contract_alpha(E_hat, D_hat, leg, comp) + contract_alpha(B_hat, H_hat, comp, leg) - half * delta_leg_comp * (ED_scalar + BH_scalar))
        return q, j

    # Execute Engine
    E_Results, B_Results = [], []
    for i in range(4):
        E, B = get_dynamical_fields(i)
        E_Results.append(E)
        B_Results.append(B)

    E_hat, B_hat, D_hat, H_hat = constitutive_relations(E_Results, B_Results, Tetrad)
    rho_hat, s_hat = calculate_charges_and_currents(E_hat, B_hat, D_hat, H_hat)
    q_hat, j_hat = calculate_PRL_charges_and_currents(E_hat, B_hat, D_hat, H_hat, sqrt_det_g, alpha_func)
    
    return {
        'E_hat': E_hat, 'B_hat': B_hat, 
        'D_hat': D_hat, 'H_hat': H_hat,
        'rho_hat': rho_hat, 's_hat': s_hat,
        'q_hat': q_hat, 'j_hat': j_hat,
        'symbols': metric_data['symbols']
    }