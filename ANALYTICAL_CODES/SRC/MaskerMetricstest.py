# ==========================================
# dGREM TENSOR ENGINE: FULL MASKED METRIC LIBRARY
# Architecture: 
# 1D/2D Dummies for Spherical (Base symbols r, theta)
# 3D Multi-Variable Dummies for Cartesian (x, y, z)
# ==========================================
import sympy as sp

# ------------------------------------------
# 1. SCHWARZSCHILD GEOMETRY
# ------------------------------------------

def get_schwarzschild_spherical_masked():
    """1. Standard Static Slicing (Spherical)"""
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M = sp.symbols('M', real=True, positive=True)
    
    A_d = sp.Function('Alpha')(r)
    actual_A = sp.sqrt(1 - 2*M/r)
    
    g_inv = sp.diag(-1/(A_d**2), A_d**2, 1/r**2, 1/(r**2 * sp.sin(theta)**2))
    Tetrad = sp.Matrix([[A_d, 0, 0, 0], [0, 1/A_d, 0, 0], [0, 0, r, 0], [0, 0, 0, r*sp.sin(theta)]])
    
    subs_dict = {sp.Derivative(A_d, r): sp.diff(actual_A, r), A_d: actual_A}
    
    return {'coords': [t, r, theta, phi], 'alpha_func': A_d, 'g_inv': g_inv,
            'sqrt_det_g': r**2 * sp.sin(theta), 'Tetrad': Tetrad, 'n_cov': sp.Matrix([-A_d, 0, 0, 0]),
            'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M}, 'subs_dict': subs_dict}

def get_schwarzschild_isotropic_cartesian_masked():
    """2. Static Slicing (Isotropic Cartesian)"""
    t, x, y, z = sp.symbols('t x y z', real=True)
    M = sp.symbols('M', real=True, positive=True)
    r_bar = sp.sqrt(x**2 + y**2 + z**2)
    
    Psi_d = sp.Function('Psi')(x, y, z)
    Alpha_d = sp.Function('Alpha')(x, y, z)
    
    act_Psi = 1 + M / (2 * r_bar)
    act_Alpha = (1 - M / (2 * r_bar)) / act_Psi
    
    g_inv = sp.diag(-1/(Alpha_d**2), 1/Psi_d**4, 1/Psi_d**4, 1/Psi_d**4)
    Tetrad = sp.Matrix([[Alpha_d, 0, 0, 0], [0, Psi_d**2, 0, 0], [0, 0, Psi_d**2, 0], [0, 0, 0, Psi_d**2]])
    
    subs_dict = {
        sp.Derivative(Psi_d, x): sp.diff(act_Psi, x), sp.Derivative(Alpha_d, x): sp.diff(act_Alpha, x),
        sp.Derivative(Psi_d, y): sp.diff(act_Psi, y), sp.Derivative(Alpha_d, y): sp.diff(act_Alpha, y),
        sp.Derivative(Psi_d, z): sp.diff(act_Psi, z), sp.Derivative(Alpha_d, z): sp.diff(act_Alpha, z),
        Psi_d: act_Psi, Alpha_d: act_Alpha
    }
    
    return {'coords': [t, x, y, z], 'alpha_func': Alpha_d, 'g_inv': g_inv,
            'sqrt_det_g': Alpha_d * Psi_d**6, 'Tetrad': Tetrad, 'n_cov': sp.Matrix([-Alpha_d, 0, 0, 0]),
            'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'M': M}, 'subs_dict': subs_dict}

def get_schwarzschild_pg_cartesian_masked():
    """3. Free-Falling Slicing (PG Cartesian)"""
    t, x, y, z = sp.symbols('t x y z', real=True)
    M = sp.symbols('M', real=True, positive=True)
    r = sp.sqrt(x**2 + y**2 + z**2)

    B_d = sp.Function('B')(x, y, z)
    act_B = sp.sqrt(2*M/r) / r
    beta = [B_d * x, B_d * y, B_d * z]

    g_inv = sp.zeros(4, 4)
    g_inv[0,0] = -1
    for i in range(3):
        g_inv[0, i+1] = beta[i]
        g_inv[i+1, 0] = beta[i]
        for j in range(3):
            g_inv[i+1, j+1] = (1 if i==j else 0) - beta[i]*beta[j]

    Tetrad = sp.Matrix([[1, 0, 0, 0], [beta[0], 1, 0, 0], [beta[1], 0, 1, 0], [beta[2], 0, 0, 1]])
    
    subs_dict = {sp.Derivative(B_d, v): sp.diff(act_B, v) for v in (x,y,z)}
    subs_dict[B_d] = act_B

    return {'coords': [t, x, y, z], 'alpha_func': sp.sympify(1), 'g_inv': g_inv,
            'sqrt_det_g': sp.sympify(1), 'Tetrad': Tetrad, 'n_cov': sp.Matrix([-1, 0, 0, 0]),
            'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'M': M}, 'subs_dict': subs_dict}

def get_interior_schwarzschild_spherical_masked():
    """4. Constant Density Fluid Star (Spherical)"""
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M, R = sp.symbols('M R', real=True, positive=True)

    A_d = sp.Function('Alpha')(r)
    Grr_d = sp.Function('Grr')(r)
    
    term_R = sp.sqrt(1 - 2*M/R)
    term_r = sp.sqrt(1 - (2*M*r**2)/R**3)
    act_A = (3 * term_R - term_r) / 2
    act_Grr = 1 / (term_r**2)
    
    g_inv = sp.diag(-1/(A_d**2), 1/Grr_d, 1/r**2, 1/(r**2 * sp.sin(theta)**2))
    Tetrad = sp.Matrix([[A_d, 0, 0, 0], [0, sp.sqrt(Grr_d), 0, 0], [0, 0, r, 0], [0, 0, 0, r*sp.sin(theta)]])
    
    subs_dict = {
        sp.Derivative(A_d, r): sp.diff(act_A, r), A_d: act_A,
        sp.Derivative(Grr_d, r): sp.diff(act_Grr, r), Grr_d: act_Grr
    }
    
    return {'coords': [t, r, theta, phi], 'alpha_func': A_d, 'g_inv': g_inv,
            'sqrt_det_g': sp.sqrt(Grr_d) * r**2 * sp.sin(theta), 'Tetrad': Tetrad, 
            'n_cov': sp.Matrix([-A_d, 0, 0, 0]), 'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M, 'R': R}, 
            'subs_dict': subs_dict}

# ------------------------------------------
# 2. KERR GEOMETRY FAMILY (THE HEAVYWEIGHTS)
# ------------------------------------------

def get_kerr_boyer_lindquist_masked():
    """5. Kerr Metric (Boyer-Lindquist Spherical) - 2D Mask"""
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M, a = sp.symbols('M a', real=True, positive=True)

    # 2D Dummies (functions of r and theta)
    A_d = sp.Function('Alpha')(r, theta)
    Grr_d = sp.Function('Grr')(r, theta)
    Gtt_d = sp.Function('Gtt')(r, theta)
    Gpp_d = sp.Function('Gpp')(r, theta)
    Beta_d = sp.Function('Beta')(r, theta)

    Sigma = r**2 + a**2 * sp.cos(theta)**2
    Delta = r**2 - 2*M*r + a**2
    A_term = (r**2 + a**2)**2 - a**2 * Delta * sp.sin(theta)**2

    act_A = sp.sqrt(Sigma * Delta / A_term)
    act_Grr = Sigma / Delta
    act_Gtt = Sigma
    act_Gpp = (A_term / Sigma) * sp.sin(theta)**2
    act_Beta = -2 * M * a * r / A_term

    g_inv = sp.Matrix([
        [-1/(A_d**2), 0, 0, Beta_d/(A_d**2)],
        [0, 1/Grr_d, 0, 0],
        [0, 0, 1/Gtt_d, 0],
        [Beta_d/(A_d**2), 0, 0, 1/Gpp_d - (Beta_d**2)/(A_d**2)]
    ])

    Tetrad = sp.Matrix([
        [A_d, 0, 0, 0],
        [0, sp.sqrt(Grr_d), 0, 0],
        [0, 0, sp.sqrt(Gtt_d), 0],
        [Beta_d * sp.sqrt(Gpp_d), 0, 0, sp.sqrt(Gpp_d)]
    ])

    subs_dict = {}
    for var in (r, theta):
        subs_dict.update({
            sp.Derivative(A_d, var): sp.diff(act_A, var),
            sp.Derivative(Grr_d, var): sp.diff(act_Grr, var),
            sp.Derivative(Gtt_d, var): sp.diff(act_Gtt, var),
            sp.Derivative(Gpp_d, var): sp.diff(act_Gpp, var),
            sp.Derivative(Beta_d, var): sp.diff(act_Beta, var)
        })
    subs_dict.update({A_d: act_A, Grr_d: act_Grr, Gtt_d: act_Gtt, Gpp_d: act_Gpp, Beta_d: act_Beta})

    return {'coords': [t, r, theta, phi], 'alpha_func': A_d, 'g_inv': g_inv,
            'sqrt_det_g': A_d * sp.sqrt(Grr_d * Gtt_d * Gpp_d), 'Tetrad': Tetrad, 
            'n_cov': sp.Matrix([-A_d, 0, 0, 0]), 'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M, 'a': a}, 
            'subs_dict': subs_dict}

def get_kerr_schild_cartesian_masked():
    """6. Kerr-Schild Metric (Horizon-Penetrating Cartesian) - Full 3D Mask"""
    t, x, y, z = sp.symbols('t x y z', real=True)
    M, a = sp.symbols('M a', real=True, positive=True)

    R2 = x**2 + y**2 + z**2
    r2 = (R2 - a**2)/2 + sp.sqrt((R2 - a**2)**2 / 4 + a**2 * z**2)
    r_val = sp.sqrt(r2)

    # We must mask the function `f` AND the spatial mapping vectors `l_i`
    F_d = sp.Function('F')(x, y, z)
    Lx_d = sp.Function('Lx')(x, y, z)
    Ly_d = sp.Function('Ly')(x, y, z)
    Lz_d = sp.Function('Lz')(x, y, z)

    act_f = (2 * M * r_val**3) / (r2**2 + a**2 * z**2)
    act_lx = (r_val * x + a * y) / (r2 + a**2)
    act_ly = (r_val * y - a * x) / (r2 + a**2)
    act_lz = z / r_val

    alpha_func = 1 / sp.sqrt(1 + F_d)
    l_vec = [Lx_d, Ly_d, Lz_d]

    Tetrad = sp.zeros(4, 4)
    Tetrad[0, 0] = alpha_func
    for k in range(3):
        Tetrad[k+1, 0] = (F_d / sp.sqrt(1 + F_d)) * l_vec[k]
        for i in range(3):
            Tetrad[k+1, i+1] = (1 if k == i else 0) + (sp.sqrt(1 + F_d) - 1) * l_vec[k] * l_vec[i]

    g_inv = sp.zeros(4, 4)
    l_up = [-1, Lx_d, Ly_d, Lz_d]
    eta = sp.diag(-1, 1, 1, 1)
    for mu in range(4):
        for nu in range(4):
            g_inv[mu, nu] = eta[mu, nu] - F_d * l_up[mu] * l_up[nu]

    subs_dict = {}
    for var in (x, y, z):
        subs_dict.update({
            sp.Derivative(F_d, var): sp.diff(act_f, var),
            sp.Derivative(Lx_d, var): sp.diff(act_lx, var),
            sp.Derivative(Ly_d, var): sp.diff(act_ly, var),
            sp.Derivative(Lz_d, var): sp.diff(act_lz, var)
        })
    subs_dict.update({F_d: act_f, Lx_d: act_lx, Ly_d: act_ly, Lz_d: act_lz})

    return {'coords': [t, x, y, z], 'alpha_func': alpha_func, 'g_inv': g_inv,
            'sqrt_det_g': sp.sympify(1), 'Tetrad': Tetrad, 'n_cov': sp.Matrix([-alpha_func, 0, 0, 0]),
            'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'M': M, 'a': a}, 'subs_dict': subs_dict}

# ------------------------------------------
# 3. EXOTIC SPACETIMES
# ------------------------------------------

def get_reissner_nordstrom_spherical_masked():
    """7. Reissner-Nordström (Spherical)"""
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M, Q = sp.symbols('M Q', real=True, positive=True)
    
    A_d = sp.Function('Alpha')(r)
    act_A = sp.sqrt(1 - (2*M)/r + (Q**2)/r**2)
    
    g_inv = sp.diag(-1/(A_d**2), A_d**2, 1/r**2, 1/(r**2 * sp.sin(theta)**2))
    Tetrad = sp.Matrix([[A_d, 0, 0, 0], [0, 1/A_d, 0, 0], [0, 0, r, 0], [0, 0, 0, r*sp.sin(theta)]])
    
    subs_dict = {sp.Derivative(A_d, r): sp.diff(act_A, r), A_d: act_A}
    
    return {'coords': [t, r, theta, phi], 'alpha_func': A_d, 'g_inv': g_inv, 
            'sqrt_det_g': r**2 * sp.sin(theta), 'Tetrad': Tetrad, 'n_cov': sp.Matrix([-A_d, 0, 0, 0]), 
            'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M, 'Q': Q}, 'subs_dict': subs_dict}

def get_bardeen_cartesian_masked():
    """8. Bardeen Regular Black Hole (Cartesian) - Multi-variable Mask"""
    t, x, y, z = sp.symbols('t x y z', real=True)
    M, Qm = sp.symbols('M Qm', real=True, positive=True) 
    r = sp.sqrt(x**2 + y**2 + z**2)
    
    A_d = sp.Function('Alpha')(x, y, z)
    F_d = sp.Function('F')(x, y, z)
    
    act_f = 1 - (2 * M * r**2) / (r**2 + Qm**2)**(sp.Rational(3, 2))
    act_A = sp.sqrt(act_f)
    B_fact = (1 / A_d) - 1
    
    Tetrad = sp.Matrix([
        [A_d, 0, 0, 0],
        [0, 1 + B_fact*(x**2)/r**2, B_fact*(x*y)/r**2, B_fact*(x*z)/r**2],
        [0, B_fact*(y*x)/r**2, 1 + B_fact*(y**2)/r**2, B_fact*(y*z)/r**2],
        [0, B_fact*(z*x)/r**2, B_fact*(z*y)/r**2, 1 + B_fact*(z**2)/r**2]
    ])
    
    g_inv = sp.zeros(4, 4)
    g_inv[0, 0] = -1 / F_d
    coords = [x, y, z]
    for i, c_i in enumerate(coords):
        for j, c_j in enumerate(coords):
            g_inv[i+1, j+1] = (1 if i == j else 0) + (F_d - 1) * (c_i * c_j) / r**2
            
    subs_dict = {}
    for var in (x, y, z):
        subs_dict.update({
            sp.Derivative(A_d, var): sp.diff(act_A, var),
            sp.Derivative(F_d, var): sp.diff(act_f, var)
        })
    subs_dict.update({A_d: act_A, F_d: act_f})
    
    return {'coords': [t, x, y, z], 'alpha_func': A_d, 'g_inv': g_inv, 
            'sqrt_det_g': sp.sympify(1), 'Tetrad': Tetrad, 'n_cov': sp.Matrix([-A_d, 0, 0, 0]), 
            'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'M': M, 'Qm': Qm}, 'subs_dict': subs_dict}

# ------------------------------------------
# 4. COSMOLOGICAL BACKGROUND
# ------------------------------------------

def get_flrw_cartesian_masked():
    """9. Flat FLRW Cosmological Metric"""
    t, x, y, z = sp.symbols('t x y z', real=True)
    a_t = sp.Function('a')(t) 
    
    g_inv = sp.diag(-1, 1/a_t**2, 1/a_t**2, 1/a_t**2)
    Tetrad = sp.Matrix([[1, 0, 0, 0], [0, a_t, 0, 0], [0, 0, a_t, 0], [0, 0, 0, a_t]])
    
    # FLRW naturally uses an abstract function a(t), so it requires no subs_dict!
    return {'coords': [t, x, y, z], 'alpha_func': sp.sympify(1), 'g_inv': g_inv, 
            'sqrt_det_g': a_t**3, 'Tetrad': Tetrad, 'n_cov': sp.Matrix([-1, 0, 0, 0]), 
            'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'a_t': a_t}, 'subs_dict': {}}
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