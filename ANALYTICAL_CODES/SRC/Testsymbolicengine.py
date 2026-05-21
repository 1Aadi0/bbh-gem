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

import sympy as sp

def get_schwarzschild_isotropic_cartesian_masked():
    """
    2. Masked Static Slicing with Cartesian Tetrad.
    Fixed: Uses multi-variable dummies to completely bypass SymPy's chain-rule expression bugs.
    """
    import sympy as sp
    t, x, y, z = sp.symbols('t x y z', real=True)
    M = sp.symbols('M', real=True, positive=True)
    
    r_bar = sp.sqrt(x**2 + y**2 + z**2)
    
    # --- ALGEBRAIC MASKING (Multi-Variable Approach) ---
    # Define dummies as explicit functions of the coordinate symbols
    Psi_dummy = sp.Function('Psi')(x, y, z)
    Alpha_dummy = sp.Function('Alpha')(x, y, z)
    
    # Define the real expressions
    actual_Psi = 1 + M / (2 * r_bar)
    actual_Alpha = (1 - M / (2 * r_bar)) / actual_Psi
    # -------------------------
    
    # Build Inverse Metric using Dummies
    g_inv = sp.diag(-1/(Alpha_dummy**2), 1/Psi_dummy**4, 1/Psi_dummy**4, 1/Psi_dummy**4)
    
    # Build Tetrad using Dummies (Diagonal in Cartesian)
    Tetrad = sp.Matrix([
        [Alpha_dummy, 0, 0, 0],
        [0, Psi_dummy**2, 0, 0],
        [0, 0, Psi_dummy**2, 0],
        [0, 0, 0, Psi_dummy**2]
    ])
    
    sqrt_det_g = Alpha_dummy * Psi_dummy**6
    n_cov = sp.Matrix([-Alpha_dummy, 0, 0, 0])
    
    # Package the substitution dictionary mapping explicit partials
    # This completely avoids putting an expression in the denominator of a Derivative!
    subs_dict = {
        sp.Derivative(Psi_dummy, x): sp.diff(actual_Psi, x),
        sp.Derivative(Psi_dummy, y): sp.diff(actual_Psi, y),
        sp.Derivative(Psi_dummy, z): sp.diff(actual_Psi, z),
        Psi_dummy: actual_Psi,
        
        sp.Derivative(Alpha_dummy, x): sp.diff(actual_Alpha, x),
        sp.Derivative(Alpha_dummy, y): sp.diff(actual_Alpha, y),
        sp.Derivative(Alpha_dummy, z): sp.diff(actual_Alpha, z),
        Alpha_dummy: actual_Alpha
    }
    
    return {
        'coords': [t, x, y, z], 'alpha_func': Alpha_dummy, 'g_inv': g_inv,
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov,
        'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'M': M, 'r_bar': r_bar},
        'subs_dict': subs_dict
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

def get_bardeen_cartesian():
    """Bardeen Regular Black Hole (Cartesian Coordinates)"""
    import sympy as sp
    
    t, x, y, z = sp.symbols('t x y z', real=True)
    M, Qm = sp.symbols('M Qm', real=True, positive=True) 
    
    r = sp.sqrt(x**2 + y**2 + z**2)
    
    # The Bardeen lapse function
    f = 1 - (2 * M * r**2) / (r**2 + Qm**2)**(sp.Rational(3, 2))
    alpha_func = sp.sqrt(f)
    
    # B factor for stretching the spatial tetrad along the radial vector
    B = (1 / alpha_func) - 1
    
    # 1. The Covariant Tetrad Matrix (A^a_\mu)
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],
        [0, 1 + B*(x**2)/r**2,     B*(x*y)/r**2,       B*(x*z)/r**2],
        [0,   B*(y*x)/r**2,   1 + B*(y**2)/r**2,       B*(y*z)/r**2],
        [0,   B*(z*x)/r**2,     B*(z*y)/r**2,     1 + B*(z**2)/r**2]
    ])
    
    # 2. The Inverse Metric (g^{\mu \nu})
    # g^{ij} = \delta^{ij} + (f - 1) * (x^i x^j) / r^2
    C_inv = f - 1
    g_inv = sp.zeros(4, 4)
    g_inv[0, 0] = -1 / f
    
    coords_spatial = [x, y, z]
    for i, c_i in enumerate(coords_spatial):
        for j, c_j in enumerate(coords_spatial):
            delta = 1 if i == j else 0
            g_inv[i+1, j+1] = delta + C_inv * (c_i * c_j) / r**2
            
    # 3. Determinant of the metric
    # In Cartesian mapping of spherical metrics, det(g) = 1 because the 
    # r^2 sin(theta) Jacobian perfectly cancels out.
    sqrt_det_g = sp.sympify(1)
    
    # 4. Normal vector
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    
    return {
        'coords': [t, x, y, z], 
        'alpha_func': alpha_func, 
        'g_inv': g_inv, 
        'sqrt_det_g': sqrt_det_g, 
        'Tetrad': Tetrad, 
        'n_cov': n_cov, 
        'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'M': M, 'Qm': Qm}
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

def get_painleve_gullstrand_spherical():
    """
    Painlevé-Gullstrand (Free-Falling) Coordinates for Schwarzschild.
    This represents an Eulerian observer in free-fall from infinity.
    """
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M = sp.symbols('M', real=True, positive=True)

    # In PG coordinates, the lapse is identically 1 (proper time)
    alpha_func = sp.sympify(1)
    
    # The shift vector is beta^r = sqrt(2M/r)
    beta_r = sp.sqrt(2*M/r)
    
    # The inverse metric g^{\mu\nu}
    g_inv = sp.Matrix([
        [-1,      beta_r,             0, 0],
        [beta_r,  1 - beta_r**2,      0, 0],
        [0,       0,             1/r**2, 0],
        [0,       0,                  0, 1/(r**2 * sp.sin(theta)**2)]
    ])

    # The spatial slices are flat, so det(g) is the same as flat space
    sqrt_det_g = r**2 * sp.sin(theta)

    # Tetrad: e^0 = dt, e^1 = dr + beta^r dt, e^2 = r dtheta, e^3 = r sin(theta) dphi
    Tetrad = sp.Matrix([
        [1, 0, 0, 0],
        [beta_r, 1, 0, 0],
        [0, 0, r, 0],
        [0, 0, 0, r*sp.sin(theta)]
    ])
    
    # The normal co-vector n_\mu = (-\alpha, 0, 0, 0)
    n_cov = sp.Matrix([-1, 0, 0, 0])

    return {
        'coords': [t, r, theta, phi], 'alpha_func': alpha_func, 'g_inv': g_inv,
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov,
        'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M}
    }


def get_schwarzschild_kerr_schild_cartesian_masked():
    """5. Masked Ingoing Null Slicing with Cartesian Tetrad"""
    import sympy as sp
    t, x, y, z = sp.symbols('t x y z', real=True)
    M = sp.symbols('M', real=True, positive=True)

    r = sp.sqrt(x**2 + y**2 + z**2)
    
    # --- ALGEBRAIC MASKING ---
    f_dummy = sp.Function('F')(r)
    actual_f = 2 * M / r
    actual_df = sp.diff(actual_f, r)
    # -------------------------

    l_x, l_y, l_z = x/r, y/r, z/r
    l_vec = [l_x, l_y, l_z]

    alpha_func = 1 / sp.sqrt(1 + f_dummy)

    Tetrad = sp.zeros(4, 4)
    Tetrad[0, 0] = alpha_func
    for k in range(3):
        Tetrad[k+1, 0] = (f_dummy / sp.sqrt(1 + f_dummy)) * l_vec[k]
        for i in range(3):
            delta_ki = 1 if k == i else 0
            Tetrad[k+1, i+1] = delta_ki + (sp.sqrt(1 + f_dummy) - 1) * l_vec[k] * l_vec[i]

    eta = sp.diag(-1, 1, 1, 1)
    l_up = [-1, l_x, l_y, l_z]

    g_inv = sp.zeros(4, 4)
    for mu in range(4):
        for nu in range(4):
            g_inv[mu, nu] = eta[mu, nu] - f_dummy * l_up[mu] * l_up[nu]

    sqrt_det_g = sp.sympify(1)
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])

    return {
        'coords': [t, x, y, z], 'alpha_func': alpha_func, 'g_inv': g_inv,
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov,
        'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'M': M},
        # Pass the substitution dictionary out to the engine
        'subs_dict': {sp.Derivative(f_dummy, r): actual_df, f_dummy: actual_f}
    }


def get_schwarzschild_pg_cartesian_masked():
    """4. Masked Free-Falling Slicing with Cartesian Tetrad (rho=0 when M=0)"""
    import sympy as sp
    t, x, y, z = sp.symbols('t x y z', real=True)
    M = sp.symbols('M', real=True, positive=True)

    r = sp.sqrt(x**2 + y**2 + z**2)
    alpha_func = sp.sympify(1)

    # --- ALGEBRAIC MASKING ---
    B_dummy = sp.Function('B')(r)
    actual_B = sp.sqrt(2*M/r) / r
    actual_dB = sp.diff(actual_B, r)
    # -------------------------

    beta = [B_dummy * x, B_dummy * y, B_dummy * z]

    g_inv = sp.zeros(4, 4)
    g_inv[0,0] = -1
    for i in range(3):
        g_inv[0, i+1] = beta[i]
        g_inv[i+1, 0] = beta[i]
        for j in range(3):
            delta = 1 if i==j else 0
            g_inv[i+1, j+1] = delta - beta[i]*beta[j]

    sqrt_det_g = sp.sympify(1)

    Tetrad = sp.Matrix([
        [1, 0, 0, 0],
        [beta[0], 1, 0, 0],
        [beta[1], 0, 1, 0],
        [beta[2], 0, 0, 1]
    ])
    n_cov = sp.Matrix([-1, 0, 0, 0])

    return {
        'coords': [t, x, y, z], 'alpha_func': alpha_func, 'g_inv': g_inv,
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov,
        'symbols': {'t': t, 'x': x, 'y': y, 'z': z, 'M': M},
        # Pass the substitution dictionary out to the engine
        'subs_dict': {sp.Derivative(B_dummy, r): actual_dB, B_dummy: actual_B}
    }

def get_hayward_spherical():
    """
    Hayward Regular Black Hole (Spherical Coordinates)
    Resolves the r=0 singularity using a fundamental length scale 'l'.
    """
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M, l = sp.symbols('M l', real=True, positive=True)
    
    # Hayward modification to the Schwarzschild f(r)
    f = 1 - (2 * M * r**2) / (r**3 + 2 * M * l**2)
    alpha_func = sp.sqrt(f)
    
    g_cov = sp.diag(-(alpha_func**2), 1/f, r**2, r**2 * sp.sin(theta)**2)
    g_inv = sp.diag(-1/(alpha_func**2), f, 1/r**2, 1/(r**2 * sp.sin(theta)**2))
    
    # Exact known volume element (g_tt * g_rr = -1)
    sqrt_det_g = r**2 * sp.sin(theta)
    
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],
        [0, sp.sqrt(1/f), 0, 0],
        [0, 0, r, 0],
        [0, 0, 0, r*sp.sin(theta)]
    ])
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    
    return {
        'coords': [t, r, theta, phi], 'alpha_func': alpha_func, 'g_inv': g_inv, 
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov, 
        'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M, 'l': l}
    }

def get_bronnikov_ellis_spherical():
    """
    Bronnikov-Ellis Geometry / Traversable Wormhole (Spherical Coordinates)
    Metric #3 from Sayan Kar's note.
    Corrected: Spatial metric uses 1 - b0^2/r^2
    """
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    b0 = sp.symbols('b0', real=True, positive=True) # Throat parameter
    
    alpha_func = sp.sympify(1) # Zero tidal forces in time
    f_r = 1 - (b0**2)/(r**2)
    
    g_cov = sp.diag(-1, 1/f_r, r**2, r**2 * sp.sin(theta)**2)
    g_inv = sp.diag(-1, f_r, 1/r**2, 1/(r**2 * sp.sin(theta)**2))
    
    # Custom volume element since g_tt * g_rr != -1
    sqrt_det_g = sp.sqrt(1/f_r) * r**2 * sp.sin(theta)
    
    Tetrad = sp.Matrix([
        [1, 0, 0, 0],
        [0, sp.sqrt(1/f_r), 0, 0],
        [0, 0, r, 0],
        [0, 0, 0, r*sp.sin(theta)]
    ])
    n_cov = sp.Matrix([-1, 0, 0, 0])
    
    return {
        'coords': [t, r, theta, phi], 'alpha_func': alpha_func, 'g_inv': g_inv, 
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov, 
        'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'b0': b0}
    }

def get_zero_tidal_schwarzschild_spherical():
    """
    Wormhole Schwarzschild / Zero Tidal Force (Spherical Coordinates)
    Metric #2 from Sayan Kar's note.
    """
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M = sp.symbols('M', real=True, positive=True)
    
    alpha_func = sp.sympify(1) # Time flows normally everywhere
    f_r = 1 - 2*M/r
    
    g_cov = sp.diag(-1, 1/f_r, r**2, r**2 * sp.sin(theta)**2)
    g_inv = sp.diag(-1, f_r, 1/r**2, 1/(r**2 * sp.sin(theta)**2))
    
    # Custom volume element since g_tt * g_rr != -1
    sqrt_det_g = sp.sqrt(1/f_r) * r**2 * sp.sin(theta)
    
    Tetrad = sp.Matrix([
        [1, 0, 0, 0],
        [0, sp.sqrt(1/f_r), 0, 0],
        [0, 0, r, 0],
        [0, 0, 0, r*sp.sin(theta)]
    ])
    n_cov = sp.Matrix([-1, 0, 0, 0])
    
    return {
        'coords': [t, r, theta, phi], 'alpha_func': alpha_func, 'g_inv': g_inv, 
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov, 
        'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M}
    }

def get_kappa_lambda_wormhole_spherical():
    """
    Non-Singular Wormhole (kappa, lambda variant) (Spherical Coordinates)
    Metric #1 from Sayan Kar's note.
    """
    t, r, theta, phi = sp.symbols('t r theta phi', real=True)
    M, kappa, lam = sp.symbols('M kappa lambda', real=True, positive=True)
    
    f_r = 1 - 2*M/r
    alpha_func = kappa + lam * sp.sqrt(f_r)
    
    g_cov = sp.diag(-(alpha_func**2), 1/f_r, r**2, r**2 * sp.sin(theta)**2)
    g_inv = sp.diag(-1/(alpha_func**2), f_r, 1/r**2, 1/(r**2 * sp.sin(theta)**2))
    
    # Custom volume element
    sqrt_det_g = alpha_func * sp.sqrt(1/f_r) * r**2 * sp.sin(theta)
    
    Tetrad = sp.Matrix([
        [alpha_func, 0, 0, 0],
        [0, sp.sqrt(1/f_r), 0, 0],
        [0, 0, r, 0],
        [0, 0, 0, r*sp.sin(theta)]
    ])
    n_cov = sp.Matrix([-alpha_func, 0, 0, 0])
    
    return {
        'coords': [t, r, theta, phi], 'alpha_func': alpha_func, 'g_inv': g_inv, 
        'sqrt_det_g': sqrt_det_g, 'Tetrad': Tetrad, 'n_cov': n_cov, 
        'symbols': {'t': t, 'r': r, 'theta': theta, 'phi': phi, 'M': M, 'kappa': kappa, 'lambda': lam}
    }


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

    # def calculate_invariants(E_hat, B_hat, D_hat, H_hat):
    #     def contract_4d_scalars(T1, T1_type, T2, T2_type):
    #         """
    #         Universally contracts two 4D tensors into a scalar.
    #         Handles 'up-up' (E, B) and 'up-down' (D, H) index types.
    #         Applies the Minkowski signature eta_ab = diag(-1, 1, 1, 1).
    #         """
    #         scalar = sp.sympify(0)
    #         eta = [-1, 1, 1, 1] # Minkowski metric
            
    #         for a in range(4):
    #             for b in range(4):
    #                 # STEP 1: Convert everything to pure contravariant (up-up)
    #                 # If 'up-down' (like D^a_b), we raise the second index: T^{ab} = T^a_b * eta^{bb}
    #                 val1_up_up = T1[a, b] * eta[b] if T1_type == "up-down" else T1[a, b]
    #                 val2_up_up = T2[a, b] * eta[b] if T2_type == "up-down" else T2[a, b]
                    
    #                 # STEP 2: Lower the indices of the second tensor to prepare for contraction
    #                 # T_{ab} = eta_aa * eta_bb * T^{ab}
    #                 val2_down_down = val2_up_up * eta[a] * eta[b]
                    
    #                 # STEP 3: Perform the invariant sum (T1^{ab} * T2_{ab})
    #                 scalar += val1_up_up * val2_down_down
                    
    #         return sp.cancel(scalar)

    #     # 1. Field Strength Invariants (Contravariant x Contravariant)
    #     E_sq = contract_4d_scalars(E_hat, "up-up", E_hat, "up-up")
    #     B_sq = contract_4d_scalars(B_hat, "up-up", B_hat, "up-up")
    #     L_fields = E_sq - B_sq
    #     Pontryagin = contract_4d_scalars(E_hat, "up-up", B_hat, "up-up")

    #     # 2. Macroscopic Constitutive Invariants (Mixed x Mixed)
    #     D_sq = contract_4d_scalars(D_hat, "up-down", D_hat, "up-down")
    #     H_sq = contract_4d_scalars(H_hat, "up-down", H_hat, "up-down")
    #     L_macro = D_sq - H_sq
    #     Macro_Twist = contract_4d_scalars(D_hat, "up-down", H_hat, "up-down")

    #     # 3. Cross Invariants (Contravariant x Mixed)
    #     ED_contract = contract_4d_scalars(E_hat, "up-up", D_hat, "up-down")
    #     BH_contract = contract_4d_scalars(B_hat, "up-up", H_hat, "up-down")

    #     return {
    #         'E_sq': E_sq, 'B_sq': B_sq, 'L_fields': L_fields, 'Pontryagin': Pontryagin,
    #         'D_sq': D_sq, 'H_sq': H_sq, 'L_macro': L_macro, 'Macro_Twist': Macro_Twist,
    #         'ED_contract': ED_contract, 'BH_contract': BH_contract
    #     }

    # # Execute Engine
    # E_Results, B_Results = [], []
    # for i in range(4):
    #     E, B = get_dynamical_fields(i)
    #     E_Results.append(E)
    #     B_Results.append(B)

    # E_hat, B_hat, D_hat, H_hat = constitutive_relations(E_Results, B_Results, Tetrad)
    # rho_hat, s_hat = calculate_charges_and_currents(E_hat, B_hat, D_hat, H_hat)
    # q_hat, j_hat = calculate_PRL_charges_and_currents(E_hat, B_hat, D_hat, H_hat, sqrt_det_g, alpha_func)
    
    # # Run the new invariants calculator
    # invariants = calculate_invariants(E_hat, B_hat, D_hat, H_hat)
    
    # return {
    #     'E_hat': E_hat, 'B_hat': B_hat, 
    #     'D_hat': D_hat, 'H_hat': H_hat,
    #     'rho_hat': rho_hat, 's_hat': s_hat,
    #     'q_hat': q_hat, 'j_hat': j_hat,
    #     'invariants': invariants,           # Added to output
    #     'symbols': metric_data['symbols']
    #     }