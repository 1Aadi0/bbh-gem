import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

DATA_FILE = "final_data.npz"


# --------------------------- I/O --------------------------- #
def load_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load all metric and shift data from a single .npz file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")

    d = np.load(file_path)

    alp = d["alp"]
    gxx, gxy, gyy = d["gxx"], d["gxy"], d["gyy"]
    gxz = d.get("gxz", np.zeros_like(gxx))
    gyz = d.get("gyz", np.zeros_like(gxx))
    gzz = d.get("gzz", np.ones_like(gxx))

    betax, betay = d["betax"], d["betay"]
    betaz = d.get("betaz", np.zeros_like(betax))

    x = d["x"]
    y = d["y"]

    return dict(
        alp=alp,
        gxx=gxx, gxy=gxy, gyy=gyy,
        gxz=gxz, gyz=gyz, gzz=gzz,
        betax=betax, betay=betay, betaz=betaz,
        x=x, y=y,
    )


# --------------------------- Helpers --------------------------- #
def lower_shift(gxx, gxy, gxz, gyy, gyz, gzz, betax, betay, betaz):
    b_x = gxx * betax + gxy * betay + gxz * betaz
    b_y = gxy * betax + gyy * betay + gyz * betaz
    b_z = gxz * betax + gyz * betay + gzz * betaz
    return b_x, b_y, b_z


def coefficients(gxx, gxy, gyy, gxz, gyz):
    A = 1.0 / np.sqrt(gxx)
    det2 = gxx * gyy - gxy**2
    det2 = np.clip(det2, 1e-30, None)
    B = 1.0 / np.sqrt(det2)
    C = gxx * gyz - gxy * gxz
    return A, B, C, det2


def spatial_grads(arr, dx, dy) -> Tuple[np.ndarray, np.ndarray]:
    d_dx, d_dy = np.gradient(arr, dx, dy, edge_order=2)
    return d_dx, d_dy


# --------------------------- Tetrads --------------------------- #
def build_tetrads(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    alp = data["alp"]
    gxx, gxy, gyy = data["gxx"], data["gxy"], data["gyy"]
    gxz, gyz, gzz = data["gxz"], data["gyz"], data["gzz"]
    betax, betay, betaz = data["betax"], data["betay"], data["betaz"]

    b_x, b_y, b_z = lower_shift(gxx, gxy, gxz, gyy, gyz, gzz, betax, betay, betaz)
    A, B, C, det2 = coefficients(gxx, gxy, gyy, gxz, gyz)
    B2 = B**2

    g1 = dict(t=b_x, x=gxx, y=gxy, z=gxz)
    g2 = dict(t=b_y, x=gxy, y=gyy, z=gyz)
    g3 = dict(t=b_z, x=gxz, y=gyz, z=gzz)

    theta = {}

    # hat{0}
    theta["0_t"] = alp
    theta["0_x"] = np.zeros_like(alp)
    theta["0_y"] = np.zeros_like(alp)
    theta["0_z"] = np.zeros_like(alp)

    # hat{1}
    theta["1_t"] = A * g1["t"]
    theta["1_x"] = A * g1["x"]
    theta["1_y"] = A * g1["y"]
    theta["1_z"] = A * g1["z"]

    # hat{2}
    theta["2_t"] = A * B * (gxx * g2["t"] - gxy * g1["t"])
    theta["2_x"] = A * B * (gxx * g2["x"] - gxy * g1["x"])
    theta["2_y"] = A * B * (gxx * g2["y"] - gxy * g1["y"])
    theta["2_z"] = A * B * (gxx * g2["z"] - gxy * g1["z"])

    # hat{3}
    for comp, g3nu, g2nu, g1nu in [
        ("t", g3["t"], g2["t"], g1["t"]),
        ("x", g3["x"], g2["x"], g1["x"]),
        ("y", g3["y"], g2["y"], g1["y"]),
        ("z", g3["z"], g2["z"], g1["z"]),
    ]:
        term1 = gxx * g3nu - gxz * g1nu
        term2 = (gxx * g2nu - gxy * g1nu) * B2 * C
        theta[f"3_{comp}"] = A * (term1 - term2)

    theta["det2"] = det2
    theta["A"], theta["B"], theta["C"] = A, B, C
    theta["b_x"], theta["b_y"], theta["b_z"] = b_x, b_y, b_z
    return theta


# --------------------------- Fields --------------------------- #
def field_strength(theta: Dict[str, np.ndarray], dx: float, dy: float):
    F = {k: {} for k in ["0", "1", "2", "3"]}

    for hat in ["0", "1", "2", "3"]:
        th_t = theta[f"{hat}_t"]
        th_x = theta[f"{hat}_x"]
        th_y = theta[f"{hat}_y"]

        dth_t_dx, dth_t_dy = spatial_grads(th_t, dx, dy)
        dth_x_dx, dth_x_dy = spatial_grads(th_x, dx, dy)
        dth_y_dx, dth_y_dy = spatial_grads(th_y, dx, dy)

        F[hat]["F_xt"] = dth_t_dx
        F[hat]["F_yt"] = dth_t_dy
        F[hat]["F_xy"] = dth_y_dx - dth_x_dy

    return F


def gem_fields(theta: Dict[str, np.ndarray], F: Dict[str, Dict[str, np.ndarray]]):
    alp = theta["0_t"]
    n_t = 1.0 / np.clip(alp, 1e-30, None)

    E = {}
    for hat in ["0", "1", "2", "3"]:
        E_xt = n_t * F[hat]["F_xt"]
        E_yt = n_t * F[hat]["F_yt"]
        E[hat] = (E_xt, E_yt)

    Bz = {hat: F[hat]["F_xy"] for hat in ["0", "1", "2", "3"]}
    return E, Bz


def shift_curl_bz(betax, betay, dx, dy):
    dbetay_dx, _ = spatial_grads(betay, dx, dy)
    _, dbetax_dy = spatial_grads(betax, dx, dy)
    return dbetay_dx - dbetax_dy


# --------------------------- Orthonormality --------------------------- #
def orthonormal_checks(theta: Dict[str, np.ndarray], gxx, gxy, gyy, gxz, gyz, gzz):
    def inner(a_idx, b_idx):
        a = {c: theta[f"{a_idx}_{c}"] for c in ["t", "x", "y", "z"]}
        b = {c: theta[f"{b_idx}_{c}"] for c in ["t", "x", "y", "z"]}
        gtt = -np.ones_like(gxx)
        term = (
            gtt * a["t"] * b["t"]
            + gxx * a["x"] * b["x"]
            + 2 * gxy * a["x"] * b["y"]
            + 2 * gxz * a["x"] * b["z"]
            + gyy * a["y"] * b["y"]
            + 2 * gyz * a["y"] * b["z"]
            + gzz * a["z"] * b["z"]
        )
        return float(np.mean(term))

    checks = {}
    for a in ["0", "1", "2", "3"]:
        for b in ["0", "1", "2", "3"]:
            checks[f"{a}{b}"] = inner(a, b)
    return checks


# --------------------------- Rotation (Appendix B) --------------------------- #
def rotate_tetrad_in_plane(theta: Dict[str, np.ndarray], angle_rad: float) -> Dict[str, np.ndarray]:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotated = theta.copy()
    for comp in ["t", "x", "y", "z"]:
        e1 = theta[f"1_{comp}"]
        e2 = theta[f"2_{comp}"]
        rotated[f"1_{comp}"] = c * e1 - s * e2
        rotated[f"2_{comp}"] = s * e1 + c * e2
    return rotated


# --------------------------- Main --------------------------- #
def main(
    data_path: str = DATA_FILE,
    rotation_angle_rad: Optional[float] = None,
    out_png: str = "Paper_Method_Tetrads.png",
):
    data = load_data(data_path)
    x, y = data["x"], data["y"]
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    theta = build_tetrads(data)
    if rotation_angle_rad is not None:
        theta = rotate_tetrad_in_plane(theta, rotation_angle_rad)

    F = field_strength(theta, dx, dy)
    E, Bz = gem_fields(theta, F)
    Bz_shift = shift_curl_bz(data["betax"], data["betay"], dx, dy)
    checks = orthonormal_checks(
        theta, data["gxx"], data["gxy"], data["gyy"], data["gxz"], data["gyz"], data["gzz"]
    )

    print("Grid shape:", data["alp"].shape)
    print("det2 min/max:", theta["det2"].min(), theta["det2"].max())
    print("Orthonormality checks <g(theta^a, theta^b)> (diag target [-1,1,1,1]):")
    for k, v in checks.items():
        print(f"  g({k}) = {v:+.6e}")

   # ... inside main() ...

    # Visualize hat{0} fields
    # TRANSPOSE EVERYTHING passed to plotting functions (.T)
        # Visualize hat{0} fields
        # Visualize hat{0} fields
        # Visualize hat{0} fields
    Xc, Yc = np.meshgrid(x, y)      # shapes (57,123)
    x_edges = np.concatenate([x - dx/2, [x[-1] + dx/2]])
    y_edges = np.concatenate([y - dy/2, [y[-1] + dy/2]])

    Ex0, Ey0 = E["0"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Electric field
    im1 = axes[0].pcolormesh(
        x_edges, y_edges, data["alp"].T,
        cmap="Reds", shading="flat", vmin=0.0, vmax=1.0
    )
    axes[0].streamplot(
        Xc, Yc, Ex0.T, Ey0.T,
        color="black", density=1.5, arrowsize=1.2
    )
    axes[0].set_aspect("equal")
    fig.colorbar(im1, ax=axes[0])

    # Magnetic
    limit = np.max(np.abs(Bz_shift)) * 0.8 or 0.1
    im2 = axes[1].pcolormesh(
        x_edges, y_edges, Bz_shift.T,
        cmap="RdBu_r", shading="flat", vmin=-limit, vmax=limit
    )
    cs = axes[1].contour(Xc, Yc, Bz_shift.T, colors="k", linewidths=0.8)
    axes[1].clabel(cs)
    axes[1].set_aspect("equal")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)

if __name__ == "__main__":
    main()


    






