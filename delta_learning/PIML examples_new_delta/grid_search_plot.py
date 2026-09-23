

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # keep for 3D
import matplotlib.tri as mtri
from matplotlib.colors import PowerNorm, LogNorm, Normalize, ListedColormap

plt.rcParams.update({'image.cmap': 'viridis'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
                                    'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
                                    'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
                                    'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.fontset': 'custom'})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.it': 'serif:italic'})
plt.rcParams.update({'mathtext.bf': 'serif:bold'})
plt.close('all')

# 
xlsx_path = r"hyperparameters_leaky_relu.xlsx"
assert os.path.exists(xlsx_path), f"Excel not found: {xlsx_path}"

df0 = pd.read_excel(xlsx_path, header=None)

header_row = 1
start_data_row = 2

# hidden-layer headers (row 1, cols >= 2)
hl, hl_pos = [], []
for j, v in enumerate(df0.iloc[header_row, 2:].tolist(), start=2):
    try:
        val = float(v)
        if not np.isnan(val):
            hl.append(int(val) if float(val).is_integer() else float(val))
            hl_pos.append(j)
    except Exception:
        pass

# node values (col 1, rows >= 2)
nodes, node_rows = [], []
for i, v in enumerate(df0.iloc[start_data_row:, 1].tolist(), start=start_data_row):
    try:
        val = float(v)
        if not np.isnan(val):
            nodes.append(int(val) if float(val).is_integer() else float(val))
            node_rows.append(i)
    except Exception:
        pass

# Build tidy table
recs = []
for i, n in zip(node_rows, nodes):
    for j, h in zip(hl_pos, hl):
        v = df0.iat[i, j]
        try:
            rmse = float(v)
        except Exception:
            rmse = np.nan
        if not (rmse is None or (isinstance(rmse, float) and np.isnan(rmse))):
            recs.append({"hidden_layers": h, "nodes": n, "RMSE": rmse})

tidy = pd.DataFrame.from_records(recs)

# 
def _fmt_val(v):
    if v == 0:
        return "0"
    av = abs(v)
    if 1e-2 <= av < 1e3:
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return s
    return f"{v:.2e}"

def apply_3d_common(ax, p):
    if p.get("xlabel"): ax.set_xlabel(p["xlabel"], fontsize=p.get("label_size", 14))
    if p.get("ylabel"): ax.set_ylabel(p["ylabel"], fontsize=p.get("label_size", 14))
    if p.get("zlabel"): ax.set_zlabel(p["zlabel"], fontsize=p.get("label_size", 14),
                                      rotation=p.get("zlabel_rotation", 90))
    ax.tick_params(axis="both", labelsize=p.get("tick_size", 14))
    ax.tick_params(axis="z", labelsize=p.get("tick_size", 14))
    if p.get("show_grid", True):
        ax.xaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": ":"})
        ax.yaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": ":"})
        ax.zaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": ":"})

# prepare arrays & ranges 
X = tidy["hidden_layers"].to_numpy(float)
Y = tidy["nodes"].to_numpy(float)
Z_lin = tidy["RMSE"].to_numpy(float)
Z_log = np.log10(np.where(Z_lin > 0, Z_lin, 1e-6))   # geometry only

tri = mtri.Triangulation(X, Y)

COLOR_VMIN = .2 # changed minimum value for colorbar 
COLOR_VMAX = 1 # changed maximum value for colorbar

cmap = plt.get_cmap("viridis")

#  plot 1 (linear)_ uses PowerNorm for coloring 
GAMMA = 0.55
norm_linear_colors = PowerNorm(gamma=GAMMA, vmin=COLOR_VMIN, vmax=COLOR_VMAX)

# 
def build_warped_cmap(base_cmap, norm, vmin, vmax, N=256):
    vals = np.linspace(vmin, vmax, N)
    pos = norm(vals)               
    colors = base_cmap(pos)
    return ListedColormap(colors)

warped_cmap_linearbar = build_warped_cmap(cmap, norm_linear_colors, COLOR_VMIN, COLOR_VMAX)
lin_norm_for_bar = Normalize(vmin=COLOR_VMIN, vmax=COLOR_VMAX)

# 
tri_vals = Z_lin[tri.triangles].max(axis=1)
facecolors_linear = cmap(norm_linear_colors(tri_vals))

#  plot 2 (log) 
EPS = 1e-6
vmin_log = COLOR_VMIN
vmax_log = COLOR_VMAX
norm_log_colors = LogNorm(vmin=vmin_log, vmax=vmax_log)
facecolors_log = cmap(norm_log_colors(tri_vals))

# colorbar ticks
n_ticks = 5
ticks_linear = np.linspace(COLOR_VMIN, COLOR_VMAX, n_ticks)
ticks_linear = np.linspace(COLOR_VMIN, COLOR_VMAX, n_ticks)
labels_linear = [_fmt_val(t) for t in ticks_linear]
kmin=np.log10(.2) # set kmin to be log10(.2)
kmax = int(np.log10(vmax_log))
ticks_log = np.logspace(kmin,kmax,n_ticks)
ticks_log=np.round(ticks_log,2)
labels_log = [_fmt_val(t) for t in ticks_log]

# Plot 1: 3D Surface (Linear Z) 
P1 = {
    "figsize": (3.8,3.6),
    "elev": 28, "azim": 45,
    "surface_alpha": 1.0, "surface_linewidth": 0.2,
    "show_scatter": True, "scatter_size": 28, "scatter_edgecolor": "k",
    "xlabel": "hidden layers", "ylabel": "nodes per layer", "zlabel": "RMSE",
    "zlabel_rotation": 90,
    "label_size": 11, "tick_size": 11,
    "dpi": 500,
    "save_path": os.path.join(os.path.dirname(xlsx_path), "rmse_3d_linear.png"),
} # changed azimuth to 45 for both plots; consistent with Part III plots

fig = plt.figure(figsize=P1["figsize"])
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=P1["elev"], azim=P1["azim"])

surf1 = ax.plot_trisurf(tri, Z_lin,
                        linewidth=P1["surface_linewidth"],
                        antialiased=True,
                        shade=False,
                        alpha=P1["surface_alpha"])
surf1.set_facecolors(facecolors_linear)

if P1["show_scatter"]:
    ax.scatter(X, Y, Z_lin, c=Z_lin, cmap=cmap, norm=norm_linear_colors,
               s=P1["scatter_size"], edgecolor=P1["scatter_edgecolor"])

# change z-tick labels
ax.set_zlim(.2, 1) # z axis from .2 to 1
ax.set_ylim(0,100) # changed nodes layer to include 0

# colorbar adjustment for plot 1
sm1 = plt.cm.ScalarMappable(cmap=warped_cmap_linearbar, 
                            norm=lin_norm_for_bar);sm1.set_array([])
cb1 = fig.colorbar(sm1, ax=ax, shrink=1, aspect=14, pad=0.06,location='top')
cb1.set_label("RMSE", fontsize=P1["label_size"])
cb1.ax.tick_params(labelsize=P1['tick_size'])
cb1.set_ticks(ticks_linear); cb1.set_ticklabels(labels_linear)

apply_3d_common(ax, P1)
ax.set_position([0.2,.1,.65,.65]) # adjusted position of the graph
plt.savefig(P1["save_path"], dpi=P1["dpi"], bbox_inches="tight")
print("Saved:", P1["save_path"])

#%%  plot 3- 3D Surface (TRUE log Z via log10 transform) =========
P2 = {
    "figsize": (3.8,3.6),
    "elev": 28, "azim": 45,
    "surface_alpha": 1.0, "surface_linewidth": 0.2,
    "show_scatter": True, "scatter_size": 28, "scatter_edgecolor": "k",
    "xlabel": "hidden layers", "ylabel": "nodes per layer", "zlabel": "RMSE",
    "zlabel_rotation": 90,
    "label_size": 11, "tick_size": 11,
    "dpi": 500,
    "save_path": os.path.join(os.path.dirname(xlsx_path), "rmse_3d_logtrue.png"),
}

fig = plt.figure(figsize=P2["figsize"])
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=P2["elev"], azim=P2["azim"])

surf2 = ax.plot_trisurf(tri, Z_log,
                        linewidth=P2["surface_linewidth"],
                        antialiased=True,
                        shade=False,
                        alpha=P2["surface_alpha"])
surf2.set_facecolors(facecolors_log)

if P2["show_scatter"]:
    ax.scatter(X, Y, Z_log, c=Z_lin, cmap=cmap, norm=norm_log_colors,
               s=P2["scatter_size"], edgecolor=P2["scatter_edgecolor"])

# Z-axis ticks in log10 units
kmin_axis = int(np.floor(np.nanmin(Z_log)))
kmax_axis = int(np.ceil(np.nanmax(Z_log)))
kmin_axis=np.log10(.2) # set kmin_axis to be log10(.2)
ax.set_zticks(np.linspace(kmin_axis,kmax_axis+1,2))
ytick=np.linspace(kmin_axis,kmax_axis,n_ticks) # 
ax.set_zticks(ytick) 
ax.set_zticklabels([rf"${10**k:.1f}$" for k in ytick])
ax.set_zlim(kmin_axis, kmax_axis)
ax.set_ylim(0,100) # changed nodes layer to include 0

# colorbar for Plot 2
sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_log_colors); sm2.set_array([])
cb2 = fig.colorbar(sm2, ax=ax, shrink=1, aspect=14, pad=0.06,location='top')
cb2.set_label("RMSE (log scale)", fontsize=P2["label_size"])
cb2.ax.tick_params(labelsize=P2["tick_size"])
cb2.set_ticks(ticks_log); cb2.set_ticklabels(labels_log)

apply_3d_common(ax, P2)
ax.set_position([0.2,.1,.65,.65]) # adjusted position of the graph
plt.savefig(P2["save_path"], dpi=P2["dpi"], bbox_inches="tight")
print("Saved:", P2["save_path"])
#%% if needed
# # Plot 3: 2D Heatmap (same numeric span as linear)
# P3 = {
#     "figsize": (8, 6),
#     "cmap": "viridis",
#     "xlabel": "Hidden layers", "ylabel": "Nodes per layer",
#     "label_size": 12, "tick_size": 10,
#     "vmin": COLOR_VMIN, "vmax": COLOR_VMAX,
#     "show_grid": True,
#     "dpi": 300,
#     "save_path": os.path.join(os.path.dirname(xlsx_path), "rmse_heatmap.png"),
#     "show_colorbar": True, "cbar_label": "RMSE",
# }

# heatmap_df = tidy.pivot_table(index="nodes", columns="hidden_layers", values="RMSE")
# fig, ax = plt.subplots(figsize=P3["figsize"])
# im = ax.imshow(
#     heatmap_df.values,
#     origin="lower",
#     cmap=P3["cmap"],
#     aspect="auto",
#     extent=[
#         heatmap_df.columns.min(), heatmap_df.columns.max(),
#         heatmap_df.index.min(), heatmap_df.index.max()
#     ],
#     vmin=P3["vmin"], vmax=P3["vmax"],
# )

# if P3["show_colorbar"]:
#     cb3 = fig.colorbar(im, ax=ax)
#     cb3.set_label(P3["cbar_label"], fontsize=P3["label_size"])
#     cb3.ax.tick_params(labelsize=P3["tick_size"])

# if P3["xlabel"]: ax.set_xlabel(P3["xlabel"], fontsize=P3["label_size"])
# if P3["ylabel"]: ax.set_ylabel(P3["ylabel"], fontsize=P3["label_size"])
# ax.tick_params(axis="both", labelsize=P3["tick_size"])
# if P3["show_grid"]:
#     ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

# plt.tight_layout()
# plt.savefig(P3["save_path"], dpi=P3["dpi"], bbox_inches="tight")
# print("Saved:", P3["save_path"])
