"""Real data snippets for the interactive explainer (small JSON + hand renders)."""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
warnings.filterwarnings("ignore")
from semg2pose.data import FS, JOINT_NAMES, list_sessions, load_session, find_session
from semg2pose.signals import bandpass, rms_envelope, log_envelope

OUT = os.path.join(os.path.dirname(__file__), "..", "explainer")
os.makedirs(OUT, exist_ok=True)
sessions = list_sessions()
D = {}

# 1. finger topographies (120-250 Hz, common effort removed), as in spectral_spatial_fingerprints §3
FING = {"pulgar": ["THUMB_CMC_FE", "THUMB_MCP_FE", "THUMB_IP_FE"]}
for nm, es in [("INDEX", "índice"), ("MIDDLE", "corazón"), ("RING", "anular"), ("PINKY", "meñique")]:
    FING[es] = [f"{nm}_MCP_FE", f"{nm}_PIP_FE", f"{nm}_DIP_FE"]
FI = {k: [JOINT_NAMES.index(n) for n in v] for k, v in FING.items()}
labs, feats = [], []
for _, row in sessions.iterrows():
    emg, angles, ok, _ = load_session(row.path)
    idx = np.arange(0, len(emg), 100)
    F = np.log(rms_envelope(bandpass(emg, 120, 250), window_s=0.1) + 1)[idx]
    vel = signal.savgol_filter(angles, 401, 2, deriv=1, delta=1 / FS, axis=0)[idx]
    good = ~np.convolve(~ok[idx], np.ones(5), mode="same").astype(bool)
    fv = np.stack([vel[:, FI[k]].mean(1) for k in FING], 1)
    sp = np.abs(fv); o = np.sort(sp, 1); top = sp.argmax(1)
    mov = (o[:, -1] > 1.0) & (o[:, -1] > 1.5 * o[:, -2])
    lab = np.full(len(fv), "", dtype=object)
    names = list(FING)
    for i in np.where(mov)[0]:
        lab[i] = f"{names[top[i]]}|{'flex' if fv[i, top[i]] > 0 else 'ext'}"
    lab[o[:, -1] < 0.15] = "rest"
    Fl = np.vstack([F[:1], F[:-1]])
    keep = good & (lab != "")
    labs.append(lab[keep]); feats.append(Fl[keep])
labs, feats = np.concatenate(labs), np.concatenate(feats)
base = feats[labs == "rest"].mean(0)
topo = {}
for e in np.unique(labs):
    if e == "rest" or (labs == e).sum() < 100:
        continue
    v = feats[labs == e].mean(0) - base
    topo[e] = {"raw": np.round(v, 3).tolist(), "n": int((labs == e).sum())}
D["topo"] = topo

# 2. lag explorer: mean z-scored envelope vs hand speed, HandClawGraspFlicks right, 50 Hz, 40 s
emg, ang, ok, _ = load_session(find_session("HandClawGraspFlicks", "right", sessions))
env = log_envelope(emg, band=(20, 450), window_s=0.05)[::40]
env = ((env - env.mean(0)) / env.std(0)).mean(1)
vel = signal.savgol_filter(ang, 301, 3, deriv=1, delta=1 / FS, axis=0)[::40]
speed = np.linalg.norm(vel, axis=1)
okd = ~np.convolve(~ok, np.ones(601), mode="same").astype(bool)[::40]
s0 = 5 * 50
sl = slice(s0, s0 + 40 * 50)
spd = np.where(okd[sl], speed[sl], np.nan)
D["lag"] = {"fs": 50, "env": np.round(env[sl], 3).tolist(),
            "speed": [None if np.isnan(x) else round(float(x), 3) for x in spd]}

# 3. PCA toy: two joint pairs (degrees), subsampled
X = np.vstack([load_session(p)[1][::40][load_session(p)[2][::40]] for p in sessions.path])
rng = np.random.default_rng(0)
sub = rng.choice(len(X), 700, replace=False)
pair = lambda a, b: np.round(np.degrees(X[sub][:, [JOINT_NAMES.index(a), JOINT_NAMES.index(b)]]), 1).tolist()
D["pca_pairs"] = {"MIDDLE_MCP_FE|RING_MCP_FE": pair("MIDDLE_MCP_FE", "RING_MCP_FE"),
                  "INDEX_MCP_FE|PINKY_MCP_FE": pair("INDEX_MCP_FE", "PINKY_MCP_FE"),
                  "THUMB_IP_FE|INDEX_PIP_FE": pair("THUMB_IP_FE", "INDEX_PIP_FE")}

# 4. PCA of all 20 angles: loadings of PC1..PC4 and hand renders along PC1
mu = X.mean(0)
_, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
if Vt[0, JOINT_NAMES.index("MIDDLE_PIP_FE")] < 0:
    Vt[0] = -Vt[0]                       # + = closing
sd = s / np.sqrt(len(X) - 1)
ev = s ** 2 / (s ** 2).sum()
D["pc_loadings"] = {f"PC{k + 1}": np.round(Vt[k], 3).tolist() for k in range(4)}
D["pc_var"] = np.round(100 * ev, 2).tolist()
D["joint_names"] = JOINT_NAMES
from emg2pose.visualization import skin_mesh_from_angles
BG = "#FBFAF6"
for k, tag in [(0, "pc1"), (3, "pc4"), (2, "pc3")]:
    for i, m in enumerate(np.linspace(-2, 2, 9 if k == 0 else 5)):
        fig = plt.figure(figsize=(3, 2.6), facecolor=BG)
        ax = fig.add_subplot(projection="3d"); ax.set_facecolor(BG)
        v, tri = skin_mesh_from_angles(mu + m * sd[k] * Vt[k])
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=np.asarray(tri), color="#E7CDB5", edgecolor="none", shade=True)
        ax.view_init(20, -60); ax.set_box_aspect(np.ptp(v, axis=0)); ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(f"{OUT}/hand_{tag}_{i}.png", dpi=110, facecolor=BG)
        plt.close(fig)

# 5. Santello curves
fr_rows = []
for sid, row in sessions.iterrows():
    _, angles, ok, _ = load_session(row.path)
    a, okd = angles[::40], ok[::40]
    v = signal.savgol_filter(a, 11, 2, deriv=1, delta=1 / 50, axis=0)
    good = ~np.convolve(~okd, np.ones(7), mode="same").astype(bool)
    fr_rows.append(pd.DataFrame(a[good], columns=JOINT_NAMES).assign(session=sid, t=np.arange(len(a))[good] / 50,
                                                                    speed=np.linalg.norm(v, axis=1)[good]))
fr = pd.concat(fr_rows, ignore_index=True)
Fd = {n: fr[n].values for n in JOINT_NAMES}
S15 = np.column_stack([Fd[f"{f}_MCP_FE"] for f in ["INDEX", "MIDDLE", "RING", "PINKY"]]
                      + [Fd[f"{f}_PIP_FE"] for f in ["INDEX", "MIDDLE", "RING", "PINKY"]]
                      + [Fd["INDEX_MCP_AA"] - Fd["MIDDLE_MCP_AA"], Fd["MIDDLE_MCP_AA"] - Fd["RING_MCP_AA"],
                         Fd["RING_MCP_AA"] - Fd["PINKY_MCP_AA"]]
                      + [Fd["THUMB_CMC_FE"], Fd["THUMB_CMC_AA"], Fd["THUMB_MCP_FE"], Fd["THUMB_IP_FE"]])
still = fr.speed.values < 0.5
rid = np.cumsum(np.r_[True, (np.diff(still.astype(int)) != 0) | (np.diff(fr.session.values) != 0)
                      | (np.diff(fr.t.values) > 1.5 / 50)])
ln = pd.Series(rid[still]).value_counts()
m = np.isin(rid, ln.index[ln >= 15]) & still


def cum(Z):
    e = np.linalg.svd(np.degrees(Z) - np.degrees(Z).mean(0), compute_uv=False) ** 2
    return np.round(100 * np.cumsum(e) / e.sum(), 1).tolist()


D["santello"] = {"all20": cum(fr[JOINT_NAMES].values), "all15": cum(S15),
                 "held20": cum(pd.DataFrame(fr[JOINT_NAMES].values[m]).assign(r=rid[m]).groupby("r").mean().values),
                 "held15": cum(pd.DataFrame(S15[m]).assign(r=rid[m]).groupby("r").mean().values)}

json.dump(D, open(f"{OUT}/data.json", "w"), separators=(",", ":"), ensure_ascii=False)
print({k: (len(v) if hasattr(v, "__len__") else v) for k, v in D.items()}, os.path.getsize(f"{OUT}/data.json"))
print(sorted(D["topo"]))
