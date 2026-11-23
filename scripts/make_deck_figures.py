"""Clean, consistent figures for the deck. Numbers either recomputed (same code as the notebooks) or copied
from the executed notebooks' printed outputs (marked SOURCE)."""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "deck")
os.makedirs(OUT, exist_ok=True)

BG, INK, MUTED, GRID = "#FAFAF7", "#1B2433", "#5B6474", "#E2E3DE"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#A3A8AE"
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # ordinal blue ramp
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 17,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.facecolor": BG, "figure.facecolor": BG, "savefig.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlesize": 18, "axes.titleweight": "normal",
    "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("saved", name)


from semg2pose.data import FS, JOINT_NAMES, list_sessions, load_session, find_session
from semg2pose.signals import bandpass, rms_envelope, log_envelope
sessions = list_sessions()
which = sys.argv[1:] or ["all"]
want = lambda n: "all" in which or n in which

# ---------------------------------------------------------------- pipeline strip
if want("strip"):
    emg, ang, ok, _ = load_session(find_session("HandClawGraspFlicks", "right", sessions))
    t = np.arange(len(emg)) / FS
    sel = (t >= 12) & (t < 16)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    sp = 6 * np.median(emg.std(0))
    for c in range(16):
        a1.plot(t[sel], emg[sel, c] + (15 - c) * sp, lw=0.35, color=INK)
    a1.set_yticks([(15 - c) * sp for c in (0, 7, 15)], ["ch 0", "ch 7", "ch 15"])
    a1.grid(False); a1.spines["left"].set_visible(False); a1.tick_params(left=False)
    a1.set_title("input: 16 EMG channels, 2 kHz", loc="left", fontsize=17, color=MUTED)
    for j, col in zip(["INDEX_PIP_FE", "MIDDLE_MCP_FE", "THUMB_IP_FE"], [BLUE, ORANGE, "#1baf7a"]):
        y = np.degrees(ang[:, JOINT_NAMES.index(j)])
        y[np.convolve(~ok, np.ones(201), mode="same") > 0] = np.nan   # hide IK failures and their edges
        a2.plot(t[sel], y[sel], color=col, lw=1.8, label=j.replace("_", " ").lower())
    a2.set_ylabel("angle (°)"); a2.set_xlabel("time (s)")
    a2.set_title("target: 20 joint angles", loc="left", fontsize=17, color=MUTED)
    a2.legend(fontsize=12, ncol=3, loc="upper right", borderaxespad=0.1, handlelength=1.2, columnspacing=1.0)
    a2.set_ylim(-30, 175)
    save(fig, "strip.png")

# ---------------------------------------------------------------- band locality (SOURCE: motor_units_zoom §8.4 table)
if want("locality"):
    dist_mm = np.arange(1, 9) * 9.9
    tab = {"20–60 Hz": [0.65, 0.46, 0.35, 0.28, 0.25, 0.24, 0.23, 0.23],
           "60–120 Hz": [0.72, 0.46, 0.32, 0.25, 0.22, 0.20, 0.20, 0.20],
           "120–250 Hz": [0.61, 0.31, 0.18, 0.14, 0.11, 0.11, 0.10, 0.10],
           "250–500 Hz": [0.59, 0.34, 0.21, 0.14, 0.11, 0.09, 0.08, 0.08],
           "500–900 Hz": [0.52, 0.32, 0.20, 0.14, 0.10, 0.08, 0.07, 0.06]}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for (name, v), col in zip(tab.items(), RAMP[1:]):
        ax.plot(dist_mm, v, "o-", color=col, ms=5, label=name)
    ax.set_xlabel("distance between electrodes (mm)"); ax.set_ylabel("mean |correlation|")
    ax.set_ylim(0, 0.8); ax.legend(fontsize=13, title="band", title_fontsize=13)
    ax.set_title("high bands are more local", loc="left", color=MUTED)
    save(fig, "locality.png")

# ---------------------------------------------------------------- learned first-layer filters
if want("filters"):
    from semg2pose.model import load_pretrained
    W = load_pretrained("tracking_vemg2pose").model.network.layers[0].conv[0].weight.detach().numpy()
    tk = np.array([np.linalg.svd(w, full_matrices=False)[2][0] for w in W])
    f = np.fft.rfftfreq(256, 1 / FS)
    H = np.abs(np.fft.rfft(tk, 256, axis=1)) ** 2
    Hn = H / H.max(1, keepdims=True)
    cen = (H * f).sum(1) / H.sum(1)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.imshow(Hn[np.argsort(cen)], aspect="auto", extent=[0, FS / 2, 256, 0], cmap="Blues", interpolation="nearest")
    ax.set_xlim(0, 700); ax.grid(False)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("filter (sorted)")
    ax.set_title(f"256 learned filters: band-pass, median {np.median(cen):.0f} Hz", loc="left", color=MUTED)
    save(fig, "filters.png")

# ---------------------------------------------------------------- finger-triggered maps (same as fingerprints §3)
if want("fingers"):
    BANDS = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 900)]
    FING = {"thumb": ["THUMB_CMC_FE", "THUMB_MCP_FE", "THUMB_IP_FE"]}
    for nm in ["INDEX", "MIDDLE", "RING", "PINKY"]:
        FING[nm.lower()] = [f"{nm}_MCP_FE", f"{nm}_PIP_FE", f"{nm}_DIP_FE"]
    FI = {k: [JOINT_NAMES.index(n) for n in v] for k, v in FING.items()}
    labs, feats = [], []
    for _, row in sessions.iterrows():
        emg, angles, ok, _ = load_session(row.path)
        idx = np.arange(0, len(emg), 100)
        F = np.stack([np.log(rms_envelope(bandpass(emg, lo, hi), window_s=0.1) + 1)[idx] for lo, hi in BANDS], 1)
        vel = signal.savgol_filter(angles, 401, 2, deriv=1, delta=1 / FS, axis=0)[idx]
        good = ~np.convolve(~ok[idx], np.ones(5), mode="same").astype(bool)
        fv = np.stack([vel[:, FI[k]].mean(1) for k in FING], 1)
        sp = np.abs(fv); o = np.sort(sp, 1); top = sp.argmax(1)
        mov = (o[:, -1] > 1.0) & (o[:, -1] > 1.5 * o[:, -2])
        lab = np.full(len(fv), "", dtype=object)
        names = list(FING)
        for i in np.where(mov)[0]:
            lab[i] = f"{names[top[i]]} {'flex' if fv[i, top[i]] > 0 else 'extend'}"
        lab[o[:, -1] < 0.15] = "rest"
        Fl = np.vstack([np.repeat(F[:1], 1, 0), F[:-1]])
        keep = good & (lab != "")
        labs.append(lab[keep]); feats.append(Fl[keep])
    labs, feats = np.concatenate(labs), np.concatenate(feats)
    base = feats[labs == "rest"].mean(0)
    show = ["index flex", "index extend", "thumb flex", "thumb extend", "pinky flex"]
    ang_c = 2 * np.pi * np.arange(16) / 16
    fig, axs = plt.subplots(1, len(show), figsize=(15, 3.6), subplot_kw={"projection": "polar"})
    for ax, e in zip(axs, show):
        v = (feats[labs == e].mean(0) - base)[2]
        v = v - v.mean()                                 # remove the common "effort" part: where is it MORE active
        v = np.clip(v / np.abs(v).max(), 0, None)
        th = 2 * np.pi * (np.arange(16) + 0.5) / 16
        ax.bar(th, v, width=2 * np.pi / 16 * 0.82, color=BLUE if "flex" in e else ORANGE, edgecolor=BG, lw=1)
        ax.set_theta_zero_location("W"); ax.set_theta_direction(1)   # ulnar left, palm bottom, back top
        ax.set_xticks(th, [str(c) if c in (0, 4, 8, 12) else "" for c in range(16)], fontsize=12)
        ax.set_yticklabels([]); ax.set_ylim(0, 1.05); ax.grid(color=GRID)
        ax.set_title(e, fontsize=17, pad=34)
        ax.text(np.pi * 0.5, 1.42, "palm", ha="center", va="center", fontsize=12, color=MUTED)
        ax.text(np.pi * 1.5, 1.42, "back", ha="center", va="center", fontsize=12, color=MUTED)
    fig.text(0.5, -0.1, "120–250 Hz power above the movement's own average (common effort removed); ring = wrist, ulnar side left",
             ha="center", fontsize=14, color=MUTED)
    save(fig, "fingers.png")

# ---------------------------------------------------------------- encoding R² (SOURCE: emg_kinematics_encoding §3)
if want("encoding"):
    enc = pd.Series({"position": 0.066, "velocity (signed)": 0.054, "speed |v|": 0.085,
                     "acceleration (split by direction)": 0.070, "velocity (split by direction)": 0.140,
                     "position + velocity split": 0.198, "+ acceleration split": 0.219})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    cols = [GREY, GREY, GREY, GREY, BLUE, INK, INK]
    ax.barh(range(len(enc)), enc.values, color=cols, height=0.62)
    ax.set_yticks(range(len(enc)), enc.index); ax.invert_yaxis()
    for i, v in enumerate(enc.values):
        ax.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=14, color=INK)
    ax.set_xlabel("mean R² over 16 channels (blocked CV)"); ax.set_xlim(0, 0.26); ax.grid(axis="y", visible=False)
    ax.set_title("kinematics → EMG envelope", loc="left", color=MUTED)
    save(fig, "encoding.png")

# ---------------------------------------------------------------- TRF: derivative vs integral (same as encoding §4)
if want("trf"):
    STEP, LAGS = 40, [0, 1, 2, 3, 5, 8, 11, 15]
    Xs, Zp, Zv = [], [], []
    recs = []
    for _, row in sessions.iterrows():
        if row.stage == "unconstrained":
            continue
        emg, angles, ok, _ = load_session(row.path)
        env = log_envelope(emg, band=(20, 450), window_s=0.05)[::STEP]
        pos = signal.savgol_filter(angles, 301, 3, deriv=0, axis=0)[::STEP]
        vel = signal.savgol_filter(angles, 301, 3, deriv=1, delta=1 / FS, axis=0)[::STEP]
        good = ~np.convolve(~ok, np.ones(601), mode="same").astype(bool)[::STEP]
        recs.append((env, pos, vel, good))
    P = np.vstack([r[1][r[3]] for r in recs]); mu = P.mean(0)
    w1 = np.linalg.svd(P - mu, full_matrices=False)[2][0]
    if w1[JOINT_NAMES.index("MIDDLE_PIP_FE")] < 0:
        w1 = -w1                                       # orient PC1 so that + = closing the hand
    lag = lambda X: np.hstack([X if l == 0 else np.vstack([np.repeat(X[:1], l, 0), X[:-l]]) for l in LAGS])
    X = np.vstack([lag(r[0])[r[3]] for r in recs])
    yp = np.concatenate([((r[1] - mu) @ w1)[r[3]] for r in recs])
    yv = np.concatenate([(r[2] @ w1)[r[3]] for r in recs])
    A = (X - X.mean(0)) / (X.std(0) + 1e-9)
    G = A.T @ A + 10 * np.eye(A.shape[1])
    Wv = np.linalg.solve(G, A.T @ (yv - yv.mean())).reshape(len(LAGS), 16)
    Wp = np.linalg.solve(G, A.T @ (yp - yp.mean())).reshape(len(LAGS), 16)
    ch = int(np.argmax(np.abs(Wv).sum(0)))
    lag_ms = -np.array(LAGS) * 20
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, Wx, name, col in [(axs[0], Wv, "decoding velocity", BLUE), (axs[1], Wp, "decoding position", ORANGE)]:
        w = Wx[:, ch] / np.abs(Wx[:, ch]).max()
        ax.axhline(0, color=MUTED, lw=1)
        ax.plot(lag_ms, w, "o-", color=col, ms=7)
        ax.set_title(f"{name} (channel {ch})", loc="left", color=MUTED)
        ax.set_xlabel("how long before now (ms)"); ax.set_ylim(-1.15, 1.15)
    axs[0].set_ylabel("decoder weight (normalised)")
    save(fig, "trf.png")
    print("TRF channel", ch)

# ---------------------------------------------------------------- Santello (same as compression_tolerance §2)
if want("santello"):
    rows = []
    for sid, row in sessions.iterrows():
        _, angles, ok, _ = load_session(row.path)
        a, okd = angles[::40], ok[::40]
        vel = signal.savgol_filter(a, 11, 2, deriv=1, delta=1 / 50, axis=0)
        good = ~np.convolve(~okd, np.ones(7), mode="same").astype(bool)
        rows.append(pd.DataFrame(a[good], columns=JOINT_NAMES).assign(
            session=sid, t=np.arange(len(a))[good] / 50, speed=np.linalg.norm(vel, axis=1)[good]))
    fr = pd.concat(rows, ignore_index=True)
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
    keep = ln.index[ln >= 15]
    m = np.isin(rid, keep) & still
    holds = pd.DataFrame(S15[m]).assign(r=rid[m]).groupby("r").mean().values

    def cum(Z):
        e = np.linalg.svd(np.degrees(Z) - np.degrees(Z).mean(0), compute_uv=False) ** 2
        return np.cumsum(e) / e.sum()
    cA, cD = cum(fr[JOINT_NAMES].values), cum(holds)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(range(1, 21), 100 * cA, "o-", color=BLUE, ms=5, label="emg2pose: all frames, 20 angles")
    ax.plot(range(1, 16), 100 * cD, "o-", color=ORANGE, ms=5, label=f"held postures (n={len(holds)}), Santello's 15 angles")
    ax.plot([2], [80], marker="*", color=INK, ms=16, ls="", label="Santello 1998: 2 PCs > 80 %")
    ax.axhline(80, color=MUTED, lw=1)
    ax.set_xlim(0.5, 12.5); ax.set_ylim(40, 101)
    ax.set_xlabel("number of PCs"); ax.set_ylabel("cumulative variance (%)")
    ax.legend(fontsize=13, loc="lower right")
    save(fig, "santello.png")
    print("PC1+2:", 100 * cA[1], 100 * cD[1])

# ---------------------------------------------------------------- eigen-postures (PC1)
if want("hands"):
    from emg2pose.visualization import skin_mesh_from_angles
    X = np.vstack([load_session(p)[1][::40][load_session(p)[2][::40]] for p in sessions.path])
    mu = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    sd = s / np.sqrt(len(X) - 1)
    fig = plt.figure(figsize=(12, 3.4))
    specs = [(0, -2, "PC1 −2σ"), (0, 0, "mean pose"), (0, 2, "PC1 +2σ"), (2, 2, "PC3 +2σ"), (3, 2, "PC4 +2σ")]
    for i, (k, mlt, title) in enumerate(specs):
        ax = fig.add_subplot(1, len(specs), i + 1, projection="3d")
        v, tri = skin_mesh_from_angles(mu + mlt * sd[k] * Vt[k])
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=np.asarray(tri), color="#E9D3BF", edgecolor="none", shade=True)
        ax.view_init(20, -60); ax.set_box_aspect(np.ptp(v, axis=0)); ax.set_axis_off()
        ax.set_facecolor(BG)
        ax.set_title(title, fontsize=17, y=0.92)
    save(fig, "hands.png")

# ---------------------------------------------------------------- per-PC R² of the pretrained model (SOURCE: compression_tolerance §5)
if want("pcr2"):
    trk = [0.86, 0.44, 0.63, 0.52, 0.35, 0.33, 0.39, 0.33, 0.23, 0.25, 0.01, 0.18, 0.11, -0.04, -0.15, -0.33]
    reg = [0.79, -0.02, 0.45, 0.11, 0.01, -0.00, 0.09, 0.12, -0.01, -0.23, -0.82, -0.02, -0.39, -1.00, -0.56, -1.18]
    var = [53.2, 12.4, 7.3, 5.4, 4.2, 3.1, 3.0, 2.3, 2.0, 1.3, 1.2, 0.9, 0.8, 0.7, 0.7, 0.6]
    x = np.arange(len(trk))
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.bar(x - 0.2, np.clip(trk, -0.6, None), 0.38, color=BLUE, label="tracking (knows start pose)")
    ax.bar(x + 0.2, np.clip(reg, -0.6, None), 0.38, color=ORANGE, label="regression (EMG only)")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x, [f"PC{i + 1}\n{v:.0f}%" if v >= 1 else f"PC{i + 1}\n<1%" for i, v in enumerate(var)], fontsize=12)
    ax.set_ylabel("R² on held-out user"); ax.set_ylim(-0.62, 1.0); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=14, loc="upper right")
    save(fig, "pcr2.png")

# ---------------------------------------------------------------- post-hoc bottleneck (SOURCE: compression_tolerance §6)
if want("projection"):
    K = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
    data = {"tracking": ([13.46, 12.86, 12.32, 11.84, 11.51, 11.30, 10.68, 10.16, 9.90, 9.67, 9.56], 9.56),
            "regression": ([13.95, 13.88, 13.48, 13.23, 13.16, 13.08, 12.86, 12.76, 12.75, 13.05, 13.13], 13.13)}
    floor = [12.60, 11.10, 10.18, 9.34, 8.62, 8.04, 6.49, 4.98, 4.07, 1.98, 0.00]
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, (name, (proj, full)) in zip(axs, data.items()):
        col = BLUE if name == "tracking" else ORANGE
        ax.axhline(full, color=INK, lw=1.4, ls=(0, (5, 3)), label="model, all 20 dims")
        ax.plot(K, proj, "o-", color=col, ms=6, label="model output squeezed to k dims")
        ax.plot(K, floor, "o-", color=GREY, ms=5, label="ground truth squeezed (floor)")
        ax.set_title(name, loc="left", color=MUTED); ax.set_xlabel("k (latent size)")
        ax.set_xticks([1, 4, 8, 12, 16, 20])
    axs[0].set_ylabel("angle error, MAE (°)"); axs[0].set_ylim(0, 15)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend([h[0], h[1], h[2]], ["model, all 20 dims", "model output squeezed to k dims (colour)",
               "ground truth squeezed to k dims (floor)"], fontsize=13, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.1))
    save(fig, "projection.png")
