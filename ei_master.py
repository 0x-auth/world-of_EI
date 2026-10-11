#!/usr/bin/env python3
"""
ei_master.py — every experiment from the 11–12 Oct 2026 session, in one place.

Self-contained (numpy; scipy for p-values; mpmath only for `outside`).
Each experiment is a subcommand. Expected results are in EI_MASTER.md.

    python3 ei_master.py list
    python3 ei_master.py all            # everything except the long ones (~3 min)

  geometry of t and -1/t
    identities   dt=t+1/t, dx=t-1/t: interval, mirrors, rotation          (instant)
    bracket      too-high / too-low (Babylonian sqrt), in t-variables      (instant)
  the organism (200 seeds, 40k steps, dim 6; baseline = self-referential-seed FINDINGS)
    clock        self-prediction weighted t on E1, -1/t on E2              (~70 s)
    paired       predicts its state AND its inverse partner -> shunya       (~40 s)
    memory       how much older states add to predicting the next move      (~20 s)
  reading with no alphabet (noisy 8x8 glyphs, never labelled, 20 worlds)
    squeeze      unlimited room vs finite room; contrast                    (~60 s)
    errors       similarity, then follow the recurring error                (~10 s)
    recursive    errors of errors: 3 shapes x twin x sub-twin = 12 letters  (~5 s)
  primes
    primes N     walk primes to N: last-digit memory, gap oscillation       (200M: ~4 s)
                 (for 100M+ primes with kill/resume use prime_walk.py)
    outside      rebuild the prime staircase from 200 zeta zeros            (~45 s)
"""
import sys, math, time
import numpy as np

PHI = (1 + 5 ** 0.5) / 2

# ───────────────────────────── geometry ─────────────────────────────

def identities():
    rng = np.random.default_rng(0)
    T = rng.normal(size=1000) + 1j * rng.normal(size=1000)
    dt = lambda t: t + 1 / t; dx = lambda t: t - 1 / t
    ok = lambda a, b: bool(np.allclose(a, b))
    print("checked on 1000 random complex t:")
    print("  dt^2 - dx^2 = 4 (interval; E^2 - p^2 = m^2, m = 2) :", ok(dt(T) ** 2 - dx(T) ** 2, 4))
    print("  dt*dx = t^2 - 1/t^2                                 :", ok(dt(T) * dx(T), T ** 2 - T ** -2))
    print("  speed dx/dt = (t^2-1)/(t^2+1) = tanh(ln t)          :", ok(dx(T) / dt(T), (T ** 2 - 1) / (T ** 2 + 1)))
    print("  t -> i t swaps time and space (dt'=i dx, dx'=i dt)  :", ok(dt(1j * T), 1j * dx(T)) and ok(dx(1j * T), 1j * dt(T)))
    print("  -1/t = e^{i pi}/t (time reversal = half turn)       :", ok(-1 / T, np.exp(1j * np.pi) / T))
    print("\nmirrors at t = 2:")
    for name, u in (("t", 2.0), ("1/t", 0.5), ("-1/t", -0.5), ("-t", -2.0)):
        print(f"  {name:>5}: dt={u + 1/u:+.3f}  dx={u - 1/u:+.3f}")
    print("\nspecial points:")
    for name, t in (("now      t=1", 1), ("golden   t=phi", PHI), ("timeless t=i", 1j)):
        print(f"  {name:<16} dt={complex(dt(t)):.4g}  dx={complex(dx(t)):.4g}")
    print("\nz -> z + t, t alternating t, -1/t (E1, E2): drift per pair = dx")
    for t0 in (1.0, PHI, 2.0, 0.5):
        z, seq = 0.0, []
        for n in range(6):
            z += t0 if n % 2 == 0 else -1 / t0; seq.append(round(z, 3))
        print(f"  t={t0:<6.3f} {seq}   drift/pair={t0 - 1/t0:+.3f}")


def bracket():
    N, g = 2.0, 5.0
    print("the number sqrt(2), pinned by a too-high guess g and its inverse partner N/g")
    print(f"{'step':>4} {'too high':>20} {'too low':>20} {'width':>10}")
    for k in range(7):
        hi, lo = max(g, N / g), min(g, N / g)
        print(f"{k:>4} {hi:>20.15f} {lo:>20.15f} {hi - lo:>10.2e}")
        g = (g + N / g) / 2
    print(f"true {math.sqrt(2):.15f}")
    print("\nin t-variables: next guess = dt/2 = (t+1/t)/2, remaining error = dx = t-1/t")
    t = 5 / math.sqrt(2)
    for _ in range(5):
        print(f"  t={t:.10f}  dt/2={(t + 1/t)/2:.10f}  dx={t - 1/t:+.2e}"); t = (t + 1 / t) / 2

# ───────────────────────────── organism ─────────────────────────────
# Same setup as self-referential-seed/duo_order.py and alive.py.
DIM, SEEDS, STEPS, MU, ALIVE = 6, 200, 40000, 0.01, 1e-4
RANK_EXPLORING, VAR_FLOOR = 2.5, 1e-12

def mv(M, v): return np.einsum("sij,sj->si", M, v)

def make(seed, rho0=1.05):
    rng = np.random.default_rng(seed)
    R = rng.normal(0, 1, (SEEDS, DIM, DIM))
    for i in range(SEEDS): R[i] *= rho0 / max(abs(np.linalg.eigvals(R[i])))
    WA = rng.normal(0, 0.3, (SEEDS, DIM, DIM)); _ = rng.normal(0, 0.3, (SEEDS, DIM, DIM))
    x = rng.normal(0, 1, (SEEDS, DIM))
    return R, WA, x

def fates(order, er, tr):
    frozen = order < ALIVE
    exploring = (~frozen) & (tr > VAR_FLOOR) & (er > RANK_EXPLORING)
    return frozen, (~frozen) & ~exploring, exploring

def live(R, W, x, c, sched=None, paired=False, tail=8000, keep_traj=False, seed=1):
    """One population of organisms. sched(k) scales the self-prediction (clock);
    paired adds a predictor of the inverse partner x/|x|^2 and acts on the midpoint."""
    S, n = x.shape; W = W.copy(); x = x.copy()
    inv = lambda v: v / (np.sum(v * v, axis=1, keepdims=True) + 1e-12)
    V = np.random.default_rng(seed).normal(0, 0.3, (S, n, n)) if paired else None
    eta = np.full((S, n), 0.05); etv = np.full((S, n), 0.05); pv = np.ones(S)
    moved = np.zeros(S); scale = np.zeros(S); cnt = 0; half = STEPS // 2
    M2 = np.zeros((S, n, n)); mean = np.zeros((S, n)); tc = 0
    traj = np.zeros((tail + 1, S, n)) if keep_traj else None; Wstart = None
    for k in range(STEPS):
        p = mv(W, x)
        if paired:
            y = inv(x); q = mv(V, y); pred = (p + inv(q)) / 2
        else:
            pred = p
        s = sched(k) if sched else 1.0
        xn = np.tanh(mv(R, x) + c * s * pred)
        d = xn - p
        W = W + eta[:, :, None] * d[:, :, None] * x[:, None, :]
        m = np.linalg.norm(xn - pred, axis=1)
        if paired:
            dy = inv(xn) - q; sc = 1 / (1 + np.sum(y * y, axis=1))[:, None, None]
            V = V + sc * etv[:, :, None] * dy[:, :, None] * y[:, None, :]
            m = m + np.minimum(np.linalg.norm(dy, axis=1), 1e6)
            etv = np.clip(etv * (1 + MU * np.sign(pv - m)[:, None]), 1e-6, 0.5)
        eta = np.clip(eta * (1 + MU * np.sign(pv - m)[:, None]), 1e-6, 0.5); pv = m
        if not np.all(np.isfinite(W)): W = np.nan_to_num(W)
        if k >= half:
            moved += np.linalg.norm(xn - x, axis=1); scale = np.maximum(scale, np.linalg.norm(xn, axis=1)); cnt += 1
        if k >= STEPS - tail:
            M2 += x[:, :, None] * x[:, None, :]; mean += x; tc += 1
            if keep_traj: traj[k - (STEPS - tail) + 1] = xn
        if keep_traj and k == STEPS - tail - 1: Wstart = W.copy(); traj[0] = xn
        x = xn
    order = np.where(scale > VAR_FLOOR, moved / (cnt * np.maximum(scale, VAR_FLOOR)), 0.0)
    mean /= tc; cov = M2 / tc - mean[:, :, None] * mean[:, None, :]
    er = np.zeros(S); tr = np.zeros(S)
    for i, cv in enumerate(cov):
        lam = np.abs(np.linalg.eigvalsh((cv + cv.T) / 2)); s1, s2 = lam.sum(), (lam ** 2).sum()
        er[i] = s1 ** 2 / s2 if s2 > 1e-300 else 1.0; tr[i] = np.trace(cv)
    out = dict(order=order, er=er, tr=tr, xnorm=np.linalg.norm(x, axis=1))
    if keep_traj:
        out["traj"] = traj
        out["drift"] = np.linalg.norm(W - Wstart, axis=(1, 2)) / np.maximum(np.linalg.norm(W, axis=(1, 2)), 1e-12)
    return out

def row(name, o, extra=""):
    f, cy, ex = fates(o["order"], o["er"], o["tr"])
    print(f"  {name:<34}{f.mean()*100:>8.1f}%{cy.mean()*100:>9.1f}%{ex.mean()*100:>9.1f}%{extra}", flush=True)

def clock():
    c = 0.6; R, W, x = make(0)
    rs = np.random.default_rng(99).integers(0, 2, STEPS)
    alt = lambda t: (lambda k: t if k % 2 == 0 else -1 / t)
    rnd = lambda t: (lambda k: t if rs[k] == 0 else -1 / t)
    print(f"c={c}: self-prediction enters the law as c*s, s alternating t (E1) and -1/t (E2)")
    print(f"  {'clock':<34}{'settled':>9}{'cycling':>10}{'explore':>10}")
    for name, sch in (("original s=1", None), ("constant s=0.5", lambda k: 0.5),
                      ("alternating t=1  (+1,-1)", alt(1.0)), ("alternating t=phi", alt(PHI)),
                      ("alternating t=2", alt(2.0)), ("alternating t=1/phi", alt(1 / PHI)),
                      ("alternating t=0.5", alt(0.5)), ("random order t=phi", rnd(PHI))):
        row(name, live(R, W, x, c, sched=sch))

def paired():
    for c in (0.6, 1.0):
        R, W, x = make(0)
        print(f"\nc={c}   {'':<26}{'settled':>9}{'cycling':>10}{'explore':>10}{'collapsed':>12}")
        for p in (False, True):
            o = live(R, W, x, c, paired=p)
            row("paired" if p else "original", o, f"{(o['xnorm'] < 1e-6).mean()*100:>11.1f}%")

def memory():
    def gains(traj, K=4):
        T, S, n = traj.shape; out = np.full(S, np.nan)
        Y = np.arctanh(np.clip(traj[K + 1:], -0.999999, 0.999999))
        for s in range(S):
            feats = [traj[K - j:T - 1 - j, s] for j in range(K + 1)]; y = Y[:, s]; h = len(y) // 2
            errs = []
            for k in (0, K):
                F = np.hstack(feats[:k + 1]); B = np.linalg.solve(F[:h].T @ F[:h] + 1e-8 * np.eye(F.shape[1]), F[:h].T @ y[:h])
                errs.append(np.mean((F[h:] @ B - y[h:]) ** 2))
            out[s] = 1 - errs[1] / errs[0] if errs[0] > 1e-20 else np.nan
        return out
    try: from scipy.stats import mannwhitneyu
    except ImportError: mannwhitneyu = None
    for c in (0.6, 1.0):
        R, W, x = make(0); o = live(R, W, x, c, keep_traj=True)
        f, cy, ex = fates(o["order"], o["er"], o["tr"]); g = gains(o["traj"])
        exact = np.isnan(g); g0 = np.where(exact, 0.0, g)
        print(f"\nc={c}   (settled runs have no movement to predict; excluded)")
        print(f"  {'fate':<11}{'n':>4}{'present alone enough':>22}{'past gain':>11}{'self-model change':>19}")
        for name, m in (("cycling", cy), ("exploring", ex)):
            print(f"  {name:<11}{m.sum():>4}{exact[m].mean()*100:>21.1f}%{np.median(g0[m]):>11.3f}{np.median(o['drift'][m]):>19.2e}")
        if mannwhitneyu:
            print(f"  exploring vs cycling: past gain p={mannwhitneyu(g0[ex], g0[cy]).pvalue:.1e}, "
                  f"self-model change p={mannwhitneyu(o['drift'][ex], o['drift'][cy]).pvalue:.1e}")

# ─────────────────────── reading with no alphabet ───────────────────────

def glyphs(rng, tiers=2):
    """tiers=2: 3 shapes x twin (6 px) = 6 letters. tiers=3: x sub-twin (2 px) = 12."""
    out = []
    for b in (rng.random((3, 64)) < 0.5).astype(float):
        t = b.copy(); i = rng.choice(64, 6, replace=False); t[i] = 1 - t[i]
        for x in (b, t):
            if tiers == 2: out.append(x)
            else:
                s = x.copy(); j = rng.choice(64, 2, replace=False); s[j] = 1 - s[j]; out += [x, s]
    return np.array(out)

def draw(L, rng, n, noise=0.15):
    k = rng.integers(len(L), size=n); X = L[k].copy(); f = rng.random(X.shape) < noise; X[f] = 1 - X[f]; return X, k

def reader(X, room=None, contrast=False, theta=3.6, lr=0.05, lrc=0.02):
    """Similarity reader: nearest held shape moves toward the sample; surprise becomes
    a new held shape; with finite room, the two most alike are merged first."""
    P = [X[0].copy()]; W = [1]
    for x in X[1:]:
        d = np.linalg.norm(np.array(P) - x, axis=1); o = np.argsort(d); k = o[0]
        if d[k] > theta:
            if room and len(P) >= room:
                A = np.array(P); D = np.linalg.norm(A[:, None] - A[None], axis=2); np.fill_diagonal(D, 1e9)
                i, j = np.unravel_index(D.argmin(), D.shape)
                P[i] = (W[i] * P[i] + W[j] * P[j]) / (W[i] + W[j]); W[i] += W[j]; del P[j]; del W[j]
            P.append(x.copy()); W.append(1)
        else:
            P[k] += lr * (x - P[k]); W[k] += 1
            if contrast and len(P) > 1 and d[o[1]] < theta * 1.2: P[o[1]] -= lrc * (x - P[o[1]])
    return np.array(P)

def score(lab, Y, K):
    m = {}
    for l, y in zip(lab, Y): m.setdefault(l, []).append(y)
    m = {l: np.bincount(v, minlength=K).argmax() for l, v in m.items()}
    return float(np.mean([m[l] == y for l, y in zip(lab, Y)]))

def nearest(P, X): return np.linalg.norm(X[:, None] - P[None], axis=2).argmin(1)

def squeeze():
    print("6 hidden letters (3 shapes + near-twins), 15% noise, 4000 sights, no labels, 20 worlds (median)")
    print(f"  {'room':<12}{'contrast':<10}{'held':>6}{'3 shapes':>10}{'6 letters':>11}")
    for room in (None, 8, 6):
        for con in (False, True):
            R = []
            for seed in range(20):
                rng = np.random.default_rng(seed); L = glyphs(rng)
                P = reader(draw(L, rng, 4000)[0], room, con); Xt, Yt = draw(L, rng, 600); w = nearest(P, Xt)
                R.append((len(P), score(w, Yt // 2, 3), score(w, Yt, 6)))
            R = np.median(np.array(R), axis=0)
            print(f"  {str(room or 'unlimited'):<12}{('yes' if con else 'no'):<10}{R[0]:>6.0f}{R[1]*100:>9.1f}%{R[2]*100:>10.1f}%", flush=True)

def split_on_errors(X, depth):
    """Follow the recurring error: find pixels whose error keeps varying more than noise
    explains, split there, then do the same to each half's error."""
    if depth == 0 or len(X) < 80: return lambda x: ()
    v = (X - X.mean(0)).var(0); s = np.sort(v)[::-1]; cut = (s[:-1] - s[1:])[:20].argmax() + 1
    if s[cut - 1] < 1.4 * np.median(v): return lambda x: ()
    pix = np.argsort(v)[::-1][:cut]; Z = X[:, pix]
    c = np.array([Z[Z.sum(1).argmin()], Z[Z.sum(1).argmax()]]).astype(float)
    for _ in range(20):
        a = np.linalg.norm(Z[:, None] - c[None], axis=2).argmin(1)
        for j in (0, 1):
            if (a == j).any(): c[j] = Z[a == j].mean(0)
    kids = [split_on_errors(X[a == j], depth - 1) for j in (0, 1)]
    def f(x):
        j = int(np.linalg.norm(c - x[pix], axis=1).argmin()); return (j,) + kids[j](x)
    f.pix = pix
    return f

def errors_exp(tiers, depths, title):
    K = 6 if tiers == 2 else 12
    print(title)
    res = {d: [] for d in depths}; hits = []
    for seed in range(20):
        rng = np.random.default_rng(seed); L = glyphs(rng, tiers)
        P = reader(draw(L, rng, 4000)[0], room=6)
        Xw, _ = draw(L, rng, 12000); g = nearest(P, Xw)
        Xt, Yt = draw(L, rng, 1200); gt = nearest(P, Xt)
        for d in depths:
            fs = {k: split_on_errors(Xw[g == k], d) for k in set(g)}
            lab = [(k,) + fs.get(k, lambda x: ())(x) for k, x in zip(gt, Xt)]
            res[d].append((score(lab, Yt // (K // 3), 3), score(lab, Yt // (K // 6), 6), score(lab, Yt, K), len(set(lab))))
            if tiers == 2 and d == 1:                                   # did it look where the twins differ?
                for k, f in fs.items():
                    if not hasattr(f, "pix"): continue
                    b = np.bincount(Yt[gt == k], minlength=6).argmax() // 2
                    true = set(np.nonzero(L[2 * b] != L[2 * b + 1])[0]); hits.append(len(true & set(f.pix)) / len(true))
    names = {0: "similarity only (room 6)", 1: "+ follow errors", 2: "+ errors of errors"}
    for d in depths:
        R = np.median(np.array(res[d]), axis=0)
        last = f"{R[2]*100:>9.1f}%" if K == 12 else ""
        print(f"  {names[d]:<26} shapes {R[0]*100:5.1f}%  6-tier {R[1]*100:5.1f}%{('  all 12' + last) if K == 12 else ''}   groups {R[3]:.0f}")
    if hits: print(f"  error-pixels that are the true twin differences: {np.median(hits)*100:.0f}% (median)")

def errors():
    """Streaming version (as in follow_errors.py): the reader keeps living; for each held
    shape it watches ONLY the error x - shape, keeps a running variance per pixel, splits
    where the error keeps recurring, then learns the split online."""
    print("6 hidden letters, 15% noise, no labels, 20 worlds (median)")
    res, hits = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed); L = glyphs(rng)
        def one(): return draw(L, rng, 1)[0][0]
        P = reader(np.array([one() for _ in range(4000)]), room=6)
        K = len(P); n = np.zeros(K); m = np.zeros((K, 64)); s2 = np.zeros((K, 64))
        for _ in range(4000):
            x = one(); k = np.linalg.norm(P - x, axis=1).argmin(); e = x - P[k]
            n[k] += 1; dl = e - m[k]; m[k] += dl / n[k]; s2[k] += dl * (e - m[k])
        var = s2 / np.maximum(n, 1)[:, None]; split = {}
        for k in range(K):
            if n[k] < 50: continue
            v = np.sort(var[k])[::-1]; cut = (v[:-1] - v[1:])[:20].argmax() + 1
            if v[cut - 1] < 1.4 * np.median(v): continue
            pix = np.argsort(var[k])[::-1][:cut]; c = np.array([np.zeros(cut), np.ones(cut)])
            for _ in range(3000):
                x = one()
                if np.linalg.norm(P - x, axis=1).argmin() != k: continue
                z = x[pix]; j = np.linalg.norm(c - z, axis=1).argmin(); c[j] += 0.05 * (z - c[j])
            split[k] = (pix, c)
        Xt, Yt = draw(L, rng, 600); gt = nearest(P, Xt)
        lab = [(k, int(np.linalg.norm(split[k][1] - x[split[k][0]], axis=1).argmin())) if k in split else (k,)
               for k, x in zip(gt, Xt)]
        res.append((score(list(gt), Yt, 6), score(lab, Yt, 6), K + 2 * len(split)))
        for k, (pix, _) in split.items():
            if (gt == k).any():
                b = np.bincount(Yt[gt == k], minlength=6).argmax() // 2
                true = set(np.nonzero(L[2 * b] != L[2 * b + 1])[0]); hits.append(len(true & set(pix)) / len(true))
    R = np.median(np.array(res), axis=0)
    print(f"  similarity only (room 6)   6 letters {R[0]*100:5.1f}%")
    print(f"  + follow errors            6 letters {R[1]*100:5.1f}%   held things {R[2]:.0f}")
    print(f"  error-pixels that are the true twin differences: {np.median(hits)*100:.0f}% (median)")
def recursive(): errors_exp(3, (0, 1, 2), "12 hidden letters: 3 shapes x twin (6 px) x sub-twin (2 px), 20 worlds (median)")

# ───────────────────────────── primes ─────────────────────────────

def primes(N=200_000_000):
    N = int(N); s = np.ones(N // 2, dtype=bool); s[0] = False
    for i in range(3, int(N ** 0.5) + 1, 2):
        if s[i // 2]: s[i * i // 2::i] = False
    p = np.concatenate(([2], 2 * np.nonzero(s)[0] + 1)); p = p[p > 5]
    print(f"{len(p):,} primes up to {N:,}\n\nlast digit -> next prime's last digit (row %; no memory = 25)")
    lut = np.full(10, -1); lut[[1, 3, 7, 9]] = [0, 1, 2, 3]; d = lut[p % 10]
    T = np.bincount(d[:-1] * 4 + d[1:], minlength=16).reshape(4, 4).astype(float)
    print("        " + "".join(f"{x:>8}" for x in (1, 3, 7, 9)))
    for i, x in enumerate((1, 3, 7, 9)): print(f"   {x}   " + "".join(f"{v:>7.1f}%" for v in T[i] / T[i].sum() * 100))
    g = np.diff(p).astype(float); z = (g - g.mean()) / g.std()
    print(f"\ncorrelation of consecutive gaps: {np.mean(z[:-1] * z[1:]):+.4f}   (noise ~ {1/math.sqrt(len(z)):.4f})")

def outside(nz=200):
    import mpmath as mp
    from sympy import factorint
    mp.mp.dps = 20; zeros = [mp.zetazero(k) for k in range(1, nz + 1)]
    def psi(x, n):
        s = x - mp.log(2 * mp.pi) - 0.5 * mp.log(1 - x ** -2)
        for r in zeros[:n]: s -= 2 * mp.re(mp.power(x, r) / r)
        return float(s)
    xs = np.arange(2.5, 40, 0.05)
    print("where the waves of n zeta zeros add up to a step (below 40):")
    for n in (10, 50, nz):
        ys = np.array([psi(x, n) for x in xs]); found = []
        for q in range(3, 40):
            i0, i1 = np.searchsorted(xs, q - 0.25), np.searchsorted(xs, q + 0.25)
            if i1 < len(ys) and ys[i1] - ys[i0] > 0.4: found.append(q)
        print(f"  {n:>3} zeros: {found}")
    print(f"  true prime powers: {[n for n in range(3, 40) if len(factorint(n)) == 1]}")
    print(f"all {nz} zeros on Re = 1/2: {all(abs(z.real - 0.5) < 1e-15 for z in zeros)}")

# ───────────────────────────── main ─────────────────────────────

EXPS = dict(identities=identities, bracket=bracket, clock=clock, paired=paired, memory=memory,
            squeeze=squeeze, errors=errors, recursive=recursive, primes=primes, outside=outside)
QUICK = ["identities", "bracket", "errors", "recursive", "primes", "memory", "paired", "clock", "squeeze"]

if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] in ("list", "-h", "--help"):
        print(__doc__); sys.exit()
    names = QUICK if a[0] == "all" else [a[0]]
    for nm in names:
        if nm not in EXPS: sys.exit(f"unknown experiment: {nm} (try: list)")
        print("\n" + "═" * 76 + f"\n  {nm}\n" + "═" * 76)
        t0 = time.time()
        EXPS[nm](*a[1:]) if (nm == names[0] and len(a) > 1 and a[0] != "all") else EXPS[nm]()
        print(f"  ({time.time() - t0:.1f} s)")
