"""Run the pre-registered test once, with the parameters picked on validation (tune_results.json)."""
import os
DH = os.environ.get("DH_DIR", "data")
import pickle, json, collections, numpy as np, multiprocessing as mp
import organism as O

vocab, msgs = pickle.load(open(os.path.join(DH, 'stream.pkl'), 'rb')); V = len(vocab)
tune = json.load(open('tune_results_all.json'))
bestO = min((r for r in tune if r[0] == 'O'), key=lambda r: r[2])[1]
bestS = min((r for r in tune if r[0] == 'S' and r[1]['lam'] > 0), key=lambda r: r[2])[1]
U = dict(alpha=0, lam=0)
JOBS = {'U-frozen': ('S', U, False), 'S-frozen': ('S', bestS, False), 'O-frozen': ('O', bestO, False),
        'U-alive': ('S', U, True), 'S-alive': ('S', bestS, True), 'O-alive': ('O', bestO, True)}


def job(name):
    kind, p, alive = JOBS[name]
    r, model = O.run(msgs, V, kind, p, O.TEST, alive)
    if name == 'O-frozen':
        np.save(os.path.join(DH, 'W_frozen.npy'), model.W); np.save(os.path.join(DH, 'b_frozen.npy'), model.b)
    return name, r


def boot(a, b, days, n=1000, seed=0):
    """Block bootstrap over days of the per-message difference a-b."""
    rng = np.random.default_rng(seed); d = np.array(a) - np.array(b)
    by = collections.defaultdict(list)
    for x, day in zip(d, days): by[day].append(x)
    keys = list(by); sums = np.array([sum(by[k]) for k in keys]); cnt = np.array([len(by[k]) for k in keys])
    est = []
    for _ in range(n):
        i = rng.integers(0, len(keys), len(keys)); est.append(sums[i].sum() / cnt[i].sum())
    return d.mean(), np.percentile(est, [2.5, 97.5])


if __name__ == '__main__':
    print('picked on validation: organism', bestO, '| snapshot', bestS)
    with mp.Pool(2) as pool:
        res = dict(pool.map(job, list(JOBS)))
    days = [x[0] for x in res['U-frozen']]
    out = {'params': {'O': bestO, 'S': bestS}, 'n_test_messages': len(days), 'models': {}}
    for k, r in res.items():
        assert [x[0] for x in r] == days
        out['models'][k] = {'bits': float(np.mean([x[1] for x in r])), 'recall@20': float(np.mean([x[2] for x in r]))}
        print(f"{k:9s} bits {out['models'][k]['bits']:.4f}  recall@20 {out['models'][k]['recall@20']:.4f}")
    for a, b in [('O-frozen', 'S-frozen'), ('O-alive', 'S-alive'), ('O-frozen', 'U-frozen'), ('S-alive', 'S-frozen'), ('O-alive', 'O-frozen')]:
        m, ci = boot([x[1] for x in res[a]], [x[1] for x in res[b]], days)
        out[f'{a} - {b}'] = [float(m), float(ci[0]), float(ci[1])]
        print(f"{a} - {b}: {m:+.4f} bits  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    by_month = {}
    for k, r in res.items():
        mm = collections.defaultdict(list)
        for d, bits, _ in r: mm[d[:7]].append(bits)
        by_month[k] = {m: float(np.mean(v)) for m, v in sorted(mm.items())}
    out['by_month'] = by_month
    for k in by_month: print(k, {m: round(v, 3) for m, v in by_month[k].items()})
    json.dump(out, open('test_results.json', 'w'), indent=1)
