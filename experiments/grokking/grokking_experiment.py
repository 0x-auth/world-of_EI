import torch, torch.nn as nn, json, sys, time
torch.manual_seed(0)
P = 97
TRAIN_FRAC = 0.4
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
WD = float(sys.argv[1])

a = torch.arange(P).repeat_interleave(P)
b = torch.arange(P).repeat(P)
y = (a * b) % P
perm = torch.randperm(P * P)
n_tr = int(TRAIN_FRAC * P * P)
tr, te = perm[:n_tr], perm[n_tr:]

class Net(nn.Module):
    def __init__(s, d=128, h=512):
        super().__init__()
        s.ea = nn.Embedding(P, d); s.eb = nn.Embedding(P, d)
        s.mlp = nn.Sequential(nn.Linear(2 * d, h), nn.ReLU(), nn.Linear(h, P))
    def forward(s, a, b):
        return s.mlp(torch.cat([s.ea(a), s.eb(b)], -1))

m = Net()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=WD, betas=(0.9, 0.98))
lossf = nn.CrossEntropyLoss()
log = []
t0 = time.time()
for step in range(STEPS + 1):
    m.train()
    out = m(a[tr], b[tr]); loss = lossf(out, y[tr])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        m.eval()
        with torch.no_grad():
            tra = (m(a[tr], b[tr]).argmax(-1) == y[tr]).float().mean().item()
            tea = (m(a[te], b[te]).argmax(-1) == y[te]).float().mean().item()
            wn = sum(p.pow(2).sum() for p in m.parameters()).sqrt().item()
        log.append(dict(step=step, train=tra, test=tea, wnorm=wn))
        if step % 1000 == 0:
            print(f"wd={WD} step={step} train={tra:.3f} test={tea:.3f} |w|={wn:.1f} t={time.time()-t0:.0f}s", flush=True)
json.dump(log, open(f"results/grokking_log_wd{WD}.json", "w"))
