"""Read the frozen organism's map W (its Δt=0 snapshot): strongest links and the spectral axes."""
import os
DH = os.environ.get("DH_DIR", "data")
import pickle, numpy as np
vocab, _ = pickle.load(open(os.path.join(DH, 'stream.pkl'), 'rb'))
W = np.load(os.path.join(DH, 'W_frozen.npy')); V = len(vocab)
off = W.copy(); np.fill_diagonal(off, -np.inf)
print('self-persistence (diag) mean %.3f, off-diag mean %.4f, frac off-diag > 0.5: %.5f'
      % (np.diag(W).mean(), W[~np.eye(V, dtype=bool)].mean(), (off > 0.5).mean()))
flat = np.argsort(off, axis=None)[::-1][:40]
print('\nstrongest links  (saying A -> next you say B):')
for k in flat:
    i, j = divmod(k, V); print(f'  {vocab[i]:>16s} -> {vocab[j]:<16s} {W[i, j]:.2f}')
S = (W + W.T) / 2; np.fill_diagonal(S, 0)
vals, vecs = np.linalg.eigh(S)
print('\nspectral axes of the symmetric map (largest eigenvalues):')
for e in range(1, 7):
    v = vecs[:, -e]; o = np.argsort(v)
    print(f'  λ={vals[-e]:.2f}  + {[vocab[i] for i in o[::-1][:10]]}\n          - {[vocab[i] for i in o[:10]]}')
