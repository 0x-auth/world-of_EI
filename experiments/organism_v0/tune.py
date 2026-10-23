import os
DH = os.environ.get("DH_DIR", "data")
import pickle, itertools, json, numpy as np, multiprocessing as mp
import organism as O
vocab, msgs = pickle.load(open(os.path.join(DH, 'stream.pkl'),'rb')); V=len(vocab)
def job(a):
    kind, p = a
    r,_ = O.run(msgs, V, kind, p, O.VAL, False)
    return kind, p, float(np.mean([x[1] for x in r])), float(np.mean([x[2] for x in r]))
if __name__=='__main__':
    jobs=[('O',dict(alpha=a,eta=e,seed=s)) for a in (0,0.5,0.8) for e in (2,5,10,20,40) for s in (0,2)]
    jobs+=[('S',dict(alpha=a,lam=l)) for a in (0,0.5,0.8) for l in (0.3,0.5,0.7,0.9)]
    jobs+=[('S',dict(alpha=0,lam=0))]
    with mp.Pool(2) as pool:
        out=[]
        for r in pool.imap_unordered(job, jobs):
            print(r, flush=True); out.append(r)
    json.dump(out, open('tune_results.json','w'), indent=1)
