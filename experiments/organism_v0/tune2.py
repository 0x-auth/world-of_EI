import json, multiprocessing as mp
from tune import job
if __name__=='__main__':
    jobs=[('O',dict(alpha=a,eta=e,seed=s)) for a in (0,0.5,0.8) for e in (0.25,0.5,1) for s in (0,2)]
    jobs+=[('S',dict(alpha=a,lam=l)) for a in (0,0.5,0.8) for l in (0.95,0.99)]
    with mp.Pool(2) as pool:
        out=[]
        for r in pool.imap_unordered(job, jobs): print(r, flush=True); out.append(r)
    old=json.load(open('tune_results.json')); json.dump(old+out, open('tune_results_all.json','w'), indent=1)
