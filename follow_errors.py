"""OCR by following the ERRORS. Level 1 (similarity, squeezed room) finds the coarse shapes
but blurs the near-twins. Level 2 never looks at the shapes again: for each held shape it
watches only the error x - shape, pixel by pixel, while the stream continues. Noise errors
are scattered and random; the twin difference keeps recurring in the same pixels. So the
pixels with the most error-variance are where the hidden distinction lives. Split on those.
Self-contained: python3 follow_errors.py (~10 s, needs numpy)."""
import numpy as np
def make_letters(rng):
    base=(rng.random((3,64))<0.5).astype(float); L=[]
    for b in base:
        t=b.copy(); idx=rng.choice(64,6,replace=False); t[idx]=1-t[idx]; L+= [b,t]
    return np.array(L)
def sample(L,rng,noise=0.15):
    k=rng.integers(len(L)); x=L[k].copy(); f=rng.random(64)<noise; x[f]=1-x[f]; return x,k

def level1(L, rng, room=6, steps=4000, theta=3.6, lr=0.05):
    P=[]; W=[]
    for _ in range(steps):
        x,_=sample(L,rng)
        if not P: P.append(x.copy()); W.append(1); continue
        d=np.linalg.norm(np.array(P)-x,axis=1); k=d.argmin()
        if d[k]>theta:
            if len(P)>=room:
                A=np.array(P); D=np.linalg.norm(A[:,None]-A[None],axis=2); np.fill_diagonal(D,1e9)
                i,j=np.unravel_index(D.argmin(),D.shape); P[i]=(W[i]*P[i]+W[j]*P[j])/(W[i]+W[j]); W[i]+=W[j]; del P[j]; del W[j]
            P.append(x.copy()); W.append(1)
        else: P[k]+=lr*(x-P[k]); W[k]+=1
    return np.array(P)

def level2(P, L, rng, steps=4000):
    K=len(P); n=np.zeros(K); m=np.zeros((K,64)); s2=np.zeros((K,64))
    for _ in range(steps):                         # watch errors only
        x,_=sample(L,rng); k=np.linalg.norm(P-x,axis=1).argmin(); e=x-P[k]
        n[k]+=1; dlt=e-m[k]; m[k]+=dlt/n[k]; s2[k]+=dlt*(e-m[k])
    var=s2/np.maximum(n,1)[:,None]
    split={}
    for k in range(K):
        if n[k]<50: continue
        v=np.sort(var[k])[::-1]; gaps=v[:-1]-v[1:]; cut=gaps[:20].argmax()+1   # largest jump = where error stops being noise
        if v[cut-1] < 1.4*np.median(v): continue                               # no recurring error: nothing hidden here
        pix=np.argsort(var[k])[::-1][:cut]
        c=np.array([np.zeros(cut), np.ones(cut)])                              # two guesses for the hidden pixels
        for _ in range(3000):
            x,_=sample(L,rng)
            if np.linalg.norm(P-x,axis=1).argmin()!=k: continue
            z=x[pix]; j=np.linalg.norm(c-z,axis=1).argmin(); c[j]+=0.05*(z-c[j])
        split[k]=(pix,c)
    return split, var

def classify_(P, split, X):
    out=[]
    for x in X:
        k=int(np.linalg.norm(P-x,axis=1).argmin())
        if k in split:
            pix,c=split[k]; out.append(2*k+int(np.linalg.norm(c-x[pix],axis=1).argmin()))
        else: out.append(2*k)
    return np.array(out)

def score(w,Y):
    m={j:np.bincount(Y[w==j],minlength=6).argmax() for j in set(w)}; return np.mean([m[j]==y for j,y in zip(w,Y)])

res=[]; found=[]
for seed in range(20):
    rng=np.random.default_rng(seed); L=make_letters(rng)
    P=level1(L,rng); split,var=level2(P,L,rng)
    X,Y=zip(*[sample(L,rng) for _ in range(600)]); X=np.array(X); Y=np.array(Y)
    sim=score(np.linalg.norm(X[:,None]-P[None],axis=2).argmin(1),Y)
    err=score(classify_(P,split,X),Y)
    # did the error pixels land on the true twin differences?
    hits=[]
    for k,(pix,_) in split.items():
        xs=[sample(L,rng) for _ in range(200)]; lab=[y for x,y in xs if np.linalg.norm(P-x,axis=1).argmin()==k]
        if not lab: continue
        b=np.bincount(lab,minlength=6).argmax()//2; true=np.nonzero(L[2*b]!=L[2*b+1])[0]
        hits.append(len(set(pix)&set(true))/len(true))
    res.append((sim,err,len(P)+2*len(split))); found.append(np.mean(hits) if hits else 0)
R=np.array(res)
print("20 worlds, 6 hidden letters (3 shapes + near-twins), no labels\n")
print(f"similarity only, room 6     : letters told apart {np.median(R[:,0])*100:5.1f}%")
print(f"+ following the errors      : letters told apart {np.median(R[:,1])*100:5.1f}%   (held things: {np.median(R[:,2]):.0f})")
print(f"worlds where errors won     : {np.mean(R[:,1]>R[:,0]+0.05)*100:.0f}%")
print(f"error-pixels that ARE the true twin differences: {np.median(found)*100:.0f}% (median)")
