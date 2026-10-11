"""Errors of errors. 12 hidden letters in three tiers:
3 shapes -> each has a twin (6 pixels differ) -> each twin has a sub-twin (2 pixels differ).
Level 1: squeezed similarity finds shapes. Then RECURSE: at every held group, watch the
error, find where it keeps recurring, split there; then do the same to each half's error."""
import numpy as np
def letters(rng):
    out=[]
    for b in (rng.random((3,64))<0.5).astype(float):
        t=b.copy(); i=rng.choice(64,6,replace=False); t[i]=1-t[i]
        for x in (b,t):
            s=x.copy(); j=rng.choice(64,2,replace=False); s[j]=1-s[j]; out+=[x,s]
    return np.array(out)                       # 12
def draw(L,rng,n,noise=0.15):
    k=rng.integers(len(L),size=n); X=L[k].copy(); f=rng.random(X.shape)<noise; X[f]=1-X[f]; return X,k
def level1(X,room=6,theta=3.6,lr=0.05):
    P=[X[0].copy()]; W=[1]
    for x in X[1:]:
        d=np.linalg.norm(np.array(P)-x,axis=1); k=d.argmin()
        if d[k]>theta:
            if len(P)>=room:
                A=np.array(P); D=np.linalg.norm(A[:,None]-A[None],axis=2); np.fill_diagonal(D,1e9)
                i,j=np.unravel_index(D.argmin(),D.shape); P[i]=(W[i]*P[i]+W[j]*P[j])/(W[i]+W[j]); W[i]+=W[j]; del P[j]; del W[j]
            P.append(x.copy()); W.append(1)
        else: P[k]+=lr*(x-P[k]); W[k]+=1
    return np.array(P)
def split(X, depth):
    """Return a function x -> path label, built only from this group's errors."""
    if depth==0 or len(X)<80: return lambda x: ()
    e=X-X.mean(0); v=e.var(0); s=np.sort(v)[::-1]; cut=(s[:-1]-s[1:])[:20].argmax()+1
    if s[cut-1]<1.4*np.median(v): return lambda x: ()     # no recurring error: nothing hidden
    pix=np.argsort(v)[::-1][:cut]; Z=X[:,pix]
    c=np.array([Z[Z.sum(1).argmin()],Z[Z.sum(1).argmax()]]).astype(float)
    for _ in range(20):                                        # 2-means on the error pixels only
        a=np.linalg.norm(Z[:,None]-c[None],axis=2).argmin(1)
        for j in (0,1):
            if (a==j).any(): c[j]=Z[a==j].mean(0)
    kids=[split(X[a==j],depth-1) for j in (0,1)]
    def f(x):
        j=int(np.linalg.norm(c-x[pix],axis=1).argmin()); return (j,)+kids[j](x)
    return f
def acc(lab,Y,K):
    m={}
    for l,y in zip(lab,Y): m.setdefault(l,[]).append(y)
    m={l:np.bincount(v,minlength=K).argmax() for l,v in m.items()}
    return np.mean([m[l]==y for l,y in zip(lab,Y)])
print("12 hidden letters: 3 shapes x twin (6 px) x sub-twin (2 px), 15% noise, no labels, 20 worlds\n")
res={d:[] for d in (0,1,2)}
for seed in range(20):
    rng=np.random.default_rng(seed); L=letters(rng)
    Xs,_=draw(L,rng,4000); P=level1(Xs)
    Xw,_=draw(L,rng,12000); g=np.linalg.norm(Xw[:,None]-P[None],axis=2).argmin(1)
    Xt,Yt=draw(L,rng,1200); gt=np.linalg.norm(Xt[:,None]-P[None],axis=2).argmin(1)
    for d in (0,1,2):
        fs={k:split(Xw[g==k],d) for k in set(g)}
        lab=[(k,)+fs.get(k,lambda x:())(x) for k,x in zip(gt,Xt)]
        res[d].append((acc(lab,Yt//4,3),acc(lab,Yt//2,6),acc(lab,Yt,12),len(set(lab))))
for d,name in ((0,"similarity only"),(1,"+ errors (1 level)"),(2,"+ errors of errors (2)")):
    R=np.array(res[d]); print(f"{name:<24} shapes {np.median(R[:,0])*100:5.1f}%   twins {np.median(R[:,1])*100:5.1f}%   all 12 letters {np.median(R[:,2])*100:5.1f}%   groups {np.median(R[:,3]):.0f}")
