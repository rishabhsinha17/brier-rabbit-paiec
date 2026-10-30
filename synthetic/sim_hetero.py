"""C11: heteroscedastic item-feature quality. Half the items have informative features (rho=.85), half none (rho=0).
Predictor knows per-item residual variance. Policies: random, naive uncertainty p(1-p), BALD wrt pair offset theta
(mutual information between the label and theta under the current grid posterior). 1-D offset model, Bayes predictor."""
import numpy as np, math, sys, json
def sig(x): return 1/(1+np.exp(-x))
def H(p): p=np.clip(p,1e-12,1-1e-12); return -(p*np.log(p)+(1-p)*np.log(1-p))
CAPS=[0,1,3,7,15,31]; W=np.array([.1,.2,.2,.2,.2,.1])
def run(policy, n_pairs=3000, seed=7, tau=1.2, sig_eta=1.8, rhos=(0.85,0.0), Ns=(80,100,140,200)):
    out=np.zeros((n_pairs,6)); G=121
    for p in range(n_pairs):
        rng=np.random.default_rng([seed,p]); N=int(rng.choice(Ns)); n_acq=(N+1)//2
        mu0=rng.normal(0,1); theta=mu0+rng.normal(0,tau)
        eta=rng.normal(0,sig_eta,N)
        rho=np.where(rng.random(N)<0.5, rhos[0], rhos[1])
        z=rho*eta/sig_eta+np.sqrt(1-rho**2)*rng.normal(size=N)
        m=rho*sig_eta*z; s2=sig_eta**2*(1-rho**2); mod=1/np.sqrt(1+math.pi*s2/8)
        pt=sig(theta+eta)
        idx=rng.permutation(N); acq=idx[:n_acq]; ev=idx[n_acq:]; U=rng.random(N)
        ref=rng.permutation(N)[:60]   # reference sample of feature (m, mod) pairs ~ benchmark feature distribution
        grid=mu0+tau*np.linspace(-5,5,G); lp=-0.5*((grid-mu0)/tau)**2
        def w(lp): e=np.exp(lp-lp.max()); return e/e.sum()
        def P(items,lp):  # grid x items
            return sig((grid[:,None]+m[items][None,:])*mod[items][None,:])
        def pred(items,lp): return (w(lp)[:,None]*P(items,lp)).sum(0)
        def brier(lp): ph=pred(ev,lp); q=pt[ev]; return np.mean((ph-q)**2+q*(1-q))
        out[p,0]=brier(lp); labels=0; ci=1; seen=[]
        for t,c in enumerate(acq):
            items_rem=n_acq-t; lab_rem=31-labels
            if lab_rem<=0: break
            seen.append(c); acc=True
            if policy!='random' and items_rem>lab_rem:
                pool=np.array(seen) if len(seen)>=15 else np.concatenate([ref, np.array(seen)])
                Pm=P(pool,lp); wv=w(lp); pbar=(wv[:,None]*Pm).sum(0)
                if policy=='uncert': score=pbar*(1-pbar)
                else: score=H(pbar)-(wv[:,None]*H(Pm)).sum(0)   # BALD wrt theta
                thr=np.quantile(score,1-lab_rem/items_rem); acc=score[-1]>=thr
            if acc:
                y=1 if U[c]<pt[c] else 0
                pi_=sig((grid+m[c])*mod[c]); lp=lp+np.log(pi_ if y else 1-pi_); labels+=1
                while ci<6 and labels==CAPS[ci]: out[p,ci]=brier(lp); ci+=1
        while ci<6: out[p,ci]=brier(lp); ci+=1
    return out
NP=int(sys.argv[1]) if len(sys.argv)>1 else 3000
base=run('random',NP); a0=base@W
lines=[f"C11 heteroscedastic item features (half rho=.85, half rho=0)\n   random      ALC={a0.mean():.4f} B="+' '.join(f'{x:.4f}' for x in base.mean(0))]
for pol in ['uncert','bald_theta']:
    o=run(pol,NP); a=o@W; d=a-a0; dB=(o-base).mean(0)
    lines.append(f"   {pol:12s}ALC={a.mean():.4f}  dALC vs random={d.mean():+.5f} (se {d.std()/math.sqrt(NP):.5f})  dB@1,3,7,15,31= "+' '.join(f'{x:+.4f}' for x in dB[1:]))
print('\n'.join(lines)); open('sim_results_hetero.txt','w').write('\n'.join(lines)+'\n')
