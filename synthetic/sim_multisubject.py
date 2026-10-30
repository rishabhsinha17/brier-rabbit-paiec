"""Cross-pair borrowing: K subjects on the SAME held-out benchmark, shared item effects, independent 50/50 splits per pair.
true logit = theta_k + c + eta_j; prior theta_k ~ N(mu_k, tau^2) (mu_k known from training), c ~ N(0, tau_c^2) (unknown
benchmark shift shared by all subjects), eta_j ~ N(m_j, s2) (m_j from item features with corr rho).
Predictor: joint MAP + Laplace (Newton on 1+K+N dims), moderated predictions using predictive variance.
Policies: random | fresh_first (prefer items not yet labeled for any subject, while slack allows) |
          bald_theta (BALD wrt this subject's theta+c at stage-start posterior, ignoring item learning)."""
import numpy as np, math, sys
def sig(x): return 1/(1+np.exp(-x))
def Hh(p): p=np.clip(p,1e-12,1-1e-12); return -(p*np.log(p)+(1-p)*np.log(1-p))
CAPS=[0,1,3,7,15,31]; W=np.array([.1,.2,.2,.2,.2,.1])

def fit(K,N,mu,tau,tau_c,m,s2,obs):
    # params x = [c, th_1..th_K, eta_1..eta_N]; prior precision diag
    D=1+K+N
    prior_mean=np.concatenate([[0.0],mu,m]); prior_prec=np.concatenate([[1/tau_c**2],np.full(K,1/tau**2),np.full(N,1/s2)])
    x=prior_mean.copy()
    if len(obs)==0:
        return x, np.diag(1/prior_prec)
    ks=np.array([o[0] for o in obs]); js=np.array([o[1] for o in obs]); ys=np.array([o[2] for o in obs],float)
    for it in range(25):
        z=x[0]+x[1+ks]+x[1+K+js]; p=sig(z); r=ys-p; wgt=p*(1-p)
        g=-prior_prec*(x-prior_mean)
        np.add.at(g,0,r.sum()); np.add.at(g,1+ks,r); np.add.at(g,1+K+js,r)
        Hm=np.diag(prior_prec).astype(float)
        idx=[np.zeros_like(ks),1+ks,1+K+js]
        for a in idx:
            for b in idx:
                np.add.at(Hm,(a,b),wgt)
        step=np.linalg.solve(Hm,g); x=x+step
        if np.abs(step).max()<1e-6: break
    cov=np.linalg.inv(Hm)
    return x,cov

def predict(K,N,x,cov,k,items):
    i0=0; i1=1+k; i2=1+K+items
    mean=x[i0]+x[i1]+x[i2]
    var=cov[i0,i0]+cov[i1,i1]+cov[i2,i2]+2*cov[i0,i1]+2*cov[i0,i2]+2*cov[i1,i2]
    return sig(mean/np.sqrt(1+math.pi*var/8))

def run(policy,K=4,N=120,reps=600,seed=3,tau=0.8,tau_c=1.2,sig_eta=1.8,rho=0.6):
    allB=[]
    for rep in range(reps):
        rng=np.random.default_rng([seed,rep,K])
        mu=rng.normal(0,1,K); theta=mu+rng.normal(0,tau,K); c=rng.normal(0,tau_c)
        eta=rng.normal(0,sig_eta,N); z=rho*eta/sig_eta+math.sqrt(1-rho**2)*rng.normal(size=N)
        m=rho*sig_eta*z; s2=sig_eta**2*(1-rho**2)
        PT=sig(theta[:,None]+c+eta[None,:])
        splits=[rng.permutation(N) for _ in range(K)]
        n_acq=(N+1)//2
        acqs=[sp[:n_acq] for sp in splits]; evs=[sp[n_acq:] for sp in splits]
        U=rng.random((K,N))
        obs=[]; labeled_items=set(); pos=[0]*K; lab=[0]*K
        B=np.zeros((K,6))
        x,cov=fit(K,N,mu,tau,tau_c,m,s2,obs)
        for k in range(K):
            ph=predict(K,N,x,cov,k,evs[k]); q=PT[k,evs[k]]; B[k,0]=np.mean((ph-q)**2+q*(1-q))
        for ci,cap in enumerate(CAPS[1:],start=1):
            for k in range(K):
                while lab[k]<cap and pos[k]<n_acq:
                    cnd=acqs[k][pos[k]]; items_rem=n_acq-pos[k]; lab_rem=31-lab[k]; pos[k]+=1
                    acc=True
                    if items_rem>lab_rem and policy!='random':
                        if policy=='fresh_first':
                            acc = cnd not in labeled_items
                        elif policy=='bald_theta':
                            # score all remaining-like items: use this subject's acquisition pool as reference
                            ref=acqs[k][:pos[k]]
                            mean=x[0]+x[1+k]+x[1+K+ref]
                            vt=cov[0,0]+cov[1+k,1+k]+2*cov[0,1+k]
                            vi=np.diag(cov)[1+K+ref]
                            gg=np.linspace(-3,3,13); wq=np.exp(-gg**2/2); wq/=wq.sum()
                            pp=sig((mean[None,:]+gg[:,None]*math.sqrt(vt))/np.sqrt(1+math.pi*vi[None,:]/8))
                            pbar=(wq[:,None]*pp).sum(0); score=Hh(pbar)-(wq[:,None]*Hh(pp)).sum(0)
                            thr=np.quantile(score,1-lab_rem/items_rem); acc=score[-1]>=thr
                    if acc:
                        y=1 if U[k,cnd]<PT[k,cnd] else 0
                        obs.append((k,int(cnd),y)); labeled_items.add(int(cnd)); lab[k]+=1
            x,cov=fit(K,N,mu,tau,tau_c,m,s2,obs)
            for k in range(K):
                ph=predict(K,N,x,cov,k,evs[k]); q=PT[k,evs[k]]; B[k,ci]=np.mean((ph-q)**2+q*(1-q))
        allB.append(B)
    return np.concatenate(allB,0)

if __name__=='__main__':
    reps=int(sys.argv[1]) if len(sys.argv)>1 else 400
    lines=[]
    for K in [1,2,4]:
        base=run('random',K=K,reps=reps); a0=base@W
        lines.append(f"K={K} subjects share the benchmark | random ALC={a0.mean():.4f} B="+' '.join(f'{v:.4f}' for v in base.mean(0)))
        for pol in ['fresh_first','bald_theta']:
            if K==1 and pol=='fresh_first': continue
            o=run(pol,K=K,reps=reps); a=o@W; d=a-a0; dB=(o-base).mean(0)
            lines.append(f"   {pol:12s} ALC={a.mean():.4f} dALC={d.mean():+.5f} (se {d.std()/math.sqrt(len(d)):.5f}) dB@1,3,7,15,31= "+' '.join(f'{v:+.4f}' for v in dB[1:]))
        print('\n'.join(lines[-3:]),flush=True)
    open('sim_results_multisubject.txt','w').write('\n'.join(lines)+'\n')
