"""
Synthetic study of PAIEC acquisition policies and Brier-specific choices (A6 literature agent, 2026-10-04).

Protocol mimicked: per subject-benchmark pair with N distinct items, a 50/50 split into acquisition (stream, random order)
and evaluation pools; streaming accept/skip; caps 0,1,3,7,15,31; Brier on eval pool at each cap;
ALC = .1 B0 + .2 B1 + .2 B3 + .2 B7 + .2 B15 + .1 B31; unweighted mean over pairs.
Single pair at a time (no cross-pair borrowing), which isolates the acquisition question.

Generative model (logit scale): y ~ Bern(sigmoid(theta + kappa*eta_j)) with possible "unsolvable" items.
Predictor (Bayesian, 1-D grid over theta, or 2-D grid over (theta, slope)):
  prior theta ~ N(mu0, tau^2); item feature gives m_j = E[eta_j | feature], residual var s2;
  P(y=1 | theta) ~= sigmoid((theta + m_j) / sqrt(1 + pi*s2/8))   (MacKay probit approx)
Expected Brier on eval items is computed exactly given true p_j: (phat - p)^2 + p(1-p).
"""
import numpy as np, sys, json, math, time

def sig(x): return 1.0/(1.0+np.exp(-x))
CAPS = [0,1,3,7,15,31]
W = np.array([.1,.2,.2,.2,.2,.1])

def run(cond, policy, n_pairs=2000, seed=0, tau_assumed=None, plugin=False, fit_slope=False):
    tau = cond.get('tau',1.2); sig_eta = cond.get('sig_eta',1.8); rho = cond['rho']
    tau_a = tau if tau_assumed is None else tau_assumed
    s2 = sig_eta**2*(1-rho**2)
    mod = 1.0/math.sqrt(1+math.pi*s2/8)
    G = 121
    out = np.zeros((n_pairs,6)); n_used = np.zeros((n_pairs,6))
    Ns = cond.get('Ns',[80,100,140,200])
    for p in range(n_pairs):
        rng = np.random.default_rng([seed, p])
        N = int(rng.choice(Ns))
        n_acq = (N+1)//2; n_ev = N//2
        mu0 = rng.normal(0,1.0)
        theta = mu0 + rng.normal(0,tau)
        eta = rng.normal(0,sig_eta,N)
        z = rho*eta/sig_eta + math.sqrt(1-rho**2)*rng.normal(size=N)
        m = rho*sig_eta*z                      # predictor's posterior mean of eta given feature (assumes kappa=1)
        kappa = 1.0
        if cond.get('kappa_sd',0)>0: kappa = math.exp(rng.normal(0,cond['kappa_sd']))
        lam = 1.0
        if 'lam' in cond: lam = rng.uniform(*cond['lam'])   # feature validity on this benchmark (can be useless/wrong)
        if cond.get('clusters',0)>0:
            # two item clusters (e.g., formats) with different predicted difficulty AND an unmodelled
            # cluster-specific subject offset; predictor only has a single pair offset (misspecified)
            cl = (rng.random(N) < 0.5).astype(int)
            shift = cond['clusters']*(2*cl-1)
            eta = eta + shift
            m = m + shift                      # features see the cluster shift (predicted difficulty differs)
            dlt = rng.normal(0, cond.get('cl_sd',0.8), 2)
            eta = eta + dlt[cl]                # subject-specific cluster effect not captured by features
        true_logit = theta + kappa*eta
        if cond.get('lam_on_truth',False):
            pass
        ptrue = sig(true_logit)
        if cond.get('unsolv',0)>0:
            bad = rng.random(N) < cond['unsolv']
            ptrue = np.where(bad, 0.01, ptrue)
        if 'lam' in cond:
            # features relate to truth with pair-specific validity lam: truth uses lam-scaled relation
            ptrue = sig(theta + lam*kappa*eta + math.sqrt(max(0.0,1-lam**2))*0.0)
            if cond.get('unsolv',0)>0: ptrue = np.where(bad, 0.01, ptrue)
        idx = rng.permutation(N); acq = idx[:n_acq]; ev = idx[n_acq:]
        U = rng.random(N)                      # common random numbers for acquisition responses
        ref_m = rho*sig_eta*rng.normal(size=60) # reference feature sample (stands in for training-pool feature distribution)
        # grids
        if not fit_slope:
            grid = mu0 + tau_a*np.linspace(-5,5,G)
            logpost = -0.5*((grid-mu0)/tau_a)**2
            def post_w(lp):
                w = np.exp(lp-lp.max()); return w/w.sum()
            def pred(items, lp):
                w = post_w(lp)
                if plugin:
                    th = grid[np.argmax(lp)]
                    return sig(th + m[items])
                return (w[:,None]*sig((grid[:,None]+m[items][None,:])*mod)).sum(0)
            def loglik(i, y):
                pi_ = sig((grid+m[i])*mod)
                return np.log(pi_ if y==1 else 1-pi_)
            def pred_ref(mv, lp):
                w = post_w(lp)
                return (w[:,None]*sig((grid[:,None]+mv[None,:])*mod)).sum(0)
        else:
            gt = mu0 + tau_a*np.linspace(-5,5,61)
            gk = np.exp(np.linspace(-1.2,1.2,17))
            TT, KK = np.meshgrid(gt, gk, indexing='ij')
            logpost = -0.5*((TT-mu0)/tau_a)**2 - 0.5*(np.log(KK)/0.5)**2
            def post_w(lp):
                w = np.exp(lp-lp.max()); return w/w.sum()
            def pred(items, lp):
                w = post_w(lp)
                if plugin:
                    a,b = np.unravel_index(np.argmax(lp), lp.shape)
                    return sig(TT[a,b] + KK[a,b]*m[items])
                return np.einsum('ab,abn->n', w, sig((TT[:,:,None]+KK[:,:,None]*m[items][None,None,:])*mod))
            def loglik(i, y):
                pi_ = sig((TT+KK*m[i])*mod)
                return np.log(pi_ if y==1 else 1-pi_)
            def pred_ref(mv, lp):
                w = post_w(lp)
                return np.einsum('ab,abn->n', w, sig((TT[:,:,None]+KK[:,:,None]*mv[None,None,:])*mod))
        def brier(lp):
            ph = pred(ev, lp); pt = ptrue[ev]
            return np.mean((ph-pt)**2 + pt*(1-pt))
        out[p,0] = brier(logpost)
        labels = 0; ci = 1
        seen = []
        for t, c in enumerate(acq):
            items_rem = n_acq - t; lab_rem = 31 - labels
            if lab_rem <= 0: break
            seen.append(c)
            accept = True
            if policy != 'random' and items_rem > lab_rem:
                use_u = (policy=='uncert') or (policy=='uncert_early' and labels < 7) or (policy=='spread') or (policy=='spread_early' and labels<7)
                if use_u:
                    ph_seen = pred(np.array(seen), logpost)
                    if len(seen) < 15:
                        # append predictions for reference features so early decisions have a sensible threshold
                        ph_ref = pred_ref(ref_m, logpost)
                        ph_all = np.concatenate([ph_ref, ph_seen])
                    else:
                        ph_all = ph_seen
                    if policy.startswith('uncert'):
                        score = ph_all*(1-ph_all)
                    else:  # D-opt for offset+slope: target p near .18/.82 -> score = closeness to those
                        score = -np.minimum(np.abs(ph_all-0.18), np.abs(ph_all-0.82))
                    q = lab_rem/items_rem          # fraction we must accept on average from here on
                    thr = np.quantile(score, 1-q)
                    accept = score[-1] >= thr
            if accept:
                y = 1 if U[c] < ptrue[c] else 0
                logpost = logpost + loglik(c, y)
                labels += 1
                while ci < 6 and labels == CAPS[ci]:
                    out[p,ci] = brier(logpost); n_used[p,ci]=labels; ci += 1
        while ci < 6:
            out[p,ci] = brier(logpost); n_used[p,ci]=labels; ci += 1
    alc = out@W
    return dict(alc=float(alc.mean()), alc_se=float(alc.std()/math.sqrt(n_pairs)), B=[float(x) for x in out.mean(0)],
                used31=float(n_used[:,5].mean()), alc_pairs=alc.tolist(), B_pairs=out.tolist())

if __name__ == '__main__':
    conds = {
      'C1_wellspec_rho.6': dict(rho=0.6),
      'C2_wellspec_rho.3': dict(rho=0.3),
      'C3_wellspec_rho0':  dict(rho=0.0),
      'C4_slope_mis_rho.6': dict(rho=0.6, kappa_sd=0.5),
      'C5_unsolvable20_rho.6': dict(rho=0.6, unsolv=0.2),
      'C6_feature_validity_U(-.5,1.5)_rho.6': dict(rho=0.6, lam=(-0.5,1.5)),
    }
    which = sys.argv[1] if len(sys.argv)>1 else 'main'
    res = {}
    t0=time.time()
    if which=='main':
        for cn, c in conds.items():
            for pol in ['random','uncert','uncert_early']:
                r = run(c, pol, n_pairs=int(sys.argv[2]) if len(sys.argv)>2 else 1500, seed=11)
                res[f'{cn}|{pol}'] = r
                print(cn, pol, 'ALC=%.4f +- %.4f'%(r['alc'],r['alc_se']), 'B=',' '.join('%.4f'%b for b in r['B']), 'used31=%.1f'%r['used31'], '%.0fs'%(time.time()-t0), flush=True)
    elif which=='brier':
        c = conds['C1_wellspec_rho.6']
        for name, kw in [('bayes_moderated',{}),('plugin_MAP',dict(plugin=True)),('tau_too_small_0.4',dict(tau_assumed=0.4)),('tau_too_big_3.0',dict(tau_assumed=3.0))]:
            r = run(c,'random',n_pairs=int(sys.argv[2]) if len(sys.argv)>2 else 1500, seed=11, **kw)
            res[name]=r
            print('brier-choices', name, 'ALC=%.4f +- %.4f'%(r['alc'],r['alc_se']), 'B=',' '.join('%.4f'%b for b in r['B']), flush=True)
    elif which=='slope':
        c = conds['C4_slope_mis_rho.6']
        for pol in ['random','uncert','spread','spread_early']:
            r = run(c, pol, n_pairs=int(sys.argv[2]) if len(sys.argv)>2 else 800, seed=11, fit_slope=True)
            res[f'slopefit|{pol}']=r
            print('slope-fit predictor', pol, 'ALC=%.4f +- %.4f'%(r['alc'],r['alc_se']), 'B=',' '.join('%.4f'%b for b in r['B']), flush=True)
    json.dump(res, open(f'sim_results_{which}.json','w'), indent=1)
