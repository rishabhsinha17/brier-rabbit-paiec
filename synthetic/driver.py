import numpy as np, json, math, sys, time
from sim_acquisition import run
NP = int(sys.argv[1]) if len(sys.argv)>1 else 3000
conds = {
  'C1 well-specified, item-feature corr rho=0.6': dict(rho=0.6),
  'C2 well-specified, rho=0.3': dict(rho=0.3),
  'C3 well-specified, rho=0 (no item signal)': dict(rho=0.0),
  'C4 pair-level slope misspec kappa~LogN(0,.5), rho=0.6': dict(rho=0.6, kappa_sd=0.5),
  'C5 20% unsolvable items (p=.01), rho=0.6': dict(rho=0.6, unsolv=0.2),
  'C6 feature validity lam~U(-.5,1.5) per pair, rho=0.6': dict(rho=0.6, lam=(-0.5,1.5)),
  'C7 small pools only (N=80 -> 40 candidates), rho=0.6': dict(rho=0.6, Ns=[80]),
  'C8 large pools only (N=200 -> 100 candidates), rho=0.6': dict(rho=0.6, Ns=[200]),
}
res = {}
t0 = time.time()
lines = []
for cn, c in conds.items():
    base = run(c, 'random', n_pairs=NP, seed=7)
    a0 = np.array(base['alc_pairs']); B0 = np.array(base['B_pairs'])
    line = f"{cn}\n   random      ALC={base['alc']:.4f}  B@0,1,3,7,15,31= " + ' '.join(f'{b:.4f}' for b in base['B'])
    print(line, flush=True); lines.append(line)
    res[cn] = {'random': {k:v for k,v in base.items() if k not in ('alc_pairs','B_pairs')}}
    for pol in ['uncert','uncert_early']:
        r = run(c, pol, n_pairs=NP, seed=7)
        a = np.array(r['alc_pairs']); B = np.array(r['B_pairs'])
        d = a - a0; dB = (B - B0).mean(0); dBse = (B-B0).std(0)/math.sqrt(NP)
        line = f"   {pol:12s}ALC={r['alc']:.4f}  dALC vs random={d.mean():+.5f} (se {d.std()/math.sqrt(NP):.5f})  dB@1,3,7,15,31= " + ' '.join(f'{x:+.4f}' for x in dB[1:]) + f"  ({time.time()-t0:.0f}s)"
        print(line, flush=True); lines.append(line)
        res[cn][pol] = {k:v for k,v in r.items() if k not in ('alc_pairs','B_pairs')}
        res[cn][pol]['dALC_mean']=float(d.mean()); res[cn][pol]['dALC_se']=float(d.std()/math.sqrt(NP))
        res[cn][pol]['dB']=dB.tolist(); res[cn][pol]['dB_se']=dBse.tolist()
json.dump(res, open('sim_results_driver.json','w'), indent=1)
open('sim_results_driver.txt','w').write('\n'.join(lines)+'\n')
