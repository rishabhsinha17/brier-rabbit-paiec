import numpy as np, json, math, sys, time
from sim_acquisition import run
NP = int(sys.argv[1]) if len(sys.argv)>1 else 3000
conds = {
  'C9 two item clusters (shift +-1.5) with unmodelled per-cluster subject offsets sd .8, rho=0.6': dict(rho=0.6, clusters=1.5, cl_sd=0.8),
  'C10 same but shift +-2.5, cl_sd 1.2': dict(rho=0.6, clusters=2.5, cl_sd=1.2),
}
lines=[]; res={}
for cn, c in conds.items():
    base = run(c, 'random', n_pairs=NP, seed=7)
    a0 = np.array(base['alc_pairs']); B0 = np.array(base['B_pairs'])
    line = f"{cn}\n   random      ALC={base['alc']:.4f}  B@0,1,3,7,15,31= " + ' '.join(f'{b:.4f}' for b in base['B'])
    print(line, flush=True); lines.append(line); res[cn]={'random':base['B']}
    for pol in ['uncert','uncert_early']:
        r = run(c, pol, n_pairs=NP, seed=7)
        a = np.array(r['alc_pairs']); B = np.array(r['B_pairs']); d=a-a0; dB=(B-B0).mean(0)
        line = f"   {pol:12s}ALC={r['alc']:.4f}  dALC vs random={d.mean():+.5f} (se {d.std()/math.sqrt(NP):.5f})  dB@1,3,7,15,31= " + ' '.join(f'{x:+.4f}' for x in dB[1:])
        print(line, flush=True); lines.append(line); res[cn][pol]=dict(dALC=float(d.mean()), se=float(d.std()/math.sqrt(NP)), dB=dB.tolist())
open('sim_results_driver2.txt','w').write('\n'.join(lines)+'\n'); json.dump(res,open('sim_results_driver2.json','w'),indent=1)
