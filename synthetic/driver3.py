import numpy as np, math
from sim_acquisition import run
c=dict(rho=0.6); NP=3000
base=run(c,'random',n_pairs=NP,seed=7); a0=np.array(base['alc_pairs']); B0=np.array(base['B_pairs'])
lines=[f"Brier-choice study (C1 well-specified, rho=.6, random acquisition, true tau=1.2)\n   bayes_moderated  ALC={base['alc']:.4f} B="+' '.join(f'{b:.4f}' for b in base['B'])]
for name,kw in [('plugin_MAP (no moderation)',dict(plugin=True)),('prior tau=0.4 (over-shrunk)',dict(tau_assumed=0.4)),('prior tau=0.8',dict(tau_assumed=0.8)),('prior tau=2.0',dict(tau_assumed=2.0)),('prior tau=3.0 (under-shrunk)',dict(tau_assumed=3.0))]:
    r=run(c,'random',n_pairs=NP,seed=7,**kw); a=np.array(r['alc_pairs']); d=a-a0; dB=(np.array(r['B_pairs'])-B0).mean(0)
    lines.append(f"   {name:28s} ALC={r['alc']:.4f} dALC={d.mean():+.5f} (se {d.std()/math.sqrt(NP):.5f}) dB@0,1,3,7,15,31= "+' '.join(f'{x:+.4f}' for x in dB))
print('\n'.join(lines)); open('sim_results_brier.txt','w').write('\n'.join(lines)+'\n')
