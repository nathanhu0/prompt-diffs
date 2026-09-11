"""Per-method compute for the optimizer-comparison appendix: scored proposals per
run (from the result JSONs the animal tables use) and wall-clock per run
(`extra.optimizer_sec` where the run recorded it, else SLURM elapsed via sacct on
the job ids logged in .commands_auto.sh), bucketed by GPU class. Run from repo root."""
import re, subprocess, json, statistics, collections
from pathlib import Path
GPU = {}
for l in subprocess.run(['sinfo','-N','-h','-o','%N %G'],capture_output=True,text=True).stdout.splitlines():
    p=l.split(); GPU[p[0]]=p[1].split(':')[1] if len(p)>1 and ':' in p[1] else '?'
CLASS = {'a100':'A100-80G','h100':'H100-80G','a6000':'48G','rtx6000ada':'48G','l40s':'48G','a40':'48G'}
lines = Path('.commands_auto.sh').read_text().splitlines()
jobs = {}
for l in lines:
    m = re.search(r'job=(\d+) ebatch (\S+) (\S+) (.*)', l)
    if not m or 'run_comparison.py' not in m.group(4): continue
    cmd = m.group(4); out = re.search(r'--output (\S+)', cmd); topic = re.search(r'--topic (\w+)', cmd)
    if not out or not topic: continue
    out, topic = out.group(1), topic.group(1)
    if topic not in ('cat','dog','eagle','owl'): continue
    if 'optimizer_comparison_schrodi' in out: tree = 'largo_t25' if 'largo_t25' in out else 'main'
    elif 'induction_methods/Qwen2.5-7B-Instruct/filtered_schrodi' in out: tree = 'ind_finalpool' if 'finalpool' in out else 'ind'
    else: continue
    method = '+'.join(Path(c).stem for c in re.findall(r'--config (\S+)', cmd))
    seed = re.search(r'seed=(\d+)', cmd); seed = int(seed.group(1)) if seed else 42
    jobs[m.group(1)] = (tree, method, topic, seed)
res = subprocess.run(['sacct','-j',','.join(jobs),'-X','-n','-P','--format=JobID,State,Elapsed,NodeList'],capture_output=True,text=True)
elapsed = {}
for l in res.stdout.splitlines():
    jid,state,el,node = l.split('|')
    if state!='COMPLETED': continue
    p = el.split(':'); h = int(p[0]) if '-' not in p[0] else int(p[0].split('-')[0])*24+int(p[0].split('-')[1])
    elapsed[jobs[jid]] = (h+int(p[1])/60+int(p[2])/3600, CLASS.get(GPU.get(node,'?'),GPU.get(node,'?')), node)
SCR=Path("/nlp/scr/nathu/latent_rewrite/optimizer_comparison_schrodi"); IND=Path("/nlp/scr/nathu/latent_rewrite/induction_methods/Qwen2.5-7B-Instruct/filtered_schrodi")
# cells as the table uses them
cells = []  # (label, seed, task, json_path, job_key)
for seed in [42,43,44,45,46]:
  for task in ["cat","dog","eagle","owl"]:
    d=SCR/f"seed{seed}"/"filtered_schrodi"/task
    for label,stem,jm in [("SALVE","salve_beam","salve"),("GCG","gcg_L30","gcg+gcg_polish"),("GCG-reg","gcg_polish_L30","gcg+gcg_polish"),("OPRO","opro","opro"),("PGD","pgd_noaux_L30","pgd_noaux"),("AutoDAN","autodan_L64","autodan"),("GBDA","gbda_L30","gbda"),("GBDA-reg","gbda_fluency_L30","gbda_fluency")]:
        if (d/f"{stem}.json").exists(): cells.append((label,seed,task,d/f"{stem}.json",('main',jm,task,seed)))
    j=SCR/"largo_t25"/f"seed{seed}"/"filtered_schrodi"/task/"largo.json"
    if j.exists(): cells.append(("LARGO",seed,task,j,('largo_t25','largo',task,seed)))
    if task!="cat" and seed<=45:
        j=IND/f"seed{seed}_finalpool"/"prefill_t1"/task/"salve_beam.json"
        if j.exists(): cells.append(("SALVE",seed,task,j,('ind_finalpool','salve',task,seed)))
        j=IND/f"seed{seed}"/"prefill_t1"/task/"largo.json"
        if j.exists(): cells.append(("LARGO",seed,task,j,('ind','largo',task,seed)))
seen=set(); rows=collections.defaultdict(list)
for label,seed,task,p,jk in cells:
    if (label,seed,task) in seen: continue
    seen.add((label,seed,task))
    r=json.load(open(p)); osec=(r.get('extra') or {}).get('optimizer_sec')
    el=elapsed.get(jk)
    rows[label].append((seed,task,r.get('n_proposals'),osec/3600 if osec else None, el[0] if el else None, el[1] if el else None, el[2] if el else None))
print(f"{'method':9s} {'cells':>5s} {'n_prop(med)':>12s} | optimizer_sec hours by GPU class (median [n]) | job elapsed hours by GPU class (median [n])")
for label in ["SALVE","LARGO","GCG","GCG-reg","OPRO","PGD","AutoDAN","GBDA","GBDA-reg"]:
    v=rows[label]
    def by_class(idx):
        g=collections.defaultdict(list)
        for x in v:
            if x[idx] is not None and x[5]: g[x[5]].append(x[idx])
        return '  '.join(f"{k}:{statistics.median(vs):.2f}[{len(vs)}]" for k,vs in sorted(g.items()))
    nprop=[x[2] for x in v if x[2] is not None]
    print(f"{label:9s} {len(v):5d} {statistics.median(nprop) if nprop else float('nan'):12.0f} | {by_class(3):45s} | {by_class(4)}")
print("\nunjoined cells (no COMPLETED job found in .commands_auto.sh):")
for label,v in rows.items():
    for x in v:
        if x[5] is None: print("  ",label,x[:2])
