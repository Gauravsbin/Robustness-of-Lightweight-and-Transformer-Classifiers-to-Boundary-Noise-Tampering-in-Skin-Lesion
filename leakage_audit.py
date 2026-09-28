import pandas as pd, numpy as np, json
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
m = pd.read_csv('/home/claude/ham/HAM10000_metadata')
out = {}
out['n_images'] = len(m); out['n_lesions'] = m.lesion_id.nunique()
out['columns'] = list(m.columns)
cnt = m.dx.value_counts()
out['class_counts'] = cnt.to_dict()
per = m.groupby('lesion_id').size()
out['lesions_multi_image'] = int((per>1).sum())
out['frac_lesions_multi'] = float((per>1).mean())
out['images_in_multi'] = int(per[per>1].sum())
out['frac_images_in_multi'] = float(per[per>1].sum()/len(m))
out['max_images_per_lesion'] = int(per.max())
# image-level stratified 80/20 leakage
seeds=[17,42,123,2024,3407]
y=m.dx.values; les=m.lesion_id.values
res=[]; permel=[]; perclass={c:[] for c in cnt.index}
for s in seeds:
    sss=StratifiedShuffleSplit(1,test_size=0.2,random_state=s)
    tr,te=next(sss.split(m,y))
    trl=set(les[tr]); leak=np.array([l in trl for l in les[te]])
    res.append(float(leak.mean()))
    for c in cnt.index:
        mk=y[te]==c; perclass[c].append(float(leak[mk].mean()))
out['image_split_leak_frac_per_seed']=dict(zip(map(str,seeds),res))
out['image_split_leak_frac_mean']=float(np.mean(res)); out['image_split_leak_frac_sd']=float(np.std(res,ddof=1))
out['image_split_leak_by_class_mean']={c:float(np.mean(v)) for c,v in perclass.items()}
# lesion-grouped stratified 80/20 (5 folds -> 1 fold as test), several seeds
gl=[]
for s in seeds:
    sgk=StratifiedGroupKFold(5,shuffle=True,random_state=s)
    tr,te=next(sgk.split(m,y,groups=les))
    overlap=len(set(les[tr])&set(les[te]))
    d=pd.Series(y[te]).value_counts(normalize=True)
    gl.append(dict(seed=s,n_train=len(tr),n_test=len(te),lesion_overlap=overlap,
                   test_class_frac={c:round(float(d.get(c,0)),4) for c in cnt.index}))
out['grouped_split']=gl
# majority baseline
out['majority_class_frac']=float(cnt.max()/len(m))
# dx_type / dataset composition
out['dx_type']=m.dx_type.value_counts().to_dict()
if 'dataset' in m: out['dataset']=m.dataset.value_counts().to_dict()
json.dump(out,open('/home/claude/leakage_audit.json','w'),indent=1)
print(json.dumps({k:v for k,v in out.items() if k!='grouped_split'},indent=1)); print(gl[0])
