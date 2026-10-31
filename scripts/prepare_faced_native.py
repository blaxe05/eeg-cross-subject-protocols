"""Prepare provider DE features and orders for labelled FACED reconstruction."""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import pickle
import numpy as np
from scipy.io import loadmat
import faced_svm_common as shared
import run_controlled_revision as base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir/'native/prepared';out.mkdir(parents=True,exist_ok=True)
    lease=shared.acquire_lease(out/'prepare.lock')
    if (out/'inputs.json').exists():
        manifest=shared.read(out/'inputs.json')
        assert manifest['base_sha256']==shared.sha(out/'provider_de.npy')
        print('NATIVE_INPUTS_COMPLETE',flush=True);return
    original=a.data_root/'Code/Validation/Classification_validation/Svm_analysis'
    files=sorted((a.data_root/'EEG_Features/DE').glob('sub*.pkl*'))
    assert [p.name.split('.')[0] for p in files]==[f'sub{i:03d}' for i in range(123)]
    arrays=[];orders=[];input_hashes={}
    for index,path in enumerate(files):
        with path.open('rb') as stream:de=pickle.load(stream)
        assert de.shape==(28,32,30,5) and np.isfinite(de).all()
        # The released validation extractor excludes the two mastoid channels.
        arrays.append(de[:,:30,:,:].transpose(0,2,3,1))
        remark=original/'After_remarks'/f'sub{index:03d}'/'After_remarks.mat'
        entries=loadmat(remark,simplify_cells=True)['After_remark']
        order=np.array([int(v['vid'])-1 for v in entries])
        assert np.array_equal(np.sort(order),np.arange(28));orders.append(order)
        input_hashes[path.name]=shared.sha(path);input_hashes[f'sub{index:03d}_remarks']=shared.sha(remark)
    values=np.array(arrays,dtype=np.float64)
    np.save(out/'provider_de.npy',values);np.save(out/'presentation_order.npy',np.array(orders))
    # Check a fixed, outcome-independent sample against the released extractor formula.
    import mne
    checks=[]
    for subject in [0,80]:
        with (a.data_root/'Processed_data'/f'sub{subject:03d}.pkl').open('rb') as stream:raw=pickle.load(stream)
        for video in [0,18]:
            for band,(low,high) in enumerate([(1,4),(4,8),(8,14),(14,30),(30,47)]):
                filtered=mne.filter.filter_data(np.asarray(raw[video,:30,:],dtype=np.float64),250,l_freq=low,h_freq=high,verbose=False)
                de=.5*np.log(2*np.pi*np.e*np.var(filtered.reshape(30,30,250),axis=2))
                error=float(np.max(np.abs(de.T-values[subject,video,:,band,:])))
                checks.append({'subject':subject,'video':video,'band':band,'max_abs_error':error})
    scripts={p.name:shared.sha(p) for p in original.glob('*.py')}
    tree=ast.parse((original/'smooth_lds.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='LDS')
    namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<provider LDS>','exec'),namespace)
    sample=values[0].reshape(840,150)
    np.save(out/'provider_lds_sample.npy',namespace['LDS'](sample))
    manifest={'status':'complete','subjects':123,'base_shape':list(values.shape),'base_sha256':shared.sha(out/'provider_de.npy'),
              'input_hashes':input_hashes,'released_code_sha256':scripts,'mne':mne.__version__,
              'raw_feature_checks':checks,'raw_feature_check_tolerance':1e-6,
              'raw_feature_checks_pass':all(c['max_abs_error']<=1e-6 for c in checks),
              'unmodified_pipeline_executable':False,
              'released_inconsistencies':['Extractor produces five bands; normalization reshapes for four.',
                                          'Final normalization prior excludes 11 subjects; classifier holds out 15.'],
              'native_selection':'held_out_fold_accuracy_chooses_C; diagnostic reconstruction only',
              'published_accuracy_pct':{'binary':69.3,'nine':35.2},
              'published_sem_pp':{'binary':1.5,'nine':1.0}}
    original_results={}
    import pandas as pd
    for task,n in [('binary',24),('nine',28)]:
        path=original.parent/f'subject_cross_vids_{n}_valid_10-folds.csv'
        if path.exists():
            frame=pd.read_csv(path);scores=frame.iloc[:,-1].to_numpy(dtype=float)
            assert len(scores)==123 and np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1))
            original_results[task]={'mean_accuracy_pct':float(100*scores.mean()),
                                    'sem_pp':float(100*scores.std(ddof=1)/np.sqrt(123)),
                                    'input_sha256':shared.sha(path)}
            np.save(out/f'original_scores_{task}.npy',scores)
    manifest['provided_original_outputs']=original_results
    base.write_json(out/'inputs.json',manifest)
    print('NATIVE_INPUTS_COMPLETE raw_feature_checks='+str(manifest['raw_feature_checks_pass']),flush=True)


if __name__=='__main__':main()
