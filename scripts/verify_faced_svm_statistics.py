"""Independently rebuild follow-up contrasts, exact ranks and conditional tests."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,beta
from sklearn.metrics import balanced_accuracy_score,precision_recall_fscore_support,accuracy_score,f1_score
import faced_svm_common as shared
import run_controlled_revision as base


def direct_holm(p):
    order=sorted(range(len(p)),key=p.__getitem__);out=[None]*len(p);running=0.
    for k,i in enumerate(order):running=max(running,(len(p)-k)*p[i]);out[i]=min(1.,running)
    return out


def exact_pratt(values):
    x=np.round(np.asarray(values,dtype=float),10)
    ranks=rankdata(np.abs(x),method='average');weights=np.rint(2*ranks[x!=0]).astype(int)
    if not len(weights):return 1.
    positive=int(np.rint(2*ranks[x>0]).sum());total=int(weights.sum());cutoff=min(positive,total-positive)
    # Encode the subset-sum generating function in a carry-free integer base.
    width=len(weights)+1;polynomial=1
    for weight in weights:polynomial+=polynomial<<(int(weight)*width)
    mask=(1<<width)-1;tail=0
    for k in range(cutoff+1):tail+=polynomial&mask;polynomial>>=width
    return min(1.,2*tail/(1<<len(weights)))


def direct_mean_test(values,cfg,identity):
    x=np.asarray(values,dtype=float);w=np.abs(x[x!=0]);observed=abs(np.sum(x))
    if not len(w):return 1.,[1.,1.]
    tolerance=128*np.finfo(float).eps*max(1.,w.sum())
    if len(w)<=cfg['exact_signflip_max_nonzero_pairs']:
        null=np.array([0.])
        for weight in w:null=np.r_[null-weight,null+weight]
        p=float(np.mean(np.abs(null)>=observed-tolerance));return p,[p,p]
    seed=int.from_bytes(hashlib.sha256((str(cfg['signflip_seed'])+identity).encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed);tail=0;draws=cfg['signflip_monte_carlo_samples']
    for start in range(0,draws,4096):
        bits=rng.integers(0,2,size=(min(4096,draws-start),len(w)),dtype=np.int8)
        sampled=np.where(bits,w,-w).sum(axis=1)
        tail+=int(np.count_nonzero(abs(sampled)>=observed-tolerance))
    p=(tail+1)/(draws+1);alpha=cfg['alpha']/20
    lo=float(beta.ppf(alpha/2,tail,draws-tail+1)) if tail else 0.
    hi=float(beta.ppf(1-alpha/2,tail+1,draws-tail)) if tail<draws else 1.
    return p,[lo,hi]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run-dir','runs-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();analysis=a.run_dir/'analysis'
    lease=shared.acquire_lease(analysis/'statistical_verification.lock')
    result=shared.read(analysis/'followup_statistics.json');assert result['status']=='complete' and len(result['results'])==20
    design=shared.read(a.run_dir/'frozen_protocol.json')
    cfg=shared.read(Path(__file__).resolve().parents[1]/'configs/submission_evidence_analysis.json')
    svm={};neural={}
    with (analysis/'matched_classwise_scores.csv').open(newline='') as stream:class_export=list(csv.DictReader(stream))
    class_lookup={(r['dataset'],r['model'],r['target'],r['seed'],int(r['class'])):r for r in class_export}
    assert len(class_lookup)==len(class_export)==10850
    checked_class_keys=set()
    def check_classes(dataset,model,target,seed,y,scores):
        predicted=scores.argmax(axis=1);labels=np.arange(scores.shape[1])
        precision,recall,fscore,support=precision_recall_fscore_support(y,predicted,labels=labels,zero_division=0)
        for label in labels:
            key=(dataset,model,target,str(seed),int(label));r=class_lookup[key];checked_class_keys.add(key)
            for value,column in [(100*precision[label],'precision_pct'),(100*recall[label],'recall_pct'),(100*fscore[label],'f1_pct')]:
                assert np.isclose(value,float(r[column]),atol=1e-10,rtol=0)
            assert int(support[label])==int(r['true_windows'])
            assert int(np.count_nonzero(predicted==label))==int(r['predicted_windows'])
    for path in (a.run_dir/'matched_svm/production').glob('*/target_*/result.json'):
        row=shared.read(path)
        with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as z:
            y,scores=z['target_labels'],z['target_scores']
            svm[(row['dataset'],row['target'])]=100*balanced_accuracy_score(y,scores.argmax(axis=1))
            check_classes(row['dataset'],'DE+SVM',row['target'],'',y,scores)
            trial_y=[];trial_scores=[]
            for trial in np.unique(z['target_trial']):
                mask=z['target_trial']==trial;truth=np.unique(y[mask]);assert len(truth)==1
                trial_y.append(truth[0]);trial_scores.append(scores[mask].mean(axis=0))
            trial_predicted=np.asarray(trial_scores).argmax(axis=1)
            for value,key in [(balanced_accuracy_score(trial_y,trial_predicted),'balanced_accuracy'),
                              (accuracy_score(trial_y,trial_predicted),'accuracy'),
                              (f1_score(trial_y,trial_predicted,average='macro',zero_division=0),'macro_f1')]:
                assert np.isclose(value,row['trial'][key],atol=1e-12,rtol=0)
    assert len(svm)==322
    for dataset in design['datasets']:
        targets=sorted(t for d,t in svm if d==dataset)
        for model in design['models']:
            for seed in design['training_seeds']:
                for target in targets:
                    folder=(a.run_dir/'neural/production'/model/f'seed_{seed}'/f'target_{target}' if dataset=='FACED-binary'
                            else shared.neural_reference(a.runs_root,model,dataset,seed,target))
                    with np.load(folder/'scoring_arrays.npz',allow_pickle=False) as z:
                        y,prob,val=z['target_labels'],z['target_probabilities'],z['validation_bacc']
                        pooled=int(np.argmax(val.mean(axis=0)))
                        selected_pct=100*balanced_accuracy_score(y,prob[pooled].argmax(axis=1))
                        check_classes(dataset,model,target,seed,y,prob[pooled])
                        data={'neural_pooled_minus_svm':selected_pct-svm[(dataset,target)],'selected_bacc_pct':selected_pct}
                        if dataset=='FACED-binary':
                            final=balanced_accuracy_score(y,prob[-1].argmax(axis=1));chosen=np.argmax(val,axis=1)
                            other=np.mean([100*(balanced_accuracy_score(y,prob[k].argmax(axis=1))-final) for k in chosen])
                            own=np.mean([100*(val[i,k]-val[i,-1]) for i,k in enumerate(chosen)])
                            pg=100*(selected_pct/100-final)
                            data.update(individual_target_change=other,selection_to_transfer_shortfall=own-other,
                                        pooled_target_change=pg,pooled_minus_individual=pg-other)
                        neural[(dataset,model,target,seed)]=data
    assert len(neural)==1932
    assert len(checked_class_keys)==10850
    with (analysis/'benchmark_summary.csv').open(newline='') as stream:benchmarks=list(csv.DictReader(stream))
    assert len(benchmarks)==18 and len({(r['dataset'],r['model']) for r in benchmarks})==18
    for row in benchmarks:
        dataset,model=row['dataset'],row['model'];targets=sorted(t for d,t in svm if d==dataset)
        values=np.array([svm[(dataset,t)] if model=='DE+SVM' else
                         np.mean([neural[(dataset,model,t,s)]['selected_bacc_pct'] for s in design['training_seeds']]) for t in targets])
        rng=np.random.default_rng(cfg['bootstrap_seed'])
        ci=np.quantile(values[rng.integers(len(values),size=(cfg['bootstrap_replicates'],len(values)))].mean(axis=1),[.025,.975])
        assert int(row['n'])==len(targets) and np.isclose(values.mean(),float(row['bacc_pct']),atol=1e-10,rtol=0)
        np.testing.assert_allclose(ci,[float(row['ci_low']),float(row['ci_high'])],atol=1e-10,rtol=0)
    matrices=np.load(analysis/'contrast_vectors.npz',allow_pickle=False)
    means=[];ranks=[];uppers=[]
    for row in result['results']:
        dataset,model,contrast=row['dataset'],row['model'],row['contrast']
        targets=sorted(t for d,t in svm if d==dataset)
        recovered=np.array([[neural[(dataset,model,t,s)][contrast] for s in design['training_seeds']] for t in targets])
        frozen=matrices[row['vector_key']]
        np.testing.assert_allclose(recovered,frozen,atol=1e-10,rtol=0)
        x=frozen.mean(axis=1);summary=row['summary']
        assert np.isclose(x.mean(),summary['mean_pp'],atol=1e-12,rtol=0)
        rng=np.random.default_rng(cfg['bootstrap_seed'])
        ci=np.quantile(x[rng.integers(len(x),size=(cfg['bootstrap_replicates'],len(x)))].mean(axis=1),[.025,.975])
        np.testing.assert_allclose(ci,summary['descriptive_ci95_pp'],atol=1e-12,rtol=0)
        identity=f'followup|{model}|{dataset}|{contrast}'
        mean,interval=direct_mean_test(x,cfg,identity);rank=exact_pratt(x)
        assert np.isclose(mean,summary['mean_signflip']['p'],atol=1e-15,rtol=0)
        assert np.isclose(rank,summary['signed_rank']['p'],atol=1e-15,rtol=0)
        np.testing.assert_allclose(interval,summary['mean_signflip']['mc_tail_interval'],atol=1e-15,rtol=0)
        means.append(mean);ranks.append(rank);uppers.append(interval[1])
    for row,mean,rank,upper in zip(result['results'],direct_holm(means),direct_holm(ranks),direct_holm(uppers)):
        assert np.isclose(mean,row['adjusted_mean_p'],atol=1e-15,rtol=0)
        assert np.isclose(rank,row['adjusted_rank_p'],atol=1e-15,rtol=0)
        assert np.isclose(upper,row['adjusted_mean_mc_upper'],atol=1e-15,rtol=0)
        assert row['conditional_concordance']==bool(mean<.05 and rank<.05 and upper<.05)
    with (analysis/'exploratory_joint163_sensitivity.csv').open(newline='') as stream:joint=list(csv.DictReader(stream))
    assert len(joint)==163 and len({r['id'] for r in joint})==163
    for row,mean,rank in zip(joint,direct_holm([float(r['mean_raw_p']) for r in joint]),direct_holm([float(r['rank_raw_p']) for r in joint])):
        assert np.isclose(mean,float(row['exploratory_joint163_mean_p']),atol=1e-15,rtol=0)
        assert np.isclose(rank,float(row['exploratory_joint163_rank_p']),atol=1e-15,rtol=0)
    sources=[('original63','evidence_synthesis_20261030/analysis/evidence_statistics.json','comparisons',63),
             ('partition20','split_sensitivity_20261030/analysis/split_sensitivity_statistics.json','results',20),
             ('temporal4','temporal_transfer_extension_20261030/analysis/temporal_transfer_statistics.json','results',4),
             ('selector_horizon56','selection_horizon_extension_20261031/analysis/extension_statistics.json','results',56)]
    checked_original=0
    for family,name,key,expected in sources:
        rows=shared.read(a.runs_root/name)[key];assert len(rows)==expected
        listed=[r for r in joint if r['family']==family];assert len(listed)==expected
        for original,exported in zip(rows,listed):
            s=original.get('summary',original)
            mean_key=next(k for k in ['p_mean_holm_global','adjusted_mean_p','global_adjusted_mean_p'] if k in original)
            rank_key=next(k for k in ['p_rank_holm_global','adjusted_rank_p','global_adjusted_rank_p'] if k in original)
            assert original['model']==exported['model']
            assert original.get('dataset',original.get('setting'))==exported['dataset']
            assert original.get('contrast',original.get('outcome'))==exported['contrast']
            for value,column in [(s['mean_signflip']['p'],'mean_raw_p'),(s['signed_rank']['p'],'rank_raw_p'),
                                 (original[mean_key],'original_mean_adjusted_p'),(original[rank_key],'original_rank_adjusted_p')]:
                assert np.isclose(value,float(exported[column]),atol=1e-15,rtol=0)
            checked_original+=1
    assert checked_original==143
    with (analysis/'native_subject_scores.csv').open(newline='') as stream:native_export=list(csv.DictReader(stream))
    native_lookup={(r['task'],int(r['bands']),int(r['fold']),r['normalization_prior'],int(r['target'])):r for r in native_export}
    assert len(native_lookup)==len(native_export)==552
    checked_native=set()
    for path in (a.run_dir/'native/production').glob('*/fold_*/result.json'):
        r=shared.read(path)
        with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as z:
            accuracies=100*np.mean(z['predictions'][r['selected_index']]==z['labels'],axis=1)
        for target,accuracy in zip(r['held_out_subjects'],accuracies):
            key=(r['task'],r['bands'],r['fold'],r['normalization_prior'],target)
            exported=native_lookup[key];checked_native.add(key)
            assert np.isclose(accuracy,float(exported['accuracy_pct']),atol=1e-12,rtol=0)
            assert np.isclose(r['selected_C'],float(exported['selected_C']),atol=1e-15,rtol=0)
    assert len(checked_native)==552
    with (analysis/'native_reconstruction_summary.csv').open(newline='') as stream:native_summary=list(csv.DictReader(stream))
    assert len(native_summary)==8
    for r in native_summary:
        task,bands,prior=r['task'],int(r['bands']),r['final_fold_prior'];values=[];fold_means=[]
        for fold in range(10):
            label=prior if fold==9 else 'paper_fold_exclusion'
            targets=range(fold*12,(fold+1)*12 if fold<9 else 123)
            current=[float(native_lookup[(task,bands,fold,label,t)]['accuracy_pct']) for t in targets]
            assert len(current)==(15 if fold==9 else 12);values.extend(current);fold_means.append(np.mean(current))
        x=np.array(values);assert len(x)==123
        original=100*np.load(a.run_dir/f'native/prepared/original_scores_{task}.npy')
        for value,column in [(x.mean(),'accuracy_pct'),(x.std(ddof=1)/np.sqrt(123),'sem_pp'),
                             (np.mean(fold_means),'mean_fold_accuracy_pct'),
                             (np.mean(abs(x-original)),'mean_absolute_difference_from_released_outputs_pp')]:
            assert np.isclose(value,float(r[column]),atol=1e-10,rtol=0)
    for filename,count in [('neural_DGCNN.json',369),('neural_CDCN.json',369),('svm_matched.json',322),('svm_native.json',44)]:
        proof=shared.read(a.run_dir/'verification'/filename);assert proof['status']=='pass' and proof['records']==count
    proof={'status':'pass','independently_recomputed_contrasts':20,'neural_records':738,'svm_records':322,
           'native_records':44,'all_1932_reference_scores_recomputed':True,'bootstrap_intervals_checked':20,
           'exact_pratt_generating_function_checks':20,'mean_randomization_checks':20,
           'new_holm20_adjustments_checked':True,'exploratory_joint163_checked':True,
           'independent_classwise_checks':10850,'native_subject_score_checks':552,
           'absolute_benchmark_intervals_checked':18,'native_aggregate_checks':8,
           'original_families_retained':[63,20,4,56],'original_raw_and_adjusted_values_preserved':checked_original,
           'verification_code_sha256':shared.sha(Path(__file__)),
           'verified_statistics_sha256':shared.sha(analysis/'followup_statistics.json'),
           'verified_input_sha256':{name:shared.sha(analysis/name) for name in ['followup_statistics.json',
               'contrast_vectors.npz','contrast_vector_targets.json','benchmark_summary.csv',
               'matched_classwise_scores.csv','native_subject_scores.csv','native_reconstruction_summary.csv',
               'exploratory_joint163_sensitivity.csv']}}
    base.write_json(analysis/'independent_verification.json',proof)
    base.write_json(a.run_dir/'analysis_status.json',{'status':'complete_independently_verified','analysis_dir':str(analysis)})
    print('FOLLOWUP_INDEPENDENT_VERIFICATION_PASS',flush=True)


if __name__=='__main__':main()
