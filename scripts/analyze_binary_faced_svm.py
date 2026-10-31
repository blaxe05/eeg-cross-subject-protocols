"""Summarize the complete binary FACED/SVM follow-up and its fixed test family."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
import faced_svm_common as shared
import run_controlled_revision as base
from analyze_submission_evidence import paired_summary,holm


def write_csv(path,rows):
    assert rows
    with path.open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def interval(x):
    x=np.asarray(x,dtype=float);rng=np.random.default_rng(20261029)
    return np.quantile(x[rng.integers(len(x),size=(20000,len(x)))].mean(axis=1),[.025,.975]).tolist()


def classwise_rows(dataset,model,target,seed,y,scores):
    predicted=scores.argmax(axis=1);rows=[]
    for label in range(scores.shape[1]):
        truth=int(np.count_nonzero(y==label));assigned=int(np.count_nonzero(predicted==label))
        correct=int(np.count_nonzero((y==label)&(predicted==label)))
        assert truth>0
        rows.append({'dataset':dataset,'model':model,'target':target,'seed':seed,'class':label,
                     'recall_pct':100*correct/truth,'precision_pct':100*correct/assigned if assigned else 0.,
                     'f1_pct':200*correct/(truth+assigned),'true_windows':truth,'predicted_windows':assigned})
    return rows


def native_candidate_summary(predictions,labels):
    if predictions.ndim!=3 or labels.ndim!=1 or predictions.shape[-1]!=len(labels) or not all(predictions.shape):
        raise ValueError('Expected candidate-by-subject-by-window predictions and common window labels')
    correct=predictions==labels
    # Select from the released global window-accuracy criterion directly.
    # Averaging subject proportions first can break an exact C tie through roundoff.
    choice=int(np.argmax(correct.mean(axis=(1,2))))
    return correct.mean(axis=2),choice


def complete_paths(root):
    neural=sorted((root/'neural/production').glob('*/seed_*/target_*/result.json'))
    svm=sorted((root/'matched_svm/production').glob('*/target_*/result.json'))
    native=sorted((root/'native/production').glob('*/fold_*/result.json'))
    counts={'neural':len(neural),'matched_svm':len(svm),'native':len(native)}
    if counts!={'neural':738,'matched_svm':322,'native':44}:
        base.write_json(root/'analysis_status.json',{'status':'waiting_for_complete_records','actual':counts,
                        'expected':{'neural':738,'matched_svm':322,'native':44}})
        raise AssertionError(f'Incomplete follow-up: {counts}')
    return neural,svm,native


def original_raw_tests(runs):
    result=[]
    sources=[('original63','evidence_synthesis_20261030/analysis/evidence_statistics.json','comparisons'),
             ('partition20','split_sensitivity_20261030/analysis/split_sensitivity_statistics.json','results'),
             ('temporal4','temporal_transfer_extension_20261030/analysis/temporal_transfer_statistics.json','results'),
             ('selector_horizon56','selection_horizon_extension_20261031/analysis/extension_statistics.json','results')]
    for family,name,key in sources:
        for index,row in enumerate(shared.read(runs/name)[key]):
            summary=row.get('summary',row)
            result.append({'family':family,'id':f'{family}_{index:03d}','dataset':row.get('dataset',row.get('setting')),
                           'model':row['model'],'contrast':row.get('contrast',row.get('outcome')),
                           'mean_raw_p':summary['mean_signflip']['p'],'rank_raw_p':summary['signed_rank']['p'],
                           'original_mean_adjusted_p':row.get('p_mean_holm_global',row.get('adjusted_mean_p',row.get('global_adjusted_mean_p'))),
                           'original_rank_adjusted_p':row.get('p_rank_holm_global',row.get('adjusted_rank_p',row.get('global_adjusted_rank_p')))})
    assert len(result)==143
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run-dir','runs-root','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    lease=shared.acquire_lease(a.output_dir/'analysis.lock')
    neural_paths,svm_paths,native_paths=complete_paths(a.run_dir)
    design=shared.read(a.run_dir/'frozen_protocol.json')
    cfg=shared.read(Path(__file__).resolve().parents[1]/'configs/submission_evidence_analysis.json')
    cfg['expected_hypotheses']=20
    svm={};svm_rows=[];class_rows=[]
    for path in svm_paths:
        row=shared.read(path);assert row['status']=='complete' and not row['target_labels_used_for_selection']
        assert row['fold']['target']==row['target'] and row['target'] not in row['fold']['train']+row['fold']['validation']
        with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as z:
            y,scores=z['target_labels'],z['target_scores']
            result=base.metrics(y,scores,row['classes'])
            assert np.isclose(result['balanced_accuracy'],row['window']['balanced_accuracy'],atol=1e-12)
            choice=int(z['validation_bacc'].mean(axis=0).argmax())
            assert choice==row['selected_index'] and design['svm_c_grid'][choice]==row['selected_C']
            svm[(row['dataset'],row['target'])]=(row,y.copy(),z['target_trial'].copy(),z['source_rows'].copy())
            class_rows.extend(classwise_rows(row['dataset'],'DE+SVM',row['target'],'',y,scores))
        svm_rows.append({'dataset':row['dataset'],'target':row['target'],'selected_C':row['selected_C'],
                         'window_bacc_pct':100*result['balanced_accuracy'],'window_accuracy_pct':100*result['accuracy'],
                         'window_macro_f1_pct':100*result['macro_f1'],'training_examples':row['training_examples'],
                         'trial_bacc_pct':100*row['trial']['balanced_accuracy'],
                         'trial_accuracy_pct':100*row['trial']['accuracy'],
                         'trial_macro_f1_pct':100*row['trial']['macro_f1'],
                         'trial_aggregation':'mean_decision_scores'})
    lookup={};binary_rows=[];reference_rows=[]
    for dataset,n in design['subject_counts'].items():
        targets=sorted(t for d,t in svm if d==dataset);assert len(targets)==n
        for model in design['models']:
            for seed in design['training_seeds']:
                for target in targets:
                    folder=(a.run_dir/'neural/production'/model/f'seed_{seed}'/f'target_{target}' if dataset=='FACED-binary'
                            else shared.neural_reference(a.runs_root,model,dataset,seed,target))
                    row=shared.read(folder/'result.json');assert row['status']=='complete'
                    with np.load(folder/'scoring_arrays.npz',allow_pickle=False) as z:
                        y,prob,validation=z['target_labels'],z['target_probabilities'],z['validation_bacc']
                        assert np.array_equal(y,svm[(dataset,target)][1])
                        assert np.array_equal(z['target_trial'],svm[(dataset,target)][2])
                        assert np.array_equal(z['source_rows'],svm[(dataset,target)][3])
                        assert np.isfinite(prob).all() and np.allclose(prob.sum(axis=2),1,atol=1e-5)
                        chosen=int(validation.mean(axis=0).argmax());individual=validation.argmax(axis=1)
                        curve=np.array([base.metrics(y,s,prob.shape[2])['balanced_accuracy'] for s in prob])
                        pooled=100*(curve[chosen]-curve[-1])
                        different=float(np.mean(100*(curve[individual]-curve[-1])))
                        own=float(np.mean([100*(validation[i,j]-validation[i,-1]) for i,j in enumerate(individual)]))
                        final=base.metrics(y,prob[-1],prob.shape[2]);selected=base.metrics(y,prob[chosen],prob.shape[2])
                        values={'individual_target_change':different,'selection_to_transfer_shortfall':own-different,
                                'pooled_target_change':pooled,'pooled_minus_individual':pooled-different,
                                'selection_gain':own,'selected_bacc_pct':100*selected['balanced_accuracy'],
                                'final_bacc_pct':100*final['balanced_accuracy'],
                                'selected_accuracy_pct':100*selected['accuracy'],'selected_f1_pct':100*selected['macro_f1'],
                                'final_f1_pct':100*final['macro_f1'],
                                'final_predicted_classes':len(np.unique(prob[-1].argmax(axis=1))),
                                'selected_predicted_classes':len(np.unique(prob[chosen].argmax(axis=1))),
                                'final_dominant_prediction_share_pct':100*np.bincount(prob[-1].argmax(axis=1)).max()/len(y),
                                'selected_dominant_prediction_share_pct':100*np.bincount(prob[chosen].argmax(axis=1)).max()/len(y)}
                        class_rows.extend(classwise_rows(dataset,model,target,seed,y,prob[chosen]))
                    lookup[(dataset,model,target,seed)]=values
                    reference_rows.append(dict(dataset=dataset,model=model,target=target,seed=seed,**values))
                    if dataset=='FACED-binary':binary_rows.append(reference_rows[-1])
    output=[];matrices={};vector_labels=[]
    for model in design['models']:
        targets=sorted(t for d,t in svm if d=='FACED-binary')
        for endpoint in ['individual_target_change','selection_to_transfer_shortfall','pooled_target_change','pooled_minus_individual']:
            matrix=np.array([[lookup[('FACED-binary',model,t,s)][endpoint] for s in design['training_seeds']] for t in targets])
            summary=paired_summary(matrix,cfg,f'followup|{model}|FACED-binary|{endpoint}')
            name=f'row_{len(output):02d}';matrices[name]=matrix
            output.append({'model':model,'dataset':'FACED-binary','contrast':endpoint,'vector_key':name,'summary':summary})
            vector_labels.append({'vector_key':name,'targets':targets})
    for dataset,n in design['subject_counts'].items():
        targets=sorted(t for d,t in svm if d==dataset)
        for model in design['models']:
            matrix=np.array([[lookup[(dataset,model,t,s)]['selected_bacc_pct']-100*svm[(dataset,t)][0]['window']['balanced_accuracy']
                              for s in design['training_seeds']] for t in targets])
            summary=paired_summary(matrix,cfg,f'followup|{model}|{dataset}|neural_pooled_minus_svm')
            name=f'row_{len(output):02d}';matrices[name]=matrix
            output.append({'model':model,'dataset':dataset,'contrast':'neural_pooled_minus_svm','vector_key':name,'summary':summary})
            vector_labels.append({'vector_key':name,'targets':targets})
    assert len(output)==20
    m=holm([r['summary']['mean_signflip']['p'] for r in output]);q=holm([r['summary']['signed_rank']['p'] for r in output])
    upper=holm([r['summary']['mean_signflip']['mc_tail_interval'][1] for r in output])
    for i,row in enumerate(output):
        row.update(adjusted_mean_p=float(m[i]),adjusted_rank_p=float(q[i]),adjusted_mean_mc_upper=float(upper[i]),
                   conditional_concordance=bool(m[i]<.05 and q[i]<.05 and upper[i]<.05))
    benchmark=[]
    for dataset,n in design['subject_counts'].items():
        targets=sorted(t for d,t in svm if d==dataset)
        for model in ['DGCNN','CDCN','DE+SVM']:
            x=np.array([100*svm[(dataset,t)][0]['window']['balanced_accuracy'] if model=='DE+SVM' else
                        np.mean([lookup[(dataset,model,t,s)]['selected_bacc_pct'] for s in design['training_seeds']]) for t in targets])
            bounds=interval(x)
            benchmark.append({'dataset':dataset,'model':model,'n':n,'bacc_pct':float(x.mean()),
                              'ci_low':bounds[0],'ci_high':bounds[1]})
    native={};native_candidates=[];native_subjects=[]
    for path in native_paths:
        r=shared.read(path);assert r['status']=='complete' and r['target_labels_used_for_C_selection']
        key=(r['task'],r['bands'],r['fold'],r['normalization_prior']);assert key not in native
        with np.load(path.parent/'scoring_arrays.npz',allow_pickle=False) as z:
            pred,y=z['predictions'],z['labels'];scores,choice=native_candidate_summary(pred,y)
            assert choice==r['selected_index']
            assert np.allclose(100*scores[choice],r['subject_accuracy_pct'],atol=1e-12)
        native[key]=(r,scores[choice])
        for target,accuracy in zip(r['held_out_subjects'],scores[choice]):
            native_subjects.append({'task':r['task'],'bands':r['bands'],'fold':r['fold'],
                                    'normalization_prior':r['normalization_prior'],'target':int(target),
                                    'selected_C':r['selected_C'],'accuracy_pct':100*float(accuracy)})
        for index,c in enumerate(r['candidates']):
            native_candidates.append({'task':r['task'],'bands':r['bands'],'fold':r['fold'],'normalization_prior':r['normalization_prior'],
                                      'C_index':index,'C':c['C'],'selected':index==choice,'accuracy_pct':100*c['held_out_accuracy'],
                                      'n_iter':c['n_iter'],'warnings':' | '.join(c['warnings'])})
    native_summary=[]
    for task in ['binary','nine']:
        for bands in [5,4]:
            for prior in ['paper_fold_exclusion','released_11_subject_exclusion']:
                vectors=[];fold_means=[]
                for f in range(10):
                    label=prior if f==9 else 'paper_fold_exclusion'
                    r,v=native[(task,bands,f,label)];vectors.extend(v);fold_means.append(float(v.mean()))
                x=100*np.array(vectors);assert len(x)==123
                original=np.load(a.run_dir/f'native/prepared/original_scores_{task}.npy')*100
                native_summary.append({'task':task,'bands':bands,'final_fold_prior':prior,'n':123,
                                       'accuracy_pct':float(x.mean()),'sem_pp':float(x.std(ddof=1)/np.sqrt(123)),
                                       'mean_fold_accuracy_pct':float(100*np.mean(fold_means)),
                                       'mean_absolute_difference_from_released_outputs_pp':float(np.mean(abs(x-original)))})
    old=original_raw_tests(a.runs_root)
    for i,r in enumerate(output):
        old.append({'family':'binary_baseline20','id':f'binary_baseline20_{i:03d}','dataset':r['dataset'],'model':r['model'],
                    'contrast':r['contrast'],'mean_raw_p':r['summary']['mean_signflip']['p'],'rank_raw_p':r['summary']['signed_rank']['p'],
                    'original_mean_adjusted_p':r['adjusted_mean_p'],'original_rank_adjusted_p':r['adjusted_rank_p']})
    assert len(old)==163
    for row in old:
        for key in ['mean_raw_p','rank_raw_p','original_mean_adjusted_p','original_rank_adjusted_p']:
            assert row[key] is not None and np.isfinite(row[key]) and 0<=row[key]<=1,(row['id'],key)
    joint_m=holm([r['mean_raw_p'] for r in old]);joint_r=holm([r['rank_raw_p'] for r in old])
    for i,r in enumerate(old):r.update(exploratory_joint163_mean_p=float(joint_m[i]),exploratory_joint163_rank_p=float(joint_r[i]))
    shared.save_arrays(a.output_dir/'contrast_vectors.npz',**matrices)
    base.write_json(a.output_dir/'contrast_vector_targets.json',vector_labels)
    write_csv(a.output_dir/'binary_trajectory_summary.csv',binary_rows)
    write_csv(a.output_dir/'neural_reference_summary.csv',reference_rows)
    write_csv(a.output_dir/'matched_svm_subject_scores.csv',svm_rows)
    assert len(class_rows)==10850 and len(native_subjects)==552
    write_csv(a.output_dir/'matched_classwise_scores.csv',class_rows)
    write_csv(a.output_dir/'benchmark_summary.csv',benchmark)
    write_csv(a.output_dir/'native_reconstruction_summary.csv',native_summary)
    write_csv(a.output_dir/'native_candidate_diagnostics.csv',native_candidates)
    write_csv(a.output_dir/'native_subject_scores.csv',native_subjects)
    write_csv(a.output_dir/'exploratory_joint163_sensitivity.csv',old)
    tests=[]
    for row in output:
        s=row['summary'];tests.append({'dataset':row['dataset'],'model':row['model'],'contrast':row['contrast'],
            'n':s['n_target_people'],'mean_pp':s['mean_pp'],'ci_low':s['descriptive_ci95_pp'][0],'ci_high':s['descriptive_ci95_pp'][1],
            'mean_raw_p':s['mean_signflip']['p'],'rank_raw_p':s['signed_rank']['p'],'mean_holm20_p':row['adjusted_mean_p'],
            'rank_holm20_p':row['adjusted_rank_p'],'mean_mc_upper_holm20':row['adjusted_mean_mc_upper'],
            'conditional_concordance':row['conditional_concordance']})
    write_csv(a.output_dir/'followup_tests.csv',tests)
    report={'status':'complete','family_size':20,'records':{'neural':738,'matched_svm':322,'native':44},
            'neural_feature_trajectories_total':1932,'new_hypothesis_family':'binary_and_baseline_followup',
            'original_families_preserved':[63,20,4,56],'total_distinct_comparisons':163,'results':output,
            'native_exact_reproduction_certified':False,'native_reconstruction_variants':8,
            'protocol_sha256':shared.sha(a.run_dir/'frozen_protocol.json'),
            'analysis_code_sha256':shared.sha(Path(__file__)),
            'paired_engine_code_sha256':shared.sha(Path(__file__).with_name('analyze_submission_evidence.py')),
            'scope':'Follow-up exploratory comparisons; fitted source populations overlap; SVM is deterministic, neural seeds are averaged within targets.'}
    base.write_json(a.output_dir/'followup_statistics.json',report)
    base.write_json(a.run_dir/'analysis_status.json',{'status':'analysis_complete_independent_verification_pending','analysis_dir':str(a.output_dir)})
    print('FOLLOWUP_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
