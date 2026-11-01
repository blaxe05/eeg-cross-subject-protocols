"""Freeze matched model jobs from declared folds and prescribed update budgets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import matched_models_common as common
import faced_svm_common as original


DATASETS=['SEED','SEED-IV','FACED','FACED-binary','DREAMER-valence','DREAMER-arousal']
ARMS={'DGCNN':['source'],'CDCN':['source'],'EEGNet':['source'],'TSception':['source'],
      'RGNN':['source','target','proxy'],'NSAL-DGAT':['target','proxy'],'MS-MDA':['target','proxy']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['project-root','input-dir','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-runs',type=Path,help='Optional original run metadata; otherwise use fixed protocol hyperparameters')
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    args=parser.parse_args();original.setup(args.project_root)
    config={'experiment':'common_input_published_models_extension','version':1,'datasets':args.datasets,
        'training_seeds':[2031,2032,2033],'source_batch_size':256,'selection':'equal_subject_source_BAcc',
        'ties':'earliest_candidate','normalization_fit':'training_anchors_only',
        'inputs':'aligned_stateless_one_second_raw_and_local_DE','faced_sensor_scope':'30_scalp_channels_no_HEOR_HEOL',
        'legacy_config':common.read(ROOT/'configs/architecture_replication.json'),
        'new_planned_family_sizes':[144,60,18],'original_family_sizes':[63,20,4,56,20],
        'new_families_are_exploratory':True,'unavailable_model_slots':'missing_not_fabricated; retain planned correction denominators',
        'MGGL_status':'reference_required','source_model_arms':ARMS,'fixed_update_budgets':{},
        'implementation_notes':{
            'EEGNet':'author-architecture PyTorch port; grouped depthwise filters, Keras-compatible flatten and max norms',
            'TSception':'pinned author network; retain bilateral scalp pairs in left/right order',
            'RGNN':'author-equivalent dense signed graph; three/four-class reference soft labels, binary one-hot, explicit uniform nine-class extension',
            'NSAL-DGAT':'pinned author encoder/objective with documented input-axis, loss-logit, mode/bank and zero-degree repairs; 1000-update GRL warmup; native boost endpoint mapped to relative progress',
            'MS-MDA':'author-equivalent shared/domain networks and marginal/discrepancy loss; per-source batches, ramped marginal/discrepancy coefficients',
            'DGCNN_CDCN':'same fixed backbones/optimizers, rerun on common normalized local DE'}}
    jobs=[];reserved=[]
    for dataset in args.datasets:
        bank=common.InputBank(args.input_dir,dataset);config['fixed_update_budgets'][dataset]={}
        for target in sorted(set(bank.subject),key=lambda s:int(s) if str(s).isdigit() else str(s)):
            frozen=original.fold(args.project_root,dataset,str(target))
            fold={k:frozen[k] for k in ['train','validation','target']}
            assert not set(fold['train'])&set(fold['validation']) and target not in fold['train']+fold['validation']
            assert set(fold['train']+fold['validation']+[target])==set(bank.subject)
            reference=(original.neural_reference(args.reference_runs,'DGCNN',dataset,2031,str(target))/'result.json'
                       if args.reference_runs else None)
            if reference is not None:
                record=common.read(reference);steps=record['candidate_steps']
                assert steps[-1]==record['updates'] and len(steps) in [8,40]
                reference_hash=common.sha(reference)
            else:
                budgets=common.read(ROOT/'configs/matched_model_budgets.json')
                total=budgets['optimizer_updates'][dataset][str(target)];count=budgets['candidate_counts'][dataset]
                assert total%count==0
                steps=[total//count*i for i in range(1,count+1)];reference_hash=None
            config['fixed_update_budgets'][dataset][str(target)]=steps
            for seed in config['training_seeds']:
                for model,arms in ARMS.items():
                    for arm in arms:
                        group=f'{model}__{dataset}__{seed}__{arm}'
                        jobs.append({'id':f'{group}__{target}','group':group,'model':model,'dataset':dataset,'seed':seed,'arm':arm,
                                     'fold':fold,'candidate_steps':steps,'budget_reference_sha256':reference_hash})
                reserved.append({'model':'MGGL','dataset':dataset,'seed':seed,'target':str(target),'state':'reference_required'})
            jobs.append({'id':f'SVM__{dataset}__0__source__{target}','group':f'SVM__{dataset}__0__source',
                         'model':'SVM','dataset':dataset,'seed':0,'arm':'source','fold':fold,'candidate_steps':[]})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    common.write(args.output_dir/'protocol.json',config)
    common.write(args.output_dir/'jobs.json',{'jobs':jobs,'reserved_models':reserved,
        'supported_neural_paths':sum(r['model']!='SVM' for r in jobs),'svm_outer_fits':sum(r['model']=='SVM' for r in jobs),
        'svm_candidate_fits':12*sum(r['model']=='SVM' for r in jobs),'MGGL_paths_pending_reference':len(reserved)})
    common.write(args.output_dir/'progress.json',{'state':'prepared_pending_gates','completed':0,'planned_supported_records':len(jobs),
        'reserved_MGGL_paths':len(reserved),'new_experimental_results':0})
    print(json.dumps({'supported_neural_paths':sum(r['model']!='SVM' for r in jobs),'svm_outer_fits':sum(r['model']=='SVM' for r in jobs),
                      'MGGL_pending_reference':len(reserved),'groups':len(set(r['group'] for r in jobs))}))


if __name__=='__main__':main()
