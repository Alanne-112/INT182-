"""Train on human-reviewed segments, evaluate only on explicitly held-out videos."""
import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter
from robust_counter import FEATURES,SCHEMA,feature_vector

def discover_dataset(runs_dir,test_videos=None):
    """Pick latest fully reviewed run per video; never change stored train/test roles."""
    grouped={}
    ignored=[]
    for path in sorted(Path(runs_dir).rglob('attempts_to_label.csv')):
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        if not rows or not any(r.get('label') in ('pass','fail') for r in rows):
            ignored.append(dict(path=str(path.resolve()),reason='no pass/fail labels'));continue
        if any(r.get('label') not in ('pass','fail','unassessable') for r in rows):
            ignored.append(dict(path=str(path.resolve()),reason='review not complete'));continue
        if any(r.get('schema')!=SCHEMA for r in rows):
            ignored.append(dict(path=str(path.resolve()),reason='old schema'));continue
        videos={r.get('video_id','') for r in rows}
        if len(videos)!=1 or '' in videos:raise ValueError(f'Expected one video per run: {path}')
        eligible=[r for r in rows if r['label'] in ('pass','fail') and r.get('segment_status') not in ('tracking_lost','stopped','video_end')]
        if not eligible:
            ignored.append(dict(path=str(path.resolve()),reason='no usable segments'));continue
        splits={r.get('split') for r in eligible}
        if len(splits)!=1 or not splits<={'train','test'}:raise ValueError(f'Invalid or mixed split in {path}')
        video=next(iter(videos));split=next(iter(splits))
        grouped.setdefault(video,[]).append(dict(path=path,split=split,eligible=len(eligible),
            labels=dict(Counter(r['label'] for r in eligible))))
    selected=[]
    for video,candidates in sorted(grouped.items()):
        if len({c['split'] for c in candidates})>1:
            raise ValueError(f'Conflicting train/test assignments for video {video}. Resolve its reviewed CSVs first.')
        chosen=max(candidates,key=lambda c:(c['path'].stat().st_mtime_ns,str(c['path'])))
        for duplicate in candidates:
            if duplicate is not chosen:ignored.append(dict(path=str(duplicate['path'].resolve()),reason=f'duplicate of {video}; latest complete review selected'))
        if chosen['split']=='test' and test_videos is not None and video not in test_videos:
            ignored.append(dict(path=str(chosen['path'].resolve()),reason='test video not selected (not moved to train)'));continue
        selected.append(dict(video_id=video,split=chosen['split'],segments=chosen['eligible'],
            labels=chosen['labels'],path=str(chosen['path'].resolve())))
    if test_videos is not None:
        missing=set(test_videos)-{r['video_id'] for r in selected if r['split']=='test'}
        if missing:raise ValueError('Requested test videos need complete test labels first: '+', '.join(sorted(missing)))
    return [r['path'] for r in selected],dict(selected=selected,ignored=ignored)

def next_model_folder():
    base=Path(__file__).parent/'models'
    candidate=base/'random_forest'
    version=1
    while candidate.exists():
        version+=1;candidate=base/f'random_forest_v{version}'
    return candidate

def load_dataset(paths):
    rows=[]
    seen=set()
    skipped=0
    for path in paths:
        with Path(path).open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                if row.get('label') not in ('pass','fail'):
                    skipped+=1;continue
                if row.get('schema')!=SCHEMA:raise ValueError(f'Old feature schema: {path}; rerun counter v2')
                if row.get('segment_status') in ('tracking_lost','stopped','video_end'):
                    skipped+=1;continue
                if row.get('split') not in ('train','test'):raise ValueError('Every labeled row needs split=train or test')
                key=(row['video_id'],row['attempt_id'])
                if key in seen:raise ValueError(f'Duplicate segment {key}; use one reviewed run per video')
                seen.add(key)
                vector=feature_vector(row)
                if any(not math.isfinite(v) for v in vector[:3]):raise ValueError(f'Missing knee/time feature: {key}')
                rows.append(row)
    train=[r for r in rows if r['split']=='train']
    test=[r for r in rows if r['split']=='test']
    if not train or not test:raise ValueError('Need both training data and a separate held-out test video')
    train_videos={r['video_id'] for r in train};test_videos={r['video_id'] for r in test}
    if train_videos & test_videos:raise ValueError('Video leakage: one video appears in both train and test')
    if {r['label'] for r in train}!={'pass','fail'}:raise ValueError('Training needs human-labeled pass AND fail examples')
    train_people={r['person_id'] for r in train};test_people={r['person_id'] for r in test}
    if len(train_people|test_people)>1 and train_people & test_people:
        raise ValueError('Multiple people detected: keep test participants separate from training participants')
    return train,test,skipped

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',help='Reviewed attempts_to_label.csv files')
    p.add_argument('--out',help='New empty output folder; default: next unused model version')
    p.add_argument('--runs',default=str(Path(__file__).parent/'runs'),help='Scan reviewed runs recursively')
    p.add_argument('--test-videos',nargs='+',help='Use only these already marked test video IDs; other test clips stay excluded')
    p.add_argument('--list-data',action='store_true',help='Show selected sources/counts without training')
    p.add_argument('--choose-files',action='store_true',help='Use the old file picker instead of automatic discovery')
    args=p.parse_args()
    if args.data and args.choose_files:p.error('Use either --data or --choose-files')
    if args.test_videos and (args.data or args.choose_files):p.error('--test-videos is for automatic discovery; explicit CSVs keep their stored split')
    if not args.data and args.choose_files:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk();root.withdraw()
        try:args.data=list(filedialog.askopenfilenames(title='Choose reviewed CSV files (train + test)',filetypes=[('CSV','*.csv')]))
        finally:root.destroy()
        if not args.data:print('No CSV selected.');return
    manifest=None
    try:
        if not args.data:args.data,manifest=discover_dataset(args.runs,args.test_videos)
        train,test,skipped=load_dataset(args.data)
    except (ValueError,KeyError,OSError) as exc:p.error(str(exc))
    selection=dict(train_segments=len(train),test_segments=len(test),
        train_class_counts=dict(Counter(r['label'] for r in train)),test_class_counts=dict(Counter(r['label'] for r in test)),
        sources=manifest or dict(selected_files=[str(Path(f).resolve()) for f in args.data]))
    print('DATASET SELECTION (stored labels/splits are unchanged):')
    print(json.dumps(selection,indent=2))
    if args.list_data:return
    out=Path(args.out) if args.out else next_model_folder()
    if out.exists() and any(out.iterdir()):p.error('Output folder is not empty; choose a new --out')
    import joblib
    import sklearn
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    from sklearn.dummy import DummyClassifier
    x_train=[feature_vector(r) for r in train];y_train=[r['label'] for r in train]
    x_test=[feature_vector(r) for r in test];y_test=[r['label'] for r in test]
    pipeline=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
        RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1))
    pipeline.fit(x_train,y_train)
    pred=pipeline.predict(x_test)
    dummy=DummyClassifier(strategy='most_frequent').fit(x_train,y_train)
    notes=['Preliminary experiment: do not infer general accuracy from a small dataset.',
        'Binary form labels do not provide a diagnosis or an error-specific explanation.',
        'Hip/trunk features do not directly measure spinal curvature.']
    if len({r['person_id'] for r in train+test})==1:notes.append('Single participant: held-out videos only, not unseen-person performance.')
    if len(set(y_test))<2:notes.append('Test set has only one class; ability to distinguish both classes is not established.')
    if len(train)<50:notes.append('Fewer than 50 training segments: exploratory result, not a sample-size adequacy claim.')
    metrics=dict(schema=SCHEMA,features=FEATURES,sklearn_version=sklearn.__version__,
        train_segments=len(train),test_segments=len(test),skipped_rows=skipped,
        train_class_counts=dict(Counter(y_train)),test_class_counts=dict(Counter(y_test)),
        train_videos=sorted({r['video_id'] for r in train}),test_videos=sorted({r['video_id'] for r in test}),
        accuracy=accuracy_score(y_test,pred),majority_baseline_accuracy=accuracy_score(y_test,dummy.predict(x_test)),
        confusion_matrix_labels=['fail','pass'],confusion_matrix=confusion_matrix(y_test,pred,labels=['fail','pass']).tolist(),
        classification_report=classification_report(y_test,pred,labels=['fail','pass'],output_dict=True,zero_division=0),
        feature_importances=dict(zip(FEATURES,pipeline[-1].feature_importances_.tolist())),notes=notes)
    out.mkdir(parents=True,exist_ok=True)
    (out/'dataset_sources.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    joblib.dump(dict(schema=SCHEMA,features=FEATURES,pipeline=pipeline,train_videos=metrics['train_videos']),out/'model.joblib')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    with (out/'test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['video_id','attempt_id','label','prediction','correct']);writer.writeheader()
        for row,prediction in zip(test,pred):writer.writerow(dict(video_id=row['video_id'],attempt_id=row['attempt_id'],label=row['label'],prediction=prediction,correct=row['label']==prediction))
    with (out/'data_used.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['person_id','video_id','attempt_id','split',*FEATURES,'label','reason']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r.get(k,'') for k in fields} for r in train+test)
    print(json.dumps(metrics,indent=2))
    print(f'Model and held-out evaluation saved: {out.resolve()}')

if __name__=='__main__':main()
