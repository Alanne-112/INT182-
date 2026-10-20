"""Play a squat video with a full pose skeleton and a large rep counter."""
import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from core import angle, RepTracker
from robust_counter import StableRepTracker,FEATURES,SCHEMA,feature_vector

CONNECTIONS = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
 (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
 (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
 (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
 (27,29),(28,30),(29,31),(30,32),(27,31),(28,32)]

def choose_video():
    try:
        import tkinter as tk
        from tkinter import filedialog
        window=tk.Tk()
        window.withdraw()
        try:
            return filedialog.askopenfilename(title='BUTA AI - Choose a squat video',
                filetypes=[('Video files','*.mp4 *.mov *.avi *.mkv'),('All files','*.*')])
        finally:
            window.destroy()
    except ImportError as exc:
        raise RuntimeError('File picker unavailable. Use --video "C:\\path\\video.mp4" instead.') from exc

def write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def export_run(out,rows,tracker,args,selected_side,fps,finished):
    rep_rows=[]
    for rep in tracker.reps:
        ending=next((r for r in reversed(rows) if abs(r['time_s']-rep['end_time'])<1e-6),None)
        rep_rows.append(dict(video_id=Path(args.video).stem,rep_id=rep.get('rep_id',rep.get('attempt_id')),side=selected_side,
            start_time=rep['start_time'],end_time=rep['end_time'],duration_s=rep['duration_s'],
            knee_min=rep['knee_min'],knee_at_count=ending['knee_smoothed_deg'] if ending else '',
            label='',reason=''))
    write_csv(out/'frames.csv',rows,['frame','time_s','side','left_confidence','right_confidence',
        'weakest_joint','knee_raw_deg','knee_smoothed_deg','hip_deg','trunk_deg','tracking_status','counter_state','rep_count'])
    write_csv(out/'reps_to_review.csv',rep_rows,['video_id','rep_id','side','start_time','end_time',
        'duration_s','knee_min','knee_at_count','label','reason'])
    if hasattr(tracker,'attempts'):
        attempts=[dict(schema=SCHEMA,person_id=getattr(args,'person_id','P01'),video_id=Path(args.video).stem,
            source_video=str(Path(args.video).resolve()),side=selected_side,**a,label='',reason='',split='')
            for a in tracker.attempts]
        write_csv(out/'attempts_to_label.csv',attempts,['schema','person_id','video_id','source_video','side','attempt_id',
            'start_time','end_time','duration_s','knee_min','knee_return_peak','hip_min','trunk_max','counted','segment_status',
            'label','reason','split','rf_prediction'])
    summary=dict(video=str(Path(args.video).resolve()),processed_frames=len(rows),fps=fps,
        finished=finished,selected_side=selected_side,completed_reps=len(tracker.reps),
        valid_frame_fraction=sum(r['knee_smoothed_deg'] is not None for r in rows)/len(rows) if rows else 0,
        standing_angle=args.standing_angle,bent_angle=args.bent_angle,
        visibility_threshold=0.6,maximum_gap_s=getattr(tracker,'maximum_gap',.2),
        note='Movement counter only. Review CSV includes counted reps only; missing attempts must be reviewed in the video. knee_at_count is not maximum return angle. Labels are intentionally blank.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if hasattr(tracker,'attempts'):
        summary.update(schema=SCHEMA,attempts_for_review=len(tracker.attempts),hold_s=tracker.hold,
            note='attempts_to_label.csv includes detected partial and tracking-lost segments. Segments are not guaranteed to match every physical rep. Labels are blank. knee_return_peak is the observed peak after the lowest knee angle inside that segment. Counting is not a form judgement.')
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser(description='Squat video + skeleton + rep counter')
    p.add_argument('--video',help='Omit to choose a video in a file picker')
    p.add_argument('--side',choices=['auto','left','right'],default='auto')
    p.add_argument('--standing-angle',type=float,default=150)
    p.add_argument('--bent-angle',type=float,default=120)
    p.add_argument('--save',help='Optional new .avi path (MJPG)')
    p.add_argument('--out',help='New empty directory for CSV and summary (default: unique runs folder)')
    p.add_argument('--person-id',default='P01')
    p.add_argument('--rf-model',help='Optional locally trained RF model.joblib')
    p.add_argument('--no-display',action='store_true',help='Process without opening a window')
    p.add_argument('--model',default=str(Path(__file__).parent/'models/pose_landmarker_lite.task'))
    args=p.parse_args()
    if not args.video:
        if args.no_display: p.error('--video is required with --no-display')
        try: args.video=choose_video()
        except RuntimeError as exc: p.error(str(exc))
        if not args.video:
            print('No video selected.'); return
    import cv2
    import mediapipe as mp
    import numpy as np
    if not Path(args.model).is_file(): p.error('Model missing; run setup_model.py')
    if args.save and (Path(args.save).exists() or Path(args.save).suffix.lower()!='.avi'):
        p.error('--save must be a new .avi path')
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened(): p.error('Cannot open video')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0: p.error('Invalid video FPS')
    tracker=StableRepTracker(args.standing_angle,args.bent_angle)
    rf_bundle=None
    if args.rf_model:
        import joblib
        rf_bundle=joblib.load(args.rf_model)
        if rf_bundle.get('schema')!=SCHEMA or rf_bundle.get('features')!=FEATURES:
            p.error('RF model schema does not match this counter')
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out=Path(args.out) if args.out else Path(__file__).parent/'runs'/(Path(args.video).stem+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if out.exists() and any(out.iterdir()):
        cap.release(); p.error('--out must be empty; choose a new directory')
    out.mkdir(parents=True,exist_ok=True)
    selected_side=None if args.side=='auto' else args.side
    side_ids={'left':[23,25,27],'right':[24,26,28]}
    scores={'left':0.0,'right':0.0}
    valid_frames=0
    options=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=args.model),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_poses=1)
    writer=None
    index=0
    stamp=-1
    previous=None
    last_valid=None
    rows=[]
    finished=False
    window_name='BUTA AI - Squat rep counter'
    latest_prediction='Not trained'
    predicted_attempts=0
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as model:
            while True:
                started=time.monotonic()
                ok,frame=cap.read()
                if not ok:
                    finished=True
                    break
                h,w=frame.shape[:2]
                t=index/fps
                stamp=max(stamp+1,round(t*1000))
                result=model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),stamp)
                knee=None
                hip=trunk=None
                raw_knee=None
                weakest_joint=''
                scores={'left':0.0,'right':0.0}
                pose_found=bool(result.pose_landmarks)
                if result.pose_landmarks:
                    landmarks=result.pose_landmarks[0]
                    points=[(round(l.x*w),round(l.y*h)) for l in landmarks]
                    visible=[min(l.visibility,l.presence)>=0.5 for l in landmarks]
                    for a,b in CONNECTIONS:
                        if visible[a] and visible[b]: cv2.line(frame,points[a],points[b],(80,240,100),max(2,w//250))
                    for i,point in enumerate(points):
                        if visible[i]: cv2.circle(frame,point,max(3,w//150),(40,180,255),-1)
                    scores={side:min(min(landmarks[i].visibility,landmarks[i].presence) for i in ids)
                        for side,ids in side_ids.items()}
                    # Lock one side for the entire run: never mix legs inside a rep.
                    if selected_side is None:
                        best=max(scores,key=scores.get)
                        if scores[best]>=0.6: selected_side=best
                    if selected_side is not None and scores[selected_side]>=0.6:
                        knee=angle(*[(landmarks[i].x*w,landmarks[i].y*h) for i in side_ids[selected_side]])
                        shoulder_id=11 if selected_side=='left' else 12
                        hip_id,knee_id,_=side_ids[selected_side]
                        if min(landmarks[shoulder_id].visibility,landmarks[shoulder_id].presence)>=.6:
                            shoulder=(landmarks[shoulder_id].x*w,landmarks[shoulder_id].y*h)
                            pelvis=(landmarks[hip_id].x*w,landmarks[hip_id].y*h)
                            kneep=(landmarks[knee_id].x*w,landmarks[knee_id].y*h)
                            hip=angle(shoulder,pelvis,kneep)
                            trunk=math.degrees(math.atan2(abs(shoulder[0]-pelvis[0]),pelvis[1]-shoulder[1]))
                    if selected_side is not None:
                        ids=side_ids[selected_side]
                        weak_index=min(range(3),key=lambda j:min(landmarks[ids[j]].visibility,landmarks[ids[j]].presence))
                        weakest_joint=['hip','knee','ankle'][weak_index]
                raw_knee=knee
                if knee is not None:
                    valid_frames+=1
                    if last_valid is None or t-last_valid>tracker.maximum_gap: previous=None
                    knee=knee if previous is None else 0.3*knee+0.7*previous
                    previous=knee
                    last_valid=t
                tracker.update(t,knee,hip,trunk)
                if len(tracker.attempts)>predicted_attempts:
                    last=tracker.attempts[-1]
                    if last['segment_status']=='tracking_lost':latest_prediction='Unassessable'
                    elif rf_bundle is not None:
                        latest_prediction=str(rf_bundle['pipeline'].predict([feature_vector(last)])[0])
                        last['rf_prediction']=latest_prediction
                    predicted_attempts=len(tracker.attempts)
                status='tracking' if knee is not None else 'no_person' if not pose_found else 'leg_unclear'
                rows.append(dict(frame=index,time_s=t,side=selected_side or '',left_confidence=scores['left'],
                    right_confidence=scores['right'],weakest_joint=weakest_joint,knee_raw_deg=raw_knee,
                    knee_smoothed_deg=knee,hip_deg=hip,trunk_deg=trunk,tracking_status=status,counter_state=tracker.state,rep_count=len(tracker.reps)))
                scale=min(720/h,720/w)
                picture=cv2.resize(frame,(round(w*scale),round(h*scale)))
                ph,pw=picture.shape[:2]
                canvas=np.full((max(640,ph),pw+280,3),(35,25,15),dtype=np.uint8)
                canvas[:ph,280:]=picture
                def text(message,y,size=0.7,color=(240,240,240)):
                    cv2.putText(canvas,message,(20,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,2,cv2.LINE_AA)
                text('BUTA AI',45,1)
                text('SQUAT / REPS',100)
                text(str(len(tracker.reps)),205,3,(80,240,100))
                text(('No person detected' if not pose_found else 'Leg not clear') if knee is None else f'Knee: {knee:.0f} deg',270)
                text('DOWN' if tracker.state=='bent' else 'UP / READY' if tracker.state=='ready' else 'MOVING',315)
                text('Space: pause',355,0.55)
                text('Q / Esc: exit',385,0.55)
                text(f'Side: {selected_side or "selecting"}',425,0.5)
                text(f'Leg confidence L:{scores["left"]:.2f} R:{scores["right"]:.2f}',450,0.43)
                if knee is None:
                    text(f'Check {weakest_joint or "full body"} visibility',480,0.47,(70,190,255))
                    text('Show hip, knee and ankle',505,0.45)
                    text('Use a clear side view',530,0.45)
                elif tracker.state=='waiting_for_standing':
                    text('Stand tall to start / resume',480,0.45,(70,190,255))
                else:
                    text('Tracking OK',480,0.55,(80,240,100))
                text(f'Count: down <= {args.bent_angle:g}',560,0.45)
                text(f'then up >= {args.standing_angle:g}',585,0.45)
                text(f'RF: {latest_prediction}' if rf_bundle else 'Form model: not trained',615,0.45)
                progress=min(1,(index+1)/total_frames) if total_frames>0 else 0
                cv2.rectangle(canvas,(280,canvas.shape[0]-8),(canvas.shape[1]-1,canvas.shape[0]-1),(60,60,60),-1)
                cv2.rectangle(canvas,(280,canvas.shape[0]-8),(280+round(pw*progress),canvas.shape[0]-1),(80,240,100),-1)
                if args.save:
                    if writer is None:
                        Path(args.save).parent.mkdir(parents=True,exist_ok=True)
                        writer=cv2.VideoWriter(args.save,cv2.VideoWriter_fourcc(*'MJPG'),fps,(canvas.shape[1],canvas.shape[0]))
                        if not writer.isOpened(): raise RuntimeError('Cannot create AVI writer')
                    writer.write(canvas)
                index+=1
                if not args.no_display:
                    cv2.imshow(window_name,canvas)
                    delay=max(1,round(1000*(1/fps-(time.monotonic()-started))))
                    key=cv2.waitKey(delay)&255
                    if key==32:
                        paused=canvas.copy()
                        cv2.putText(paused,'PAUSED - Space to resume',(20,335),cv2.FONT_HERSHEY_SIMPLEX,.5,(70,190,255),1)
                        cv2.imshow(window_name,paused)
                        while True:
                            key=cv2.waitKey(50)&255
                            if key in (32,27,ord('q')) or cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE)<1: break
                    if key in (27,ord('q')): break
                    if cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE)<1: break
    finally:
        cap.release()
        if writer is not None: writer.release()
        if not args.no_display: cv2.destroyAllWindows()
    tracker.finalize('video_end' if finished else 'stopped')
    summary=export_run(out,rows,tracker,args,selected_side,fps,finished)
    if rf_bundle is not None:
        summary['rf_model']=str(Path(args.rf_model).resolve())
        summary['rf_seen_training_video']=Path(args.video).stem in rf_bundle.get('train_videos',[])
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f'Completed reps: {len(tracker.reps)} | processed frames: {index}')
    print(f'Side: {selected_side or "not detected"} | valid knee frames: {valid_frames}/{index}')
    print('Experimental movement counter; not a form assessment.')
    print(f'CSV and summary saved to: {out.resolve()}')
    print(f'Tracking available: {summary["valid_frame_fraction"]:.1%} of processed frames (not accuracy).')

if __name__=='__main__': main()
