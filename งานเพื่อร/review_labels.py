"""Watch each detected segment and manually label it. Never infer labels from counts."""
import argparse
import csv
from pathlib import Path
from rep_counter import write_csv

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',help='attempts_to_label.csv from counter v2')
    p.add_argument('--split',choices=['train','test'],default='train')
    args=p.parse_args()
    if not args.csv:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk();root.withdraw()
        runs=Path(__file__).parent/'runs'
        available=list(runs.glob('*/attempts_to_label.csv'))
        initial=max(available,key=lambda f:f.stat().st_mtime).parent if available else runs
        try:args.csv=filedialog.askopenfilename(title='Choose attempts_to_label.csv',initialdir=str(initial),filetypes=[('CSV','*.csv')])
        finally:root.destroy()
        if not args.csv:return
    path=Path(args.csv)
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    if not rows:print('No detected segments to review.');return
    if 'segment_status' not in fields:p.error('Use attempts_to_label.csv from counter v2')
    sources={r['source_video'] for r in rows}
    if len(sources)!=1:p.error('Review one video per CSV')
    for row in rows:row['split']=args.split
    backup=path.with_suffix('.csv.original')
    if not backup.exists():backup.write_bytes(path.read_bytes())
    import cv2
    cap=cv2.VideoCapture(next(iter(sources)))
    if not cap.isOpened():p.error('Cannot open source video; check source_video path')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0:p.error('Invalid source FPS')
    i=0;window='BUTA - Human label review'
    labels={ord('p'):('pass','reviewed_pass'),ord('d'):('fail','insufficient_depth'),
        ord('r'):('fail','incomplete_return'),ord('f'):('fail','other_form_issue'),ord('u'):('unassessable','unclear_or_bad_segment')}
    try:
        while 0<=i<len(rows):
            row=rows[i];first=max(0,round((float(row['start_time'])-.3)*fps));last=round((float(row['end_time'])+.3)*fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES,first)
            selected=None;frame_index=first
            while True:
                ok,frame=cap.read()
                if not ok or frame_index>last:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,first);frame_index=first
                    ok,frame=cap.read()
                    if not ok:raise RuntimeError('Cannot decode segment')
                scale=min(650/frame.shape[0],800/frame.shape[1]);frame=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
                import numpy as np
                header=np.zeros((155,frame.shape[1],3),dtype=np.uint8)
                lines=[f'Segment {i+1}/{len(rows)}  {args.split}  label: {row["label"] or "unset"}',
                    f'{float(row["start_time"]):.1f}-{float(row["end_time"]):.1f}s / {row["segment_status"]}',
                    'P: pass | D: depth fail | R: return fail',
                    'F: other form fail | U: cannot assess',
                    'N/B: next/back | Space: pause | Q: save+exit']
                for j,line in enumerate(lines):cv2.putText(header,line,(8,24+j*28),cv2.FONT_HERSHEY_SIMPLEX,.43,(230,240,240),1)
                cv2.imshow(window,np.vstack([header,frame]))
                key=cv2.waitKey(max(1,round(1000/fps)))&255
                if key==32:
                    while True:
                        key=cv2.waitKey(50)&255
                        if key!=255 or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
                if key in labels or key in (ord('n'),ord('b'),ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                    selected=key;break
                frame_index+=1
            if selected in labels:
                row['label'],row['reason']=labels[selected]
                write_csv(path,rows,fields);i+=1
            elif selected==ord('n'):i+=1
            elif selected==ord('b'):i=max(0,i-1)
            else:break
    finally:
        cap.release();cv2.destroyAllWindows();write_csv(path,rows,fields)
    print(f'Review saved: {path.resolve()}')

if __name__=='__main__':main()
