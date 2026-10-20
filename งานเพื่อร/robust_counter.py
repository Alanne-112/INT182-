"""Experimental movement segmentation, separate from human form labels."""
import math

FEATURES=['knee_min','knee_return_peak','duration_s','hip_min','trunk_max']
SCHEMA='buta-sideview-v2'

class StableRepTracker:
    def __init__(self,standing=150,bent=120,maximum_gap=.25,hold=.10,minimum_duration=.6):
        if not 0 < bent < standing-8 < 180:
            raise ValueError('Require 0 < bent < standing - 8 < 180')
        self.standing,self.bent=standing,bent
        self.maximum_gap,self.hold,self.minimum_duration=maximum_gap,hold,minimum_duration
        self.state='waiting_for_standing'
        self.reps=[]
        self.attempts=[]
        self.active=[]
        self.top_since=self.bottom_since=None
        self.last_valid=None
        self.bottom_confirmed=False
        self.returning=False
        self.return_peak=None

    def _finish(self,reason):
        if not self.active:return
        samples=self.active
        low=min(range(len(samples)),key=lambda i:samples[i][1])
        duration=samples[-1][0]-samples[0][0]
        excursion=max(s[1] for s in samples)-samples[low][1]
        # Ignore tiny standing jitter, but retain partial/uncertain attempts.
        if duration>=.25 and excursion>=15:
            hips=[s[2] for s in samples if s[2] is not None]
            trunks=[s[3] for s in samples if s[3] is not None]
            counted=reason=='returned' and self.bottom_confirmed and duration>=self.minimum_duration
            if reason=='returned' and not counted:
                reason='depth_not_reached' if not self.bottom_confirmed else 'too_fast'
            row=dict(attempt_id=len(self.attempts)+1,start_time=samples[0][0],end_time=samples[-1][0],
                duration_s=duration,knee_min=samples[low][1],
                knee_return_peak=max(s[1] for s in samples[low:]),
                hip_min=min(hips) if hips else None,trunk_max=max(trunks) if trunks else None,
                counted=counted,segment_status=reason)
            self.attempts.append(row)
            if counted:self.reps.append(row)
        self.active=[]
        self.bottom_confirmed=False
        self.bottom_since=self.top_since=None
        self.returning=False
        self.return_peak=None

    def update(self,t,knee,hip=None,trunk=None):
        if knee is None or not math.isfinite(knee):
            if self.last_valid is not None and t-self.last_valid>self.maximum_gap:
                self._finish('tracking_lost')
                self.state='waiting_for_standing'
                self.top_since=None
            return
        if self.last_valid is not None and t-self.last_valid>self.maximum_gap:
            self._finish('tracking_lost')
            self.state='waiting_for_standing'
            self.top_since=None
        self.last_valid=t
        sample=(t,knee,hip,trunk)
        if self.state=='waiting_for_standing':
            if knee>=self.standing:
                if self.top_since is None:self.top_since=t
                if t-self.top_since+1e-9>=self.hold:
                    self.state='ready';self.top_since=None
            else:self.top_since=None
            return
        if self.state=='ready':
            if knee<self.standing-8:
                self.active=[sample];self.state='moving'
            else:return
        else:
            # Detect a second descent after a partial return without merging cycles.
            if self.returning and knee<self.return_peak-12:
                peak_index=max(range(len(self.active)),key=lambda i:self.active[i][0] if self.active[i][1]>=self.return_peak-1 else -1)
                peak=self.active[peak_index]
                self._finish('incomplete_return')
                self.active=[peak,sample];self.state='moving'
            else:self.active.append(sample)
        minimum=min(s[1] for s in self.active)
        if knee<=self.bent:
            if self.bottom_since is None:self.bottom_since=t
            if t-self.bottom_since+1e-9>=self.hold:self.bottom_confirmed=True
        elif not self.bottom_confirmed:self.bottom_since=None
        if knee>=minimum+25:
            self.returning=True
            self.return_peak=knee if self.return_peak is None else max(self.return_peak,knee)
        if knee>=self.standing:
            if self.top_since is None:self.top_since=t
            if t-self.top_since+1e-9>=self.hold:
                self._finish('returned');self.state='ready'
        else:self.top_since=None
        if self.active:self.state='bent' if self.bottom_confirmed else 'moving'

    def finalize(self,reason='video_end'):
        self._finish(reason)
        self.state='waiting_for_standing'

def feature_vector(rep):
    return [float(rep[f]) if rep.get(f) not in (None,'') else float('nan') for f in FEATURES]
