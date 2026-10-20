"""Geometry and experimental rep segmentation; no form classifier."""
import math

def angle(a, b, c):
    u = (a[0]-b[0], a[1]-b[1])
    v = (c[0]-b[0], c[1]-b[1])
    denominator = math.hypot(*u)*math.hypot(*v)
    if denominator < 1e-9:
        return None
    return math.degrees(math.acos(max(-1, min(1, sum(x*y for x,y in zip(u,v))/denominator))))

class RepTracker:
    def __init__(self, standing=160, bent=120, minimum_duration=0.6, maximum_gap=0):
        if not 0 < bent < standing < 180:
            raise ValueError('Require 0 < bent < standing < 180')
        self.standing, self.bent = standing, bent
        self.minimum_duration = minimum_duration
        self.maximum_gap = maximum_gap
        self.last_valid_time = None
        self.reset()
        self.reps = []

    def reset(self):
        self.state = 'waiting_for_standing'
        self.samples = []

    def update(self, time, knee, hip=None, trunk=None):
        if knee is None:
            if self.last_valid_time is None or time-self.last_valid_time > self.maximum_gap:
                self.reset()
            return
        if self.last_valid_time is not None and time-self.last_valid_time > self.maximum_gap and self.maximum_gap > 0:
            self.reset()
        self.last_valid_time = time
        sample = (time,knee,hip,trunk)
        if self.state == 'waiting_for_standing':
            if knee >= self.standing:
                self.state = 'ready'
                self.last_standing = time
        elif self.state == 'ready':
            if knee >= self.standing:
                self.last_standing = time
            else:
                self.samples = [(self.last_standing,self.standing,None,None),sample]
                self.state = 'moving'
                if knee <= self.bent:
                    self.state = 'bent'
        else:
            self.samples.append(sample)
            if knee <= self.bent:
                self.state = 'bent'
            if knee >= self.standing:
                if self.state == 'bent' and time-self.samples[0][0] >= self.minimum_duration:
                    hips = [s[2] for s in self.samples if s[2] is not None]
                    trunks = [s[3] for s in self.samples if s[3] is not None]
                    self.reps.append(dict(rep_id=len(self.reps)+1,start_time=self.samples[0][0],end_time=time,
                        duration_s=time-self.samples[0][0],knee_min=min(s[1] for s in self.samples),
                        hip_min=min(hips) if hips else None,trunk_max=max(trunks) if trunks else None))
                self.samples = []
                self.state = 'ready'
                self.last_standing = time
