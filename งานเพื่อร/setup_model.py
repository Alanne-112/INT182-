from pathlib import Path
from urllib.request import urlretrieve

URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
if __name__ == '__main__':
    target = Path(__file__).parent / 'models' / 'pose_landmarker_lite.task'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp = target.with_suffix('.download')
        urlretrieve(URL, temp)
        temp.replace(target)
    print(target)
