import sys, os, types

PROJECT_ROOT = r'C:\VK224\Project\LOB Prediction agent'
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

class DataPoint:
    def __init__(self, seq_ix, step_in_seq, need_prediction, state):
        self.seq_ix=seq_ix; self.step_in_seq=step_in_seq
        self.need_prediction=need_prediction; self.state=state

u = types.ModuleType('utils'); u.DataPoint = DataPoint
sys.modules['utils'] = u

from solution import PredictionModel
import numpy as np, pandas as pd, time

print("Loading model...")
model = PredictionModel()

print("Loading 5000 rows...")
df = pd.read_parquet('datasets/valid.parquet').head(5000)
state_cols = [f'p{i}' for i in range(12)] + [f'v{i}' for i in range(12)] + \
             [f'dp{i}' for i in range(4)]  + [f'dv{i}' for i in range(4)]
states     = df[state_cols].values.astype('float32')
seq_ixs    = df['seq_ix'].values
steps      = df['step_in_seq'].values
need_preds = df['need_prediction'].values.astype(bool)

print("Running speed test...")
t = time.time()
for i in range(len(df)):
    dp = DataPoint(int(seq_ixs[i]), int(steps[i]), bool(need_preds[i]), states[i])
    model.predict(dp)

elapsed = time.time() - t
rate    = 5000 / elapsed
est_min = 1_301_044 / rate / 60

print(f"\nSpeed : {rate:.0f} rows/s")
print(f"Est   : {est_min:.1f} min for full test (limit=60 min)")
if est_min < 60:
    print("OK — fast enough to submit!")
else:
    print("TOO SLOW — need optimization")