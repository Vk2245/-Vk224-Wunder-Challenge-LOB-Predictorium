import sys, os, types, time
PROJECT_ROOT = r'C:\VK224\Project\LOB Prediction agent'
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

class DataPoint:
    def __init__(self, a, b, c, d):
        self.seq_ix=a; self.step_in_seq=b; self.need_prediction=c; self.state=d

u=types.ModuleType('utils'); u.DataPoint=DataPoint; sys.modules['utils']=u

from solution import PredictionModel
import numpy as np, pandas as pd, joblib

model = PredictionModel()
df = pd.read_parquet('datasets/valid.parquet').head(1000)
state_cols=[f'p{i}' for i in range(12)]+[f'v{i}' for i in range(12)]+[f'dp{i}' for i in range(4)]+[f'dv{i}' for i in range(4)]
states=df[state_cols].values.astype('float32')
seq_ixs=df['seq_ix'].values; steps=df['step_in_seq'].values
need_preds=df['need_prediction'].values.astype(bool)

# Get a sample prediction row
sample_x = None
for i in range(len(df)):
    dp = DataPoint(int(seq_ixs[i]),int(steps[i]),bool(need_preds[i]),states[i])
    if dp.seq_ix != model._seq_id:
        model._seq_id = dp.seq_ix; model._reset()
    signals = model._extract(dp.state)
    model._update_buffers(signals)
    if dp.need_prediction and sample_x is None:
        sample_x = model._build_vec(signals).copy()

print(f"Feature vector shape: {sample_x.shape}")
N = 10000

# Time each component
t = time.time()
for _ in range(N): model.lgbm_t0.predict(sample_x)
print(f"lgbm_t0.predict: {(time.time()-t)/N*1000:.3f} ms")

t = time.time()
for _ in range(N): model.xgb_t0.predict(sample_x)
print(f"xgb_t0.predict:  {(time.time()-t)/N*1000:.3f} ms")

t = time.time()
for _ in range(N): model.lgbm_t1.predict(sample_x)
print(f"lgbm_t1.predict: {(time.time()-t)/N*1000:.3f} ms")

t = time.time()
for _ in range(N): model.xgb_t1.predict(sample_x)
print(f"xgb_t1.predict:  {(time.time()-t)/N*1000:.3f} ms")

t = time.time()
for _ in range(N): model._build_vec(signals)
print(f"_build_vec:      {(time.time()-t)/N*1000:.3f} ms")

t = time.time()
for _ in range(N): model._extract(states[0])
print(f"_extract:        {(time.time()-t)/N*1000:.3f} ms")

total_ms = (
    (time.time()-t)/N*1000 +
    0  # rough sum
)
print(f"\nAt 178 rows/s = {1000/178:.1f} ms/row total")
print(f"60min budget  = {60*60*1000/1_301_044:.2f} ms/row max")