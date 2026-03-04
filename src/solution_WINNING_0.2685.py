"""
solution_WINNING_0.2685.py — BACKUP OF WINNING SUBMISSION
===========================================================
This file scored 0.2685 on competition server.
DO NOT MODIFY - keep as backup!

Single model: TimeMixer INT8
Runtime: ~21 minutes
Score: 0.2685 (45x better than 0.006)
"""
import os
import numpy as np
from collections import deque
import onnxruntime as ort
from utils import DataPoint

SEQ_LEN = 100


class PredictionModel:
    def __init__(self):
        # Path resolution
        here = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(here) == "src":
            model_dir = os.path.join(os.path.dirname(here), "models")
        else:
            model_dir = here

        # Load scaler
        sc = np.load(os.path.join(model_dir, "scaler_tm.npz"))
        self.mu = sc["mu"].astype(np.float32)
        self.inv_sig = (1.0 / sc["sig"]).astype(np.float32)

        # ONNX session options - MAXIMUM SPEED
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Load TimeMixer (INT8)
        tm_path = os.path.join(model_dir, "timemixer_int8.onnx")
        if not os.path.exists(tm_path):
            tm_path = os.path.join(model_dir, "timemixer.onnx")
        self.sess = ort.InferenceSession(tm_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name

        # Rolling buffer
        self.buf = deque(maxlen=SEQ_LEN)
        self.cur_seq = None
        
        # Pre-allocate arrays
        self.x_buf = np.empty((SEQ_LEN, 32), dtype=np.float32)
        self.x_in = np.empty((1, SEQ_LEN, 32), dtype=np.float32)

    def predict(self, dp: DataPoint):
        # Reset buffer on new sequence
        if dp.seq_ix != self.cur_seq:
            self.cur_seq = dp.seq_ix
            self.buf.clear()

        # Push current state
        x_raw = np.asarray(dp.state, dtype=np.float32)
        self.buf.append(x_raw)

        # Return None if not a prediction step
        if not dp.need_prediction:
            return None

        # Not enough context yet
        if len(self.buf) < SEQ_LEN:
            return np.zeros(2, dtype=np.float32)

        # Stack buffer
        for i, x in enumerate(self.buf):
            self.x_buf[i] = x

        # Normalize and predict
        np.subtract(self.x_buf, self.mu, out=self.x_in[0])
        np.multiply(self.x_in[0], self.inv_sig, out=self.x_in[0])
        pred = self.sess.run([self.out], {self.inp: self.x_in})[0][0]

        return pred.astype(np.float64)
