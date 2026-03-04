"""
solution_2model.py — 2-Model Ensemble: BiGRU + TimeMixer (INT8)
================================================================
Try different ensemble weights to maximize score.

Current single model: 0.2685
Target 2-model: 0.28+
"""
import os
import numpy as np
from collections import deque
import onnxruntime as ort
from utils import DataPoint

SEQ_LEN = 100


class PredictionModel:
    def __init__(self):
        here = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(here) == "src":
            model_dir = os.path.join(os.path.dirname(here), "models")
        else:
            model_dir = here

        # Load scalers
        sc1 = np.load(os.path.join(model_dir, "scaler.npz"))
        self.mu1 = sc1["mu"].astype(np.float32)
        self.inv_sig1 = (1.0 / sc1["sig"]).astype(np.float32)

        sc2 = np.load(os.path.join(model_dir, "scaler_tm.npz"))
        self.mu2 = sc2["mu"].astype(np.float32)
        self.inv_sig2 = (1.0 / sc2["sig"]).astype(np.float32)

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # BiGRU INT8
        bigru_path = os.path.join(model_dir, "bigru_int8.onnx")
        if not os.path.exists(bigru_path):
            bigru_path = os.path.join(model_dir, "bigru.onnx")
        self.sess1 = ort.InferenceSession(bigru_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.inp1 = self.sess1.get_inputs()[0].name
        self.out1 = self.sess1.get_outputs()[0].name

        # TimeMixer INT8
        tm_path = os.path.join(model_dir, "timemixer_int8.onnx")
        if not os.path.exists(tm_path):
            tm_path = os.path.join(model_dir, "timemixer.onnx")
        self.sess2 = ort.InferenceSession(tm_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.inp2 = self.sess2.get_inputs()[0].name
        self.out2 = self.sess2.get_outputs()[0].name

        # OPTIMIZED WEIGHTS (favor TimeMixer heavily)
        self.w1 = np.float32(0.25)  # BiGRU
        self.w2 = np.float32(0.75)  # TimeMixer (better model)

        self.buf = deque(maxlen=SEQ_LEN)
        self.cur_seq = None
        self.x_buf = np.empty((SEQ_LEN, 32), dtype=np.float32)
        self.x1_in = np.empty((1, SEQ_LEN, 32), dtype=np.float32)
        self.x2_in = np.empty((1, SEQ_LEN, 32), dtype=np.float32)

    def predict(self, dp: DataPoint):
        if dp.seq_ix != self.cur_seq:
            self.cur_seq = dp.seq_ix
            self.buf.clear()

        x_raw = np.asarray(dp.state, dtype=np.float32)
        self.buf.append(x_raw)

        if not dp.need_prediction:
            return None

        if len(self.buf) < SEQ_LEN:
            return np.zeros(2, dtype=np.float32)

        for i, x in enumerate(self.buf):
            self.x_buf[i] = x

        # BiGRU
        np.subtract(self.x_buf, self.mu1, out=self.x1_in[0])
        np.multiply(self.x1_in[0], self.inv_sig1, out=self.x1_in[0])
        pred1 = self.sess1.run([self.out1], {self.inp1: self.x1_in})[0][0]

        # TimeMixer
        np.subtract(self.x_buf, self.mu2, out=self.x2_in[0])
        np.multiply(self.x2_in[0], self.inv_sig2, out=self.x2_in[0])
        pred2 = self.sess2.run([self.out2], {self.inp2: self.x2_in})[0][0]

        # Ensemble
        pred = self.w1 * pred1 + self.w2 * pred2

        return pred.astype(np.float64)
