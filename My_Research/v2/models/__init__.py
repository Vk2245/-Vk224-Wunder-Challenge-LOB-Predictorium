"""
Model registry — maps model names to classes and default configs.
"""

from models.xlstm import xLSTMModel, XLSTM_DEFAULTS
from models.ttt_linear import TTTLinearModel, TTT_LINEAR_DEFAULTS
from models.sparse_moe import SparseMoEModel, SparseMoEForExport, SPARSE_MOE_DEFAULTS
from models.ms_tcn import MultiScaleTCN, MS_TCN_DEFAULTS
from models.enc_dec import EncDecModel, ENC_DEC_DEFAULTS
from models.bi_gru import BiGRUModel, BI_GRU_DEFAULTS
from models.timemixer import TimeMixerModel, TIMEMIXER_DEFAULTS
from models.lob_transformer import LOBTransformerModel, LOB_TRANSFORMER_DEFAULTS
from models.transformer_bigru import TransformerBiGRUModel, TRANSFORMER_BIGRU_DEFAULTS
from models.transformer_timemixer import TransformerTimeMixerModel, TRANSFORMER_TIMEMIXER_DEFAULTS
from models.triple_fusion import TripleFusionModel, TRIPLE_FUSION_DEFAULTS
from models.wavenet_dense import WaveNetDenseModel, WAVENET_DENSE_DEFAULTS
from models.fnet import FNetModel, FNET_DEFAULTS
from models.mlp_mixer import MLPMixerModel, MLP_MIXER_DEFAULTS
from models.patch_tst import PatchTSTModel, PATCH_TST_DEFAULTS
from models.itransformer import iTransformerModel, ITRANSFORMER_DEFAULTS
from models.dual_horizon import DualHorizonModel, DUAL_HORIZON_DEFAULTS


MODEL_REGISTRY = {
    "xlstm": {
        "class": xLSTMModel,
        "defaults": XLSTM_DEFAULTS,
        "export_wrapper": None,
    },
    "ttt_linear": {
        "class": TTTLinearModel,
        "defaults": TTT_LINEAR_DEFAULTS,
        "export_wrapper": None,
    },
    "sparse_moe": {
        "class": SparseMoEModel,
        "defaults": SPARSE_MOE_DEFAULTS,
        "export_wrapper": SparseMoEForExport,
    },
    "ms_tcn": {
        "class": MultiScaleTCN,
        "defaults": MS_TCN_DEFAULTS,
        "export_wrapper": None,
    },
    "enc_dec": {
        "class": EncDecModel,
        "defaults": ENC_DEC_DEFAULTS,
        "export_wrapper": None,
    },
    "bi_gru": {
        "class": BiGRUModel,
        "defaults": BI_GRU_DEFAULTS,
        "export_wrapper": None,
    },
    "timemixer": {
        "class": TimeMixerModel,
        "defaults": TIMEMIXER_DEFAULTS,
        "export_wrapper": None,
    },
    "lob_transformer": {
        "class": LOBTransformerModel,
        "defaults": LOB_TRANSFORMER_DEFAULTS,
        "export_wrapper": None,
    },
    "transformer_bigru": {
        "class": TransformerBiGRUModel,
        "defaults": TRANSFORMER_BIGRU_DEFAULTS,
        "export_wrapper": None,
    },
    "transformer_timemixer": {
        "class": TransformerTimeMixerModel,
        "defaults": TRANSFORMER_TIMEMIXER_DEFAULTS,
        "export_wrapper": None,
    },
    "triple_fusion": {
        "class": TripleFusionModel,
        "defaults": TRIPLE_FUSION_DEFAULTS,
        "export_wrapper": None,
    },
    "wavenet_dense": {
        "class": WaveNetDenseModel,
        "defaults": WAVENET_DENSE_DEFAULTS,
        "export_wrapper": None,
    },
    "fnet": {
        "class": FNetModel,
        "defaults": FNET_DEFAULTS,
        "export_wrapper": None,
    },
    "mlp_mixer": {
        "class": MLPMixerModel,
        "defaults": MLP_MIXER_DEFAULTS,
        "export_wrapper": None,
    },
    "patch_tst": {
        "class": PatchTSTModel,
        "defaults": PATCH_TST_DEFAULTS,
        "export_wrapper": None,
    },
    "itransformer": {
        "class": iTransformerModel,
        "defaults": ITRANSFORMER_DEFAULTS,
        "export_wrapper": None,
    },
    "dual_horizon": {
        "class": DualHorizonModel,
        "defaults": DUAL_HORIZON_DEFAULTS,
        "export_wrapper": None,
    },
}



def get_model(name: str, **overrides):
    """
    Instantiate a model by name with optional hyperparameter overrides.

    Usage:
        model = get_model("xlstm", hidden_dim=192, dropout=0.2)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    entry = MODEL_REGISTRY[name]
    config = {**entry["defaults"], **overrides}

    # Separate model kwargs from training kwargs
    model_cls = entry["class"]
    import inspect
    valid_params = set(inspect.signature(model_cls.__init__).parameters.keys()) - {"self"}
    model_kwargs = {k: v for k, v in config.items() if k in valid_params}

    return model_cls(**model_kwargs), config


def get_export_wrapper(name: str):
    """Get the ONNX export wrapper class for a model (or None)."""
    return MODEL_REGISTRY[name]["export_wrapper"]
