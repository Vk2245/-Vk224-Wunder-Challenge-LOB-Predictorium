import numpy as np
import pandas as pd


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature set — used for t0 pipeline (strong performer)."""

    df = df.copy()

    bid_prices = df[[f"p{i}" for i in range(6)]].values
    ask_prices = df[[f"p{i}" for i in range(6, 12)]].values
    bid_vols   = df[[f"v{i}" for i in range(6)]].values
    ask_vols   = df[[f"v{i}" for i in range(6, 12)]].values

    # ── Basic price ──────────────────────────────────────────────
    df["midprice"]    = (df["p0"] + df["p6"]) / 2
    df["spread"]      = df["p6"] - df["p0"]
    df["price_range"] = ask_prices.max(axis=1) - bid_prices.min(axis=1)

    # ── Volume ───────────────────────────────────────────────────
    df["bid_volume_sum"] = bid_vols.sum(axis=1)
    df["ask_volume_sum"] = ask_vols.sum(axis=1)
    df["volume_total"]   = df["bid_volume_sum"] + df["ask_volume_sum"]
    df["volume_imbalance"] = (
        df["bid_volume_sum"] - df["ask_volume_sum"]
    ) / (df["volume_total"] + 1e-6)

    # ── Microprice ───────────────────────────────────────────────
    df["microprice"] = (
        df["p0"] * df["ask_volume_sum"] +
        df["p6"] * df["bid_volume_sum"]
    ) / (df["volume_total"] + 1e-6)

    # ── Depth imbalance ──────────────────────────────────────────
    df["depth_imbalance"]    = (df["v0"] - df["v6"]) / (df["v0"] + df["v6"] + 1e-6)
    df["depth_imbalance_L2"] = (df["v1"] - df["v7"]) / (df["v1"] + df["v7"] + 1e-6)
    df["depth_imbalance_L3"] = (df["v2"] - df["v8"]) / (df["v2"] + df["v8"] + 1e-6)

    w = np.array([1.0, 0.5, 0.25])
    bid_w = (bid_vols[:, :3] * w).sum(axis=1)
    ask_w = (ask_vols[:, :3] * w).sum(axis=1)
    df["weighted_depth_imbalance"] = (bid_w - ask_w) / (bid_w + ask_w + 1e-6)

    # ── VWAP ─────────────────────────────────────────────────────
    df["bid_vwap"]    = (bid_prices * bid_vols).sum(axis=1) / (df["bid_volume_sum"] + 1e-6)
    df["ask_vwap"]    = (ask_prices * ask_vols).sum(axis=1) / (df["ask_volume_sum"] + 1e-6)
    df["vwap_spread"] = df["ask_vwap"] - df["bid_vwap"]
    df["vwap_vs_mid"] = ((df["bid_vwap"] + df["ask_vwap"]) / 2) - df["midprice"]

    # ── Book pressure ────────────────────────────────────────────
    cum_bid_3 = bid_vols[:, :3].sum(axis=1)
    cum_ask_3 = ask_vols[:, :3].sum(axis=1)
    df["book_pressure"] = (cum_bid_3 - cum_ask_3) / (cum_bid_3 + cum_ask_3 + 1e-6)

    # ── Trade features ───────────────────────────────────────────
    trade_prices = df[[f"dp{i}" for i in range(4)]].values
    trade_vols   = df[[f"dv{i}" for i in range(4)]].values

    df["trade_price_mean"]  = trade_prices.mean(axis=1)
    df["trade_volume_sum"]  = trade_vols.sum(axis=1)
    df["trade_volume_mean"] = trade_vols.mean(axis=1)
    df["trade_vs_mid"]      = df["trade_price_mean"] - df["midprice"]
    df["trade_direction_imbalance"] = np.where(
        df["trade_price_mean"] > df["midprice"],
        df["trade_volume_sum"], -df["trade_volume_sum"]
    ) / (df["trade_volume_sum"] + 1e-6)

    # ── Sequence rolling features ────────────────────────────────
    grp = df.groupby("seq_ix")

    df["midprice_change"]   = grp["midprice"].diff().fillna(0)
    df["microprice_change"] = grp["microprice"].diff().fillna(0)
    df["spread_change"]     = grp["spread"].diff().fillna(0)
    df["imbalance_change"]  = grp["volume_imbalance"].diff().fillna(0)

    for win in [5, 10, 20]:
        df[f"midprice_mean_{win}"]    = (
            grp["midprice"].rolling(win).mean()
            .reset_index(level=0, drop=True).fillna(0)
        )
        df[f"midprice_std_{win}"]     = (
            grp["midprice"].rolling(win).std()
            .reset_index(level=0, drop=True).fillna(0)
        )
        df[f"microprice_mean_{win}"]  = (
            grp["microprice"].rolling(win).mean()
            .reset_index(level=0, drop=True).fillna(0)
        )
        df[f"volume_mean_{win}"]      = (
            grp["volume_total"].rolling(win).mean()
            .reset_index(level=0, drop=True).fillna(0)
        )
        df[f"imbalance_mean_{win}"]   = (
            grp["volume_imbalance"].rolling(win).mean()
            .reset_index(level=0, drop=True).fillna(0)
        )

    df["midprice_return_5"]  = grp["midprice"].pct_change(5).fillna(0)
    df["midprice_return_10"] = grp["midprice"].pct_change(10).fillna(0)

    # ── Normalised ───────────────────────────────────────────────
    df["spread_norm"]     = df["spread"]     / (df["midprice"] + 1e-6)
    df["microprice_norm"] = df["microprice"] / (df["midprice"] + 1e-6)
    df["volume_norm"]     = df["volume_total"] / (
        grp["volume_total"].transform("mean") + 1e-6
    )

    # ── Slope ────────────────────────────────────────────────────
    df["microprice_slope"] = grp["microprice"].diff().fillna(0)
    df["volume_slope"]     = grp["volume_total"].diff().fillna(0)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df


def generate_features_t1(df: pd.DataFrame) -> pd.DataFrame:
    """
    CLEAN, MINIMAL feature set specifically tuned for t1.

    Key insight: t1 is a longer-horizon, noisier target.
    It responds better to:
      - Slower-moving signals (longer windows: 10, 20, 50)
      - Clean imbalance measures (not too many correlated variants)
      - Trade direction — who is aggressing the market
      - Longer-horizon momentum (returns over 10, 20, 50 steps)

    Do NOT use: interaction terms, EWM, L2/L3 depth, too many features.
    These overfit on t1 and hurt generalisation.
    """

    df = df.copy()

    bid_prices = df[[f"p{i}" for i in range(6)]].values
    ask_prices = df[[f"p{i}" for i in range(6, 12)]].values
    bid_vols   = df[[f"v{i}" for i in range(6)]].values
    ask_vols   = df[[f"v{i}" for i in range(6, 12)]].values

    # ── Core price ───────────────────────────────────────────────
    df["midprice"]    = (df["p0"] + df["p6"]) / 2
    df["spread"]      = df["p6"] - df["p0"]
    df["spread_norm"] = df["spread"] / (df["midprice"] + 1e-6)

    # ── Volume imbalance ─────────────────────────────────────────
    df["bid_volume_sum"] = bid_vols.sum(axis=1)
    df["ask_volume_sum"] = ask_vols.sum(axis=1)
    df["volume_total"]   = df["bid_volume_sum"] + df["ask_volume_sum"]
    df["volume_imbalance"] = (
        df["bid_volume_sum"] - df["ask_volume_sum"]
    ) / (df["volume_total"] + 1e-6)

    df["depth_imbalance_L1"] = (
        df["v0"] - df["v6"]
    ) / (df["v0"] + df["v6"] + 1e-6)

    # ── Microprice ───────────────────────────────────────────────
    df["microprice"] = (
        df["p0"] * df["ask_volume_sum"] +
        df["p6"] * df["bid_volume_sum"]
    ) / (df["volume_total"] + 1e-6)
    df["microprice_norm"] = df["microprice"] / (df["midprice"] + 1e-6)

    # ── VWAP ─────────────────────────────────────────────────────
    df["bid_vwap"]    = (bid_prices * bid_vols).sum(axis=1) / (df["bid_volume_sum"] + 1e-6)
    df["ask_vwap"]    = (ask_prices * ask_vols).sum(axis=1) / (df["ask_volume_sum"] + 1e-6)
    df["vwap_vs_mid"] = ((df["bid_vwap"] + df["ask_vwap"]) / 2) - df["midprice"]

    # ── Trade direction (most important for t1) ──────────────────
    trade_prices = df[[f"dp{i}" for i in range(4)]].values
    trade_vols   = df[[f"dv{i}" for i in range(4)]].values

    df["trade_price_mean"] = trade_prices.mean(axis=1)
    df["trade_volume_sum"] = trade_vols.sum(axis=1)
    df["trade_vs_mid"]     = df["trade_price_mean"] - df["midprice"]

    # Signed volume: positive = buyers aggressing
    df["signed_trade_volume"] = (
        np.sign(df["trade_vs_mid"]) * df["trade_volume_sum"]
    )

    # ── Long rolling windows ─────────────────────────────────────
    grp = df.groupby("seq_ix")

    df["midprice_change"] = grp["midprice"].diff().fillna(0)

    for win in [10, 20, 50]:
        df[f"midprice_mean_{win}"] = (
            grp["midprice"].rolling(win).mean()
            .reset_index(level=0, drop=True).fillna(0)
        )
        df[f"imbalance_mean_{win}"] = (
            grp["volume_imbalance"].rolling(win).mean()
            .reset_index(level=0, drop=True).fillna(0)
        )
        df[f"signed_trade_sum_{win}"] = (
            grp["signed_trade_volume"].rolling(win).sum()
            .reset_index(level=0, drop=True).fillna(0)
        )
        df[f"microprice_mean_{win}"] = (
            grp["microprice"].rolling(win).mean()
            .reset_index(level=0, drop=True).fillna(0)
        )

    # Longer-horizon returns
    df["return_10"] = grp["midprice"].pct_change(10).fillna(0)
    df["return_20"] = grp["midprice"].pct_change(20).fillna(0)
    df["return_50"] = grp["midprice"].pct_change(50).fillna(0)

    # Cumulative trade pressure over 20 steps
    df["trade_pressure_20"] = (
        grp["signed_trade_volume"].rolling(20).sum()
        .reset_index(level=0, drop=True).fillna(0)
    )

    # Volume norm
    df["volume_norm"] = df["volume_total"] / (
        grp["volume_total"].transform("mean") + 1e-6
    )

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df