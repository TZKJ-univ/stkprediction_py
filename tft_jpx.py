#!/usr/bin/env python3
"""
JPX 銘柄を月次に集約し Temporal Fusion Transformer で
22 営業日先 (≒1か月) の終値リターンを予測する。
"""

import warnings, sys, yfinance as yf, pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import QuantileRegression

CODES_FILE = Path("jpx_codes.txt")
PERIOD, INTERVAL = "10y", "1d"
AHEAD = 22                # 1 month (bus. days)
BATCH = 150               # yfinance chunk

warnings.filterwarnings("ignore")

# ---------- データ取得 ----------
codes = [
    f"{c.strip().zfill(4)}.T" for c in CODES_FILE.read_text().splitlines() if c.strip()
]
chunks = [codes[i : i + BATCH] for i in range(0, len(codes), BATCH)]
close = pd.DataFrame()
for ch in tqdm(chunks, desc="yfinance"):
    df = yf.download(" ".join(ch), period=PERIOD, interval=INTERVAL, progress=False)["Close"]
    close = pd.concat([close, df], axis=1)

# 前処理: 月次・対数リターン
close = close.resample("1M").last().ffill()
rets  = np.log(close / close.shift(1)).dropna()

# --- TimeSeries 化。データが空の銘柄を除外し、各銘柄ごとに独立スケーリング ---
series, scalers, series_scaled = {}, {}, {}
for t in rets.columns:
    s = rets[t].dropna()
    if len(s) < 12:          # 最低 1 年分（月次12点）ない銘柄はスキップ
        continue
    ts = TimeSeries.from_series(s)
    sc = Scaler()
    series_scaled[t] = sc.fit_transform(ts)
    series[t] = ts
    scalers[t] = sc

print(f"Valid series after filtering: {len(series)}")

if not series:
    sys.exit("No valid time‐series after filtering.")

# ---------- データ分割 ----------
train, val = {}, {}
for t, s in series_scaled.items():
    train[t], val[t] = s[:-AHEAD], s[-AHEAD:]

# ---------- TFT モデル ----------
model = TFTModel(
    input_chunk_length=24,
    output_chunk_length=AHEAD,
    hidden_size=32,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=256,
    likelihood=QuantileRegression(),
    random_state=42,
)

# ---------- 学習 (全系列を一括) ----------
model.fit(
    list(train.values()),
    epochs=50,
    verbose=True,
    val_series=list(val.values()),
)

# ---------- 予測 ----------
preds = {}
for t, s in series_scaled.items():
    fut = model.predict(n=AHEAD, series=s)
    # 22 日後の予測リターン → 終値
    ret_pred = scalers[t].inverse_transform(fut)  # log-return
    last_px  = close[t].iloc[-1]
    pred_px  = last_px * np.exp(ret_pred.values()[-1, 0])
    preds[t] = {"Current": last_px, "Predicted": pred_px}

df_out = (
    pd.DataFrame(preds).T.assign(Ratio=lambda d: (d.Predicted - d.Current) / d.Current)
    .query("Ratio > 0")
    .sort_values("Ratio", ascending=False)
    .head(20)
)

print(df_out.to_string(formatters={"Current": "{:.2f}".format,
                                   "Predicted":"{:.2f}".format,
                                   "Ratio":"{:.2%}".format}))
df_out.to_csv("tft_result.csv", index=True)