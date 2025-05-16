import pandas as pd
from scipy.optimize import minimize
import math
def walk_forward_splits(df: pd.DataFrame, date_col: str, n_splits: int):
    """
    Generate walk‑forward train/validation index splits based on a date column.
    """
    dates = pd.to_datetime(df[date_col]).sort_values().unique()
    n = len(dates)
    window = n // (n_splits + 1)
    splits = []
    for i in range(1, n_splits + 1):
        train_end = dates[i * window]
        valid_start = train_end + pd.Timedelta(days=1)
        valid_end = dates[(i + 1) * window] if i < n_splits else dates[-1]
        train_idx = df.index[pd.to_datetime(df[date_col]) <= train_end]
        valid_idx = df.index[
            (pd.to_datetime(df[date_col]) > train_end) &
            (pd.to_datetime(df[date_col]) <= valid_end)
        ]
        splits.append((train_idx, valid_idx))
    return splits
#!/usr/bin/env python3
import argparse, sys, time
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count
import os

import pandas as pd, numpy as np, yfinance as yf, json
import json

try:
    import torch
    import torch.nn as nn
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None

# ---- device selection ----
if torch is None:
    device = torch.device("cpu")
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon GPU (Metal Performance Shaders)
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
print(f"[INFO] Using device: {device}")

def prepare_lstm_sequences(features_df: pd.DataFrame, window_size: int = 60, shift: int = 22):
    sequences, targets, idxs = [], [], []
    for ticker, df_t in features_df.groupby('Ticker'):
        df_t = df_t.sort_values('Date')
        indices = df_t.index.to_list()
        X = df_t.drop(['Date', 'Ticker', 'target'], axis=1).values
        y = df_t['target'].values
        for i in range(window_size, len(X) - shift + 1):
            seq = X[i - window_size:i]
            label = y[i + shift - 1]
            idx = indices[i + shift - 1]
            sequences.append(seq)
            targets.append(label)
            idxs.append(idx)
    return np.array(sequences), np.array(targets), np.array(idxs)

class LSTMForecast(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --------- TransformerForecast class ---------
class TransformerForecast(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)            # (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)           # (seq_len, batch, d_model)
        x = self.encoder(x)              # (seq_len, batch, d_model)
        x = x[-1, :, :]                  # last time step (batch, d_model)
        return self.fc(x)                # (batch, 1)
from tqdm import tqdm

# Suppress pandas future warnings and general user warnings
import warnings, logging
import gc
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

FEAT_COLS: list[str] = []   # features order will be stored after first build

CODES_FILE = Path("jpx_codes_full.txt")
DATA_DIR   = Path("feather"); DATA_DIR.mkdir(exist_ok=True)
PERIOD, INTERVAL, CHUNK = "10y", "1d", 200
LAGS, SHIFT = [1,5,22,66], 22          # 1か月 = 22営業日
TRAIN_SPLIT_YEARS = 0.5   # 最後の0.5年（6ヶ月）を検証用に使う

# 記憶保持年数（コード内に固定）
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "5"))

SECTOR_FILE = Path("sectors.json")
SECTOR_DICT: dict[str, str] = {}      # filled in main()

def load_sector_map(tickers: list[str]) -> dict[str, str]:
    """
    Return {ticker: sector}.  Unknown → "Unknown".
    Cache to sectors.json so we do not hit Yahoo every run.
    """
    mp = {}
    if SECTOR_FILE.exists():
        try:
            mp = json.loads(SECTOR_FILE.read_text())
        except Exception:
            pass
    updated = False
    for t in tickers:
        if t not in mp:
            try:
                info = yf.Ticker(t).info
                mp[t] = info.get("sector", "Unknown") or "Unknown"
            except Exception:
                mp[t] = "Unknown"
            updated = True
    if updated:
        SECTOR_FILE.write_text(json.dumps(mp, ensure_ascii=False))
    return mp

SECTOR_FILE = Path("sectors.json")
FUND_FILE   = Path("fundamentals.json")

# --- 外生変数（為替・指数など） ---
EXOGENOUS_TICKERS = ["JPY=X", "^N225", "^VIX", "^TOPX"]

def load_fundamentals(tickers: list[str]) -> dict[str, dict]:
    """
    Return {ticker: {"PER": ..., "PBR": ..., "DivYield": ...}}.
    Cache to fundamentals.json to avoid repeated calls.
    """
    mp = {}
    if FUND_FILE.exists():
        try:
            mp = json.loads(FUND_FILE.read_text())
        except Exception:
            pass
    updated = False
    for t in tickers:
        if t not in mp:
            try:
                info = yf.Ticker(t).info
                # capture all numeric fundamentals and quoteType
                num_info = {k: v for k, v in info.items() if isinstance(v, (int, float))}
                num_info["quoteType"] = info.get("quoteType", "")
                mp[t] = num_info
            except Exception:
                mp[t] = {"quoteType": ""}
            updated = True
    if updated:
        FUND_FILE.write_text(json.dumps(mp, ensure_ascii=False))
    return mp


# --- 外生変数データの取得 ---
def load_exogenous(tickers: list[str], start: str, end: str, interval: str) -> pd.DataFrame:
    """
    Fetch Close prices for exogenous tickers and return a DataFrame indexed by Date.
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    # build a DataFrame of close prices
    data = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            sub = df[t]
            col = "Close" if "Close" in sub else "Adj Close"
            data[t] = sub[col]
    else:
        col = "Close" if "Close" in df else "Adj Close"
        data[tickers[0]] = df[col]
    exog = pd.DataFrame(data)
    exog.index = pd.to_datetime(exog.index)
    return exog.ffill()


def codes() -> list[str]:
    """
    Read jpx_codes.txt and return list like ["7203.T", "6758.T", ...]
    Only keep lines that are 4–5 digit numbers.
    """
    if not CODES_FILE.exists():
        sys.exit("jpx_codes.txt not found")
    out = []
    for line in CODES_FILE.read_text().splitlines():
        code = line.strip()
        if code.isdigit() and 4 <= len(code) <= 5:
            out.append(f"{code.zfill(4)}.T")
    if not out:
        sys.exit("No valid codes")
    return out

def update_ticker_csv(ticker: str, max_retry: int = 5):
    # DEPRECATED:
    """
    Keep <DATA_DIR>/<ticker>.csv with historical Close prices.
    Retry with exponential back‑off when yfinance hits rate‑limit (429/401).
    """
    fcsv = DATA_DIR / f"{ticker}.csv"
    start = "2015-01-01"
    if fcsv.exists():
        last = pd.read_csv(fcsv, usecols=["Date"]).Date.max()
        start = (pd.to_datetime(last) + pd.Timedelta(days=1)).date()

    for r in range(max_retry):
        try:
            if fcsv.exists():
                df = yf.download(
                    ticker,
                    start=start,
                    interval=INTERVAL,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            else:
                # 初回は期間指定で広めに取得
                df = yf.download(
                    ticker,
                    period=PERIOD,
                    interval=INTERVAL,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            if not df.empty:
                break  # success
        except Exception:
            pass  # fall through to retry
        wait = 2 ** r
        time.sleep(wait)
    else:
        return  # all retries failed or empty

    # --- Close 列を Series として取得 ---
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close" in df.columns.get_level_values(1)):
            s = df.xs("Close", level=1, axis=1)
        elif ("Adj Close" in df.columns.get_level_values(1)):
            s = df.xs("Adj Close", level=1, axis=1)
        else:
            return
        series = s.squeeze().dropna()
    else:
        col = "Close" if "Close" in df else "Adj Close"
        series = df[col].dropna()

    if series.empty:
        return

    # -- Create tidy DataFrame ---
    if isinstance(series, pd.Series):
        price = series.to_frame(name="Close").reset_index()
    else:  # just in case series returned as DataFrame with single column
        price = series.reset_index().rename(columns={series.columns[0]: "Close"})

    # Ensure date column name
    if price.columns[0] != "Date":
        price.rename(columns={price.columns[0]: "Date"}, inplace=True)

    price.insert(1, "Ticker", ticker)
    price["Date"] = pd.to_datetime(price["Date"]).dt.strftime("%Y-%m-%d")

    if fcsv.exists():
        (pd.concat([pd.read_csv(fcsv), price])
         .drop_duplicates(subset=["Date"])
         .to_csv(fcsv, index=False))
    else:
        price.to_csv(fcsv, index=False)

def update_chunk(chunk: list[str], idx: int):
    # Prepare list of tickers to fetch and accumulator
    to_fetch = chunk.copy()
    acc_df = None
    f = DATA_DIR / f"{idx}.feather"

    start = "2015-01-01"
    if f.exists():
        last = pd.read_feather(f, columns=["Date"]).Date.max()
        start = (pd.to_datetime(last) + pd.Timedelta(days=1)).date()
        today = datetime.now().date()
        if start > today:
            # up to date; nothing new to fetch
            return

    max_retry = 5
    for r in range(max_retry):
        try:
            # Download only the tickers still missing
            df_chunk = yf.download(
                tickers=to_fetch,
                start=start,
                interval=INTERVAL,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=False,
            )
            # Treat empty DataFrame as failure
            if df_chunk.empty:
                raise ValueError("Empty data for chunk")
            # Accumulate fetched data
            if acc_df is None:
                acc_df = df_chunk
            else:
                acc_df = pd.concat([acc_df, df_chunk], axis=1)
            # Determine which tickers were successfully fetched
            if isinstance(df_chunk.columns, pd.MultiIndex):
                available = df_chunk.columns.get_level_values(0).unique().tolist()
            else:
                available = to_fetch.copy()
            missing = list(set(to_fetch) - set(available))
            # If some tickers are still missing, retry those
            if missing and r < max_retry - 1:
                to_fetch = missing
                time.sleep(2 ** r)
                continue
            # All requested tickers fetched (or no more retries)
            break
        except Exception:
            # Exponential backoff on any error
            time.sleep(2 ** r)
    else:
        # All retries failed
        return
    # Use accumulated DataFrame for downstream processing
    df = acc_df

    # ----- Close が無いときは Adj Close を使う -----
    if isinstance(df.columns, pd.MultiIndex):
        lvl1 = df.columns.get_level_values(1)
        if "Close" in lvl1:
            closes = df.xs("Close", level=1, axis=1)
        elif "Adj Close" in lvl1:
            closes = df.xs("Adj Close", level=1, axis=1)
        else:
            return
        closes = (
            closes.stack()
            .reset_index()
            .rename(columns={0: "Close"})
        )
    else:
        closes = df[["Close"]] if "Close" in df else pd.DataFrame()
        if closes.empty and "Adj Close" in df:
            closes = df[["Adj Close"]].rename(columns={"Adj Close": "Close"})
        if closes.empty:
            return
        closes = (
            closes.reset_index()
            .rename(columns={"index": "Date"})
            .assign(Ticker=chunk[0])
        )

    closes.columns = ["Date", "Ticker", "Close"]

    # ----------- 保存 -----------
    if f.exists():
        (pd.concat([pd.read_feather(f), closes])
           .drop_duplicates(subset=["Date", "Ticker"])
           .to_feather(f))
    else:
        closes.to_feather(f)

def price_matrix(cds: list[str]) -> pd.DataFrame:
    """Build Close matrix from feather/<idx>.feather files created by update_chunk."""
    files = list(DATA_DIR.glob("*.feather"))
    if not files:
        sys.exit("no feather data")
    df = pd.concat(pd.read_feather(f) for f in files)
    # Filter to last HISTORY_YEARS to reduce memory before pivot
    df['Date'] = pd.to_datetime(df['Date'])
    cutoff = df['Date'].max() - pd.DateOffset(years=HISTORY_YEARS)
    df = df[df['Date'] >= cutoff]
    # Remove any duplicate Date/Ticker rows before pivot
    df = df.drop_duplicates(subset=["Date", "Ticker"])
    mat = (
        df.pivot(index="Date", columns="Ticker", values="Close")
          .sort_index()
          .ffill()
    )
    # ensure index is datetime for train/test split
    mat.index = pd.to_datetime(mat.index)
    return mat

def features(mat: pd.DataFrame, funds: dict[str, dict], exog: pd.DataFrame) -> pd.DataFrame:
    # 基本ラグ
    feats = {f"lag_{l}": mat.shift(l) for l in LAGS}

    # 移動平均・ボリンジャーバンド
    feats["sma_22"] = mat.rolling(22).mean()
    feats["sma_50"] = mat.rolling(50).mean()
    rolling_std     = mat.rolling(22).std()
    feats["bb_width"] = (rolling_std * 2) / feats["sma_22"]   # バンド幅比率

    # ボラティリティ(22d)
    feats["vol_22"] = mat.pct_change().rolling(22).std()
    # 価格と移動平均の比率
    feats["price_sma22_ratio"] = mat / feats["sma_22"]
    feats["price_sma50_ratio"] = mat / feats["sma_50"]

    # MACD (12,26)
    ema12 = mat.ewm(span=12, adjust=False).mean()
    ema26 = mat.ewm(span=26, adjust=False).mean()
    feats["macd"] = ema12 - ema26

    # RSI
    feats["rsi_14"] = 100 - 100 / (1 + mat.pct_change().rolling(14).mean())

    X = pd.concat(feats, axis=1).stack(future_stack=True).dropna().reset_index()
    # ensure Date column is datetime64 for merge
    X["Date"] = pd.to_datetime(X["Date"])
    X["dayofweek"] = pd.to_datetime(X["Date"]).dt.dayofweek.astype("int8")
    X["month"]     = pd.to_datetime(X["Date"]).dt.month.astype("int8")

    # --- 業種をカテゴリとして追加 ---
    global SECTOR_DICT
    X["Sector"] = X["Ticker"].map(SECTOR_DICT).fillna("Unknown")

    # --- map static fundamentals ---
    # map all numeric fundamentals dynamically (use all available yfinance indicators)
    if funds:
        # collect the union of all numeric fundamental keys except "quoteType"
        all_keys = sorted({k for v in funds.values() for k in v.keys() if k != "quoteType"})
        for key in all_keys:
            X[key] = X["Ticker"].map(lambda t, k=key: funds.get(t, {}).get(k, 0))
        # Drop quoteType column if present
        X = X.drop(columns=["quoteType"], errors="ignore")

    # 配当利回り
    if 'dividendYield' in X.columns:
        X['DividendYield'] = X['dividendYield'].fillna(0)
    else:
        X['DividendYield'] = 0

    # ROE
    if 'returnOnEquity' in X.columns:
        X['ROE'] = X['returnOnEquity'].fillna(0)
    else:
        X['ROE'] = 0

    # 自己資本比率 = (totalAssets - totalLiab) / totalAssets
    if 'totalAssets' in X.columns and 'totalLiab' in X.columns:
        X['EquityRatio'] = ((X['totalAssets'] - X['totalLiab']) / X['totalAssets']).fillna(0)
    else:
        X['EquityRatio'] = 0

    # 営業益成長率
    if 'earningsQuarterlyGrowth' in X.columns:
        X['OpIncomeGrowth'] = X['earningsQuarterlyGrowth'].fillna(0)
    else:
        X['OpIncomeGrowth'] = 0

    # --- 外生変数を特徴量に追加（高速マージ） ---
    exog_df = exog.reset_index().rename(columns={'index': 'Date'})
    X = X.merge(exog_df, on='Date', how='left')
    X.fillna(method='ffill', inplace=True)
    X.fillna(0, inplace=True)

    y = (
        (np.log(mat.shift(-SHIFT)) - np.log(mat))
        .stack()
        .reindex(X.set_index(["Date", "Ticker"]).index)
    )
    X["target"] = y.values
    # ±100% (= 約 ±0.693 in log) 以上の外れ値を除外
    X = X[X["target"].abs() <= 1.0]
    # downcast float64 to float32 to reduce memory footprint
    for col in X.select_dtypes(include=['float64']).columns:
        X[col] = X[col].astype('float32')
    # keep rows even if fundamentals have missing (filled with 0), only drop if target missing
    return X[X["target"].notna()]


# --- Transformer-based global model ---

class StockPriceTFT(nn.Module):
    """Simplified Temporal Fusion‑style model: Embedding → LSTM → Self‑Attention → Linear."""
    def __init__(self, num_tickers: int, input_dim: int, d_model: int = 128, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(num_tickers, d_model)
        self.lstm  = nn.LSTM(input_dim, d_model, batch_first=True, dropout=dropout)
        enc_layer  = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(enc_layer, num_layers=1)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq: torch.Tensor, ticker_id: torch.Tensor): # type: ignore
        # x_seq shape: (batch, seq_len, input_dim)
        # ticker_id shape: (batch,)
        h0 = self.embed(ticker_id).unsqueeze(0)          # (1, batch, d_model)
        c0 = torch.zeros_like(h0)                        # same shape as h0
        lstm_out, _ = self.lstm(x_seq, (h0, c0))         # (batch, seq_len, d_model)
        attn_out = self.transformer(lstm_out)            # (batch, seq_len, d_model)
        out = self.fc(self.dropout(attn_out[:, -1, :]))  # (batch, 1)
        return out


def build_dataset(mat: pd.DataFrame, window_size: int = 60, shift: int = 22):
    """Return tensors (X_seq, y, ticker_ids) for all tickers."""
    Xs, ys, tids = [], [], []
    tickers = mat.columns.tolist()
    for tid, tkr in enumerate(tickers):
        arr = mat[tkr].values
        for i in range(window_size, len(arr) - shift + 1):
            raw_window = arr[i - window_size:i]
            raw_label = arr[i + shift - 1]
            # Skip NaNs
            if np.isnan(raw_window).any() or np.isnan(raw_label):
                continue
            base = raw_window[-1]
            # avoid invalid base
            if base <= 0:
                continue
            # ---- log-return normalization ----
            ratio = raw_window / base
            if np.any(ratio <= 0):
                continue
            window = np.log(ratio)
            label_ratio = raw_label / base
            if label_ratio <= 0:
                continue
            label = np.log(label_ratio)
            # skip targets outside ±0.5 to remove extreme outliers
            if abs(label) > 0.5:
                continue
            Xs.append(window)
            ys.append(label)
            tids.append(tid)
    Xs = torch.tensor(Xs, dtype=torch.float32).unsqueeze(-1)  # (N, win, 1)
    ys = torch.tensor(ys, dtype=torch.float32).view(-1, 1)
    tids = torch.tensor(tids, dtype=torch.long)
    return Xs, ys, tids

def main():
    p = argparse.ArgumentParser(); p.add_argument("--csv",default="result.csv")
    args = p.parse_args()
    start_time = time.time()

    cds = codes()
    # ETF/REIT を除外（quoteType "ETF" または "REIT" のもののみ除外）
    FUNDAMENTALS = load_fundamentals(cds)
    filtered_cds = []
    for t in cds:
        qt = FUNDAMENTALS.get(t, {}).get("quoteType", "") or ""
        if qt.upper() not in ("ETF", "REIT"):
            filtered_cds.append(t)
    # 万が一全て除外された場合は元リストを使用
    if not filtered_cds:
        filtered_cds = cds.copy()
    cds = filtered_cds
    global SECTOR_DICT
    SECTOR_DICT = load_sector_map(cds)

    for i in tqdm(range(0, len(cds), CHUNK), desc="download"):
        update_chunk(cds[i:i + CHUNK], i // CHUNK)

    mat = price_matrix(cds)
    # メモリ節約: 過去5年分の履歴に限定
    cutoff = mat.index.max() - pd.DateOffset(years=HISTORY_YEARS)
    mat = mat[mat.index >= cutoff]

    # --- 最新日の終値が NaN のティッカーを除外し、残数を表示 ---
    before_cnt = mat.shape[1]
    mat = mat.loc[:, ~mat.tail(1).isna().squeeze()]
    after_cnt = mat.shape[1]
    print(f"[INFO] Dropped {before_cnt - after_cnt} tickers with NaN latest price; {after_cnt} remain.")

    gc.collect()

    # ---- Transformer‑based global model ----
    # Build supervised tensors for all tickers
    X_all, y_all, tids_all = build_dataset(mat, window_size=60, shift=SHIFT)
    # --- 標準化: 入力特徴量(X_all)とターゲット(y_all)を平均0分散1にスケーリング ---
    X_mean = X_all.mean()
    X_std = X_all.std()
    y_all_mean = y_all.mean().item()
    y_all_std = y_all.std().item()
    X_all = (X_all - X_mean) / X_std
    y_all = (y_all - y_all_mean) / y_all_std
    full_ds = TensorDataset(X_all, tids_all, y_all)
    full_dl = DataLoader(full_ds, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    eval_dl = DataLoader(full_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)

    model = StockPriceTFT(num_tickers=len(mat.columns), input_dim=1, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        for xb, tidb, yb in full_dl:
            xb, tidb, yb = xb.to(device), tidb.to(device), yb.to(device)
            # ---- debug: detect NaNs in inputs ----
            if torch.isnan(xb).any() or torch.isnan(yb).any():
                raise RuntimeError(f"NaN detected in input features or targets at epoch {epoch+1}")
            optimizer.zero_grad()
            pred = model(xb, tidb)
            # ---- debug: detect NaNs in model output ----
            if torch.isnan(pred).any():
                raise RuntimeError(f"NaN detected in model output at epoch {epoch+1}")
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        # --- Epoch-end evaluation: preview top-10 ratios ---
        model.eval()
        epoch_rows = []
        with torch.no_grad():
            for tid, tkr in enumerate(mat.columns):
                seq = torch.tensor(mat[tkr].values[-60:], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
                pred_price = model(seq, torch.tensor([tid], device=device)).item()
                cur_price  = mat[tkr].values[-1]
                ratio = pred_price / cur_price
                epoch_rows.append((tkr, ratio))  # collect all; we'll sort later
        top10 = sorted(epoch_rows, key=lambda x: x[1], reverse=True)[:10]
        if not top10:
            print(f"[Epoch {epoch+1}/10] MSE={float('nan'):.6f} | R2={float('nan'):.4f} | Top‑10 → (no predictions yet)")
            continue
        top10_str = "; ".join(f"{t}:{r:.2%}" for t, r in top10)

        # ---- MSE & R2 over all samples ----
        criterion_eval = nn.MSELoss()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, tidb, yb in eval_dl:
                xb, tidb, yb = xb.to(device), tidb.to(device), yb.to(device)
                preds = model(xb, tidb)
                all_preds.append(preds.cpu())
                all_targets.append(yb.cpu())
        if all_preds and all_targets:
            y_pred_tensor = torch.cat(all_preds, dim=0)
            y_true_tensor = torch.cat(all_targets, dim=0)
            mse = criterion_eval(y_pred_tensor, y_true_tensor).item()
            # R2 score: 1 - SS_res / SS_tot
            mean_true = y_true_tensor.mean()
            ss_res = ((y_true_tensor - y_pred_tensor) ** 2).sum().item()
            ss_tot = ((y_true_tensor - mean_true) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        else:
            mse = float("nan")
            r2 = float("nan")

        print(f"[Epoch {epoch+1}/10] MSE={mse:.6f} | R2={r2:.4f} | Top‑10 → {top10_str}")

    # ---- Final evaluation on full dataset ----
    criterion_eval = nn.MSELoss()
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for xb, tidb, yb in eval_dl:
            xb, tidb, yb = xb.to(device), tidb.to(device), yb.to(device)
            preds = model(xb, tidb)
            all_preds.append(preds.cpu())
            all_targets.append(yb.cpu())
    if all_preds and all_targets:
        y_pred_tensor = torch.cat(all_preds, dim=0)
        y_true_tensor = torch.cat(all_targets, dim=0)
        mse_final = criterion_eval(y_pred_tensor, y_true_tensor).item()
        mean_true = y_true_tensor.mean()
        ss_res = ((y_true_tensor - y_pred_tensor) ** 2).sum().item()
        ss_tot = ((y_true_tensor - mean_true) ** 2).sum().item()
        r2_final = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        mse_final, r2_final = float("nan"), float("nan")
    print(f"[Final] MSE={mse_final:.6f} | R2={r2_final:.4f}")

    # ---- 推論＆結果整形 ----
    model.eval()
    rows = []
    with torch.no_grad():
        for tid, tkr in enumerate(mat.columns):
            # 学習時と同じ正規化を推論部でも行う
            raw_window = mat[tkr].values[-60:]
            base = raw_window[-1]
            if base <= 0:
                continue
            ratio = raw_window / base
            if np.any(ratio <= 0):
                continue
            window = np.log(ratio)
            # apply same normalization as training
            seq = torch.tensor((window - X_mean.item()) / X_std.item(), dtype=torch.float32, device=device) \
                       .unsqueeze(0).unsqueeze(-1)

            # モデル出力（正規化対数リターン）を取得し、逆正規化してクリップ
            pred_norm = model(seq, torch.tensor([tid], device=device)).item()
            # de-normalize prediction back to log-return and clip to ±0.5
            pred_log_return = pred_norm * y_all_std + y_all_mean
            pred_log_return = max(min(pred_log_return, 0.5), -0.5)

            # 1ヶ月リターンに変換
            month_ratio = math.exp(pred_log_return) - 1
            cur_price   = base
            pred_price  = cur_price * math.exp(pred_log_return)
            rows.append((tkr, cur_price, pred_price, month_ratio))

    # --- 単価フィルタを追加（例：100円未満を除外） ---
    MIN_PRICE_YEN = 100
    df_all = pd.DataFrame(rows, columns=["Ticker", "Current", "Predicted", "MonthlyReturn"])
    df_all = df_all[df_all["Current"] >= MIN_PRICE_YEN]
    df_out = (
        df_all
        .sort_values("MonthlyReturn", ascending=False)
        .head(20)
    )
    df_out.to_csv(args.csv, index=False)
    print(df_out.to_string(index=False, formatters={
        "Current": "{:.2f}".format,
        "Predicted": "{:.2f}".format,
        "MonthlyReturn": "{:.2%}".format,
    }))
    elapsed = time.time() - start_time
    print(f"[INFO] Elapsed time: {elapsed:.2f} seconds")

if __name__=="__main__":
    main()