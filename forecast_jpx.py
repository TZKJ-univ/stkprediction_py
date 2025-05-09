import pandas as pd
from scipy.optimize import minimize
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

import pandas as pd, numpy as np, yfinance as yf, lightgbm as lgb, json
import json
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
except ImportError:
    torch = None

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
from sklearn.metrics import mean_squared_error, r2_score

from tqdm import tqdm

# Suppress pandas future warnings and general user warnings
import warnings, logging
import gc
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Reduce verbosity for Optuna, LightGBM, XGBoost, and CatBoost
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)
logging.getLogger("catboost").setLevel(logging.ERROR)

FEAT_COLS: list[str] = []   # features order will be stored after first build

CODES_FILE = Path("jpx_codes_full.txt")
DATA_DIR   = Path("feather"); DATA_DIR.mkdir(exist_ok=True)
PERIOD, INTERVAL, CHUNK = "10y", "1d", 200
LAGS, SHIFT = [1,5,22,66], 22          # 1か月 = 22営業日
TRAIN_SPLIT_YEARS = 0.5   # 最後の0.5年（6ヶ月）を検証用に使う

# 記憶保持年数（コード内に固定）
HISTORY_YEARS = 10

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

def _prep_xy(df: pd.DataFrame, cat: list[str]):
    """return X(num) for xgb, X(cat) for catboost, y, cat_idx, w"""
    global FEAT_COLS
    # Drop non-feature columns including quoteType if present
    X = df.drop(["Date", "target", "quoteType"], axis=1, errors="ignore")
    if not FEAT_COLS:
        FEAT_COLS = X.columns.tolist()
    X = X[FEAT_COLS]          # 固定順
    y = df["target"].values
    # --- xgb 用: cat→int ---
    X_num = X.copy()
    for c in cat:
        X_num[c] = X_num[c].cat.codes
    # --- catboost 用: cat→str ---
    X_cat = X.copy()
    for c in cat:
        X_cat[c] = X_cat[c].astype(str)
    cat_idx = [X.columns.get_loc(c) for c in cat]
    w = df["weight"].values if "weight" in df else None
    return X, X_num, X_cat, y, cat_idx, w

def train_models(
    df: pd.DataFrame,
    num_rounds: int,
    cat: list[str],
    params_lgb: dict,
    params_xgb: dict,
    xgb_rounds: int,
    params_cat: dict,
    cat_rounds: int
) -> tuple:
    """return (lgb_model, xgb_model, cat_model)"""
    X, X_num, X_cat, y, cat_idx, w = _prep_xy(df, cat)

    # --- LightGBM ---
    dtrain_lgb = lgb.Dataset(X, label=y, weight=w, categorical_feature=cat, free_raw_data=True)
    mdl_lgb = lgb.train(params_lgb, dtrain_lgb, num_boost_round=num_rounds)

    # --- XGBoost ---
    dtrain_xgb = xgb.DMatrix(X_num, label=y, weight=w)
    mdl_xgb = xgb.train(params_xgb, dtrain_xgb, xgb_rounds)

    # --- CatBoost ---
    mdl_cat = CatBoostRegressor(
        iterations=cat_rounds,
        **params_cat,
        verbose=False,
        random_seed=42
    )
    mdl_cat.fit(X_cat, y, sample_weight=w, cat_features=cat_idx)

    return mdl_lgb, mdl_xgb, mdl_cat

# ---- Optuna で LightGBM ハイパラ探索 ----
# ---- Optuna で LightGBM ハイパラ探索 ----
def tune_lgbm_params(df_train: pd.DataFrame, cat: list[str], n_trials: int = 60):
    # Remove unsupported object dtype column to avoid LightGBM error
    df_train = df_train.drop(columns=['quoteType'], errors='ignore')
    # Prepare feature matrix and target/weight series for labelled indexing
    X = df_train.drop(["Date", "target", "weight"], axis=1)
    y_series = df_train["target"]
    w_series = df_train["weight"]
    # use walk-forward CV instead of TimeSeriesSplit
    splits = walk_forward_splits(df_train, 'Date', n_splits=4)

    def objective(trial):
        params = dict(
            objective="regression",
            metric="l2",
            learning_rate=trial.suggest_float("lr", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            max_depth=trial.suggest_int("max_depth", 4, 14),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            min_child_samples = trial.suggest_int("min_child_samples", 10, 50),
            num_threads=1,
            verbosity=-1,
        )
        scores, iters = [], []
        for tr_idx, vl_idx in splits:
            if len(tr_idx) == 0 or len(vl_idx) == 0:
                continue
            # Use .loc since tr_idx and vl_idx are label-based indexers
            X_tr = X.loc[tr_idx]
            y_tr = y_series.loc[tr_idx].values
            w_tr = w_series.loc[tr_idx].values
            dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
                              categorical_feature=cat, free_raw_data=True)

            X_vl = X.loc[vl_idx]
            y_vl = y_series.loc[vl_idx].values
            w_vl = w_series.loc[vl_idx].values
            dvl = lgb.Dataset(X_vl, label=y_vl, weight=w_vl,
                              categorical_feature=cat, free_raw_data=True)

            m = lgb.train(params, dtr, 500,
                          valid_sets=[dvl],
                          valid_names=["valid"],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            # compute validation predictions at best iteration and evaluate R2
            preds = m.predict(X_vl, num_iteration=m.best_iteration)
            r2 = r2_score(y_vl, preds)
            scores.append(r2)
            iters.append(m.best_iteration)
        if not scores:
            raise ValueError("No valid CV splits")
        trial.set_user_attr("best_iters", int(np.mean(iters)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_params.update(objective="regression", metric=["r2"], num_threads=1, verbosity=-1)
    best_iter  = study.best_trial.user_attrs["best_iters"]
    return best_params, best_iter

# ---- Optuna で XGBoost ハイパラ探索 ----
def tune_xgb_params(df_train: pd.DataFrame, cat: list[str], n_trials: int = 60):
    # Remove unsupported object dtype columns
    df_train = df_train.drop(columns=['quoteType'], errors='ignore')
    X = df_train.drop(["Date", "target", "weight"], axis=1)
    y_series = df_train["target"]
    w_series = df_train["weight"]
    splits = walk_forward_splits(df_train, 'Date', n_splits=4)

    def objective(trial):
        params = dict(
            objective="reg:squarederror",
            eval_metric="rmse",
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            nthread=1,
            tree_method="hist"
        )
        scores, iters = [], []
        for tr_idx, vl_idx in splits:
            if len(tr_idx) == 0 or len(vl_idx) == 0:
                continue
            X_tr = X.loc[tr_idx].copy()
            for c in cat:
                X_tr[c] = X_tr[c].cat.codes
            y_tr = y_series.loc[tr_idx].values
            w_tr = w_series.loc[tr_idx].values
            dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)

            X_vl = X.loc[vl_idx].copy()
            for c in cat:
                X_vl[c] = X_vl[c].cat.codes
            y_vl = y_series.loc[vl_idx].values
            w_vl = w_series.loc[vl_idx].values
            dvl = xgb.DMatrix(X_vl, label=y_vl, weight=w_vl)

            m = xgb.train(params, dtr, 500,
                          evals=[(dvl, "valid")],
                          early_stopping_rounds=50,
                          verbose_eval=False)
            preds = m.predict(dvl, iteration_range=(0, m.best_iteration+1))
            r2 = r2_score(y_vl, preds)
            scores.append(r2)
            iters.append(m.best_iteration)
        if not scores:
            raise ValueError("No valid CV splits")
        trial.set_user_attr("best_iters", int(np.mean(iters)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_params.update(objective="reg:squarederror", eval_metric="rmse", nthread=1, tree_method="hist")
    best_iter = study.best_trial.user_attrs["best_iters"]
    return best_params, best_iter

# ---- Optuna で CatBoost ハイパラ探索 ----
def tune_cat_params(df_train: pd.DataFrame, cat: list[str], n_trials: int = 60):
    # Remove unsupported object dtype columns
    df_train = df_train.drop(columns=['quoteType'], errors='ignore')
    X = df_train.drop(["Date", "target", "weight"], axis=1)
    y_series = df_train["target"]
    w_series = df_train["weight"]
    splits = walk_forward_splits(df_train, 'Date', n_splits=4)

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            loss_function="MAE",
        )
        scores, iters = [], []
        for tr_idx, vl_idx in splits:
            if len(tr_idx) == 0 or len(vl_idx) == 0:
                continue
            X_tr = X.loc[tr_idx].copy()
            for c in cat:
                X_tr[c] = X_tr[c].astype(str)
            y_tr = y_series.loc[tr_idx].values
            w_tr = w_series.loc[tr_idx].values

            X_vl = X.loc[vl_idx].copy()
            for c in cat:
                X_vl[c] = X_vl[c].astype(str)
            y_vl = y_series.loc[vl_idx].values
            w_vl = w_series.loc[vl_idx].values

            cat_idx = [X.columns.get_loc(c) for c in cat]
            m = CatBoostRegressor(
                iterations=500,
                **params,
                verbose=False,
                random_seed=42
            )
            m.fit(X_tr, y_tr, sample_weight=w_tr, cat_features=cat_idx,
                  eval_set=(X_vl, y_vl), early_stopping_rounds=50)
            preds = m.predict(X_vl)
            r2 = r2_score(y_vl, preds)
            scores.append(r2)
            iters.append(m.get_best_iteration())
        if not scores:
            raise ValueError("No valid CV splits")
        trial.set_user_attr("best_iters", int(np.mean(iters)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_params.update(loss_function="MAE")
    best_iter = study.best_trial.user_attrs["best_iters"]
    return best_params, best_iter

def predict(mat: pd.DataFrame, models: tuple, weights: np.ndarray) -> pd.DataFrame:
    last_feat = {}

    for l in LAGS:
        last_feat[f"lag_{l}"] = mat.shift(l).tail(1)

    last_feat["sma_22"] = mat.rolling(22).mean().tail(1)
    last_feat["sma_50"] = mat.rolling(50).mean().tail(1)
    rolling_std = mat.rolling(22).std().tail(1)
    last_feat["bb_width"] = (rolling_std * 2) / last_feat["sma_22"]

    ema12 = mat.ewm(span=12, adjust=False).mean().tail(1)
    ema26 = mat.ewm(span=26, adjust=False).mean().tail(1)
    last_feat["macd"] = ema12 - ema26

    last_feat["rsi_14"] = 100 - 100 / (1 + mat.pct_change().rolling(14).mean()).tail(1)

    X = (
        pd.concat(last_feat, axis=1)
        .stack(future_stack=True)   # suppress FutureWarning
        .reset_index()
        .drop(["Date"], axis=1)
    )
    X["dayofweek"] = datetime.now().weekday()
    X["month"] = datetime.now().month
    X["Ticker"] = X["Ticker"].astype("category")

    global SECTOR_DICT
    X["Sector"] = X["Ticker"].map(SECTOR_DICT).fillna("Unknown").astype("category")

    X_lgb = X.copy()
    X_xgb = X.copy()
    # --- align columns with training feature order ---
    global FEAT_COLS
    missing = [c for c in FEAT_COLS if c not in X_lgb.columns]
    for m in missing:
        X_lgb[m] = 0
        X_xgb[m] = 0
    X_lgb = X_lgb[FEAT_COLS]
    X_xgb = X_xgb[FEAT_COLS].copy()
    for c in ["Ticker", "Sector"]:
        X_xgb[c] = X_xgb[c].cat.codes

    lgb_model, xgb_model, cat_model = models
    p_lgb = lgb_model.predict(X_lgb)
    p_xgb = xgb_model.predict(xgb.DMatrix(X_xgb))
    p_cat = cat_model.predict(X_lgb)
    preds = weights[0]*p_lgb + weights[1]*p_xgb + weights[2]*p_cat

    current_prices = mat.iloc[-1]
    # map to numeric current prices
    curr = X["Ticker"].map(current_prices).astype(float)

    # preds は log-return ⇒ 終値を復元
    pred_price = curr.values * np.exp(preds)
    ratio = np.exp(preds) - 1

    df_out = pd.DataFrame({
        "Ticker": X["Ticker"],
        "Current": curr.values,
        "Predicted": pred_price,
        "Ratio": ratio,
    })
    df_out = df_out[df_out["Ratio"] > 0]
    return df_out.sort_values("Ratio", ascending=False)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--csv",default="result.csv")
    args = p.parse_args()

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
    gc.collect()


    # --- 外生変数データの取得 ---
    start_date = mat.index.min().strftime("%Y-%m-%d")
    end_date   = mat.index.max().strftime("%Y-%m-%d")
    EXOG_DF = load_exogenous(EXOGENOUS_TICKERS, start_date, end_date, INTERVAL)

    df = features(mat, FUNDAMENTALS, EXOG_DF)

    # ---- サンプルウェイト (0.99 ** day_diff) ----
    max_dt = pd.to_datetime(df["Date"]).max()
    df["weight"] = 0.995 ** ((max_dt - pd.to_datetime(df["Date"])).dt.days)

    # ---- 時系列 hold‑out ----
    months = int(TRAIN_SPLIT_YEARS * 12)
    # 特徴量作成後のデータに基づいてホールドアウトを計算
    max_feat_date = pd.to_datetime(df["Date"]).max()
    split_date = max_feat_date - pd.DateOffset(months=months)
    df_train = df[pd.to_datetime(df["Date"]) <= split_date].copy()
    df_test  = df[pd.to_datetime(df["Date"]) >  split_date].copy()
    if df_test.empty:
        print("テスト用データが存在しないはずがありません（特徴量データ基準で分割）。")
        df_test = df_train.copy()

    cat = ["Ticker", "Sector"]
    for c in cat:
        df_train[c] = df_train[c].astype("category")

    for c in cat:
        for df_ in (df_train, df_test):
            df_[c] = pd.Categorical(df_[c], categories=df_train[c].cat.categories)

    # ---- グローバルモデルのチューニング＆訓練 ----
    best_params_g, best_iter_g = tune_lgbm_params(df_train, cat, n_trials=30)
    # --- XGBoost hyperparameter tuning ---
    best_params_xgb, best_rounds_xgb = tune_xgb_params(df_train, cat, n_trials=30)
    # --- CatBoost hyperparameter tuning ---
    best_params_cat, best_rounds_cat = tune_cat_params(df_train, cat, n_trials=30)
    num_boost_g = max(50, best_iter_g)
    global_models = train_models(
        df_train,
        num_boost_g,
        cat,
        best_params_g,
        best_params_xgb,
        best_rounds_xgb,
        best_params_cat,
        best_rounds_cat
    )
    # 訓練用データやOptunaオブジェクトを解放してメモリをクリア
    # 仮：等重みアンサンブル（必要に応じて検証データで重み算出してください）
    weights_global = np.array([1/3, 1/3, 1/3])

    # ---- シーケンスモデル（Transformer）訓練 ----
    # Prepare LSTM-like sequences for transformer
    X_seq_all, y_seq_all, idxs = prepare_lstm_sequences(df, window_size=60, shift=SHIFT)
    # Split sequences by train/test indices
    train_mask = np.isin(idxs, df_train.index)
    test_mask  = np.isin(idxs, df_test.index)
    X_seq_train, y_seq_train = X_seq_all[train_mask], y_seq_all[train_mask]
    X_seq_test,  y_seq_test  = X_seq_all[test_mask],  y_seq_all[test_mask]
    # Build dataloaders
    Xt_train = torch.tensor(X_seq_train, dtype=torch.float32)
    yt_train = torch.tensor(y_seq_train, dtype=torch.float32).view(-1,1)
    Xt_test  = torch.tensor(X_seq_test,  dtype=torch.float32)
    yt_test  = torch.tensor(y_seq_test,  dtype=torch.float32).view(-1,1)
    train_ds = TensorDataset(Xt_train, yt_train)
    test_ds  = TensorDataset(Xt_test,  yt_test)
    train_dl = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=128,
        num_workers=0,
        pin_memory=True,
    )
    # Instantiate and train
    trans_model = TransformerForecast(input_dim=X_seq_train.shape[2])
    optimizer_t = torch.optim.Adam(trans_model.parameters(), lr=1e-3)
    criterion_t = nn.MSELoss()
    for epoch in range(10):
        trans_model.train()
        for xb, yb in train_dl:
            optimizer_t.zero_grad()
            loss = criterion_t(trans_model(xb), yb)
            loss.backward()
            optimizer_t.step()
    # Evaluate on test sequences
    trans_model.eval()
    p_trans = []
    with torch.no_grad():
        for xb, _ in test_dl:
            p_trans.append(trans_model(xb).numpy().flatten())
    p_trans = np.concatenate(p_trans)
    # シーケンス学習用データを解放してメモリをクリア
    del X_seq_all, y_seq_all, idxs, Xt_train, yt_train, Xt_test, yt_test, train_ds, test_ds, train_dl, test_dl
    gc.collect()

    # ---- 指標 ----
    def _metric(df_part, name):
        # prepare features
        X, X_num, X_cat, y_true, _, _ = _prep_xy(df_part, cat)
        # get ensemble predictions
        p_lgb = global_models[0].predict(X)
        p_xgb = global_models[1].predict(xgb.DMatrix(X_num))
        p_cat = global_models[2].predict(X_cat)
        preds = weights_global[0]*p_lgb + weights_global[1]*p_xgb + weights_global[2]*p_cat
        mse = mean_squared_error(y_true, preds)
        r2  = r2_score(y_true, preds)
        return f"{name}  MSE: {mse:.6f} | R2: {r2:.4f}"


    # ---- グローバルモデルによる予測 ----
    # (Pre-optimization output removed)

    # --- ensemble weight optimization on validation set ---
    X_test, X_num_test, X_cat_test, y_test, _, _ = _prep_xy(df_test, cat)
    p_lgb = global_models[0].predict(X_test)
    p_xgb = global_models[1].predict(xgb.DMatrix(X_num_test))
    p_cat = global_models[2].predict(X_cat_test)
    preds_list = [p_lgb, p_xgb, p_cat, p_trans]
    # initial equal weights for four models
    weights_global = np.array([0.25, 0.25, 0.25, 0.25])
    def loss_fn(w):
        blended = sum(wi * pi for wi, pi in zip(w, preds_list))
        return -r2_score(y_test, blended)
    cons = ({'type':'eq','fun': lambda w: sum(w)-1})
    bounds = [(0,1)]*4
    res = minimize(loss_fn, weights_global, bounds=bounds, constraints=cons)
    weights_global = res.x

    # ---- Output after optimization ----
    preds_df_opt = predict(mat, global_models, weights_global[:3]).head(20)
    preds_df_opt.to_csv(args.csv, index=False)
    print(preds_df_opt.to_string(index=False, formatters={
        "Current": "{:.2f}".format,
        "Predicted": "{:.2f}".format,
        "Ratio": "{:.2%}".format,
    }))
    blended = sum(w * p for w, p in zip(weights_global, preds_list))
    print(f"Optimized weights: {weights_global}")
    print(f"Test  (opt) MSE: {mean_squared_error(y_test, blended):.6f} | R2: {r2_score(y_test, blended):.4f}")

if __name__=="__main__":
    main()