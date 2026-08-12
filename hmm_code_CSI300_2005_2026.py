import os
import math
import glob
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Keep warnings visible so convergence/numerical issues are not silently hidden.
warnings.filterwarnings('default')
np.random.seed(42)

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    raise RuntimeError(f"matplotlib failed to load properly: {e}")

try:
    from hmmlearn.hmm import GaussianHMM
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: hmmlearn.\n\n"
        "Please run the following command in your terminal first:\n"
        "pip install hmmlearn openpyxl pandas matplotlib numpy\n"
    )

# =========================
# Final global parameter configuration
# =========================
DEFAULT_OUT_DIR = 'results_hmm_CSI300_2005_2026'
DEFAULT_DATA_FILE = 'CSI300_2005_2026_returns.xlsx'

# 1. Complete state-count grid: replicate all tests from K=2 to K=7 in the paper
STATE_LIST = [2, 3, 4, 5, 6, 7]

# 2. Rebalancing and backtesting parameters
# The A-share sample has about 6,000 trading days. Use the first 1,260 days
# as the initial training window for OUT-OF-SAMPLE backtesting only.
# In-sample model comparison uses the full available sample, following the reference paper.
INITIAL_TRAIN_WINDOW = 1260

# Follow the paper strictly: retrain and rebalance at monthly frequency, about 21 trading days.
ROLLING_REBALANCE_STEP = 21

# Deprecated: in-sample analysis now uses the full sample.
TRAIN_FRAC = None
N_RANDOM_STARTS_INSAMPLE = 8
N_RANDOM_STARTS_ROLLING = 2
HMM_N_ITER = 200

# 3. Realistic trading constraints
TRANSACTION_COST_BP = 10  # Transaction cost: 10 bps, one-way, 0.1%
MAX_LEVERAGE = 1.5        # Maximum exposure: 150%, allowing 50% leverage
MIN_LEVERAGE = 0.0        # Minimum exposure: 0%, no short selling
RISK_AVERSION = 6.0       # Match the paper: risk-aversion coefficient Gamma = 6
USE_RF_COLUMN = True

DATE_CANDIDATES = ['trade_date', 'date', 'datetime', '\u65e5\u671f', '\u4ea4\u6613\u65e5\u671f']


@dataclass
class FitResult:
    model_name: str
    k_states: int
    seed: int
    logL: float
    aic: float
    bic: float
    cd: float
    avg_rank: Optional[float]
    model: GaussianHMM
    states: np.ndarray
    posterior: np.ndarray
    state_means: np.ndarray
    state_vars: np.ndarray


def parse_args():
    p = argparse.ArgumentParser(description='Final HMM replication script for CSI 300 data')
    # Keep the default as None so find_data_file() can search both the current
    # working directory and the folder that contains this script.
    p.add_argument('--file', type=str, default=None, help='Data file path, CSV or Excel')
    p.add_argument('--outdir', type=str, default=DEFAULT_OUT_DIR, help='Output directory')
    p.add_argument('--date-col', type=str, default=None, help='Date column name; leave blank for auto-detection')
    p.add_argument('--close-col', type=str, default='close', help='Close-price column name')
    p.add_argument('--quick', action='store_true', help='Quick mode: reduce the number of random initializations')
    return p.parse_args()


def find_data_file(user_path: Optional[str] = None) -> str:
    """Find the CSI 300 input data file.

    Priority order:
    1. An explicit --file path supplied by the user.
    2. DEFAULT_DATA_FILE in the current working directory.
    3. DEFAULT_DATA_FILE in the same folder as this Python script.
    4. A uniquely identifiable CSV/Excel file in those folders, preferring
       filenames that contain CSI300/CSI_300/CSI-300.

    This avoids the old behavior where the default --file value made automatic
    discovery unreachable, and removes the old non-CSI filename bias.
    """
    if user_path:
        path = os.path.abspath(os.path.expanduser(user_path))
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"The file you provided does not exist: {path}")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    search_dirs = []
    for d in [os.getcwd(), script_dir]:
        d = os.path.abspath(d)
        if d not in search_dirs:
            search_dirs.append(d)

    default_candidates = [os.path.join(d, DEFAULT_DATA_FILE) for d in search_dirs]
    for path in default_candidates:
        if os.path.exists(path):
            print(f"Using default data file: {path}")
            return path

    candidates = []
    for d in search_dirs:
        for pattern in ['*.csv', '*.xlsx', '*.xls']:
            candidates.extend(sorted(glob.glob(os.path.join(d, pattern))))

    # Deduplicate while preserving order.
    candidates = list(dict.fromkeys(os.path.abspath(c) for c in candidates))

    if len(candidates) == 1:
        print(f"Auto-detected and using data file: {candidates[0]}")
        return candidates[0]

    if len(candidates) > 1:
        preferred_tokens = ['csi300', 'csi_300', 'csi-300']
        matched = [
            c for c in candidates
            if any(token in os.path.basename(c).lower() for token in preferred_tokens)
        ]
        if len(matched) == 1:
            print(f"Auto-detected and using matched CSI 300 file: {matched[0]}")
            return matched[0]

        raise FileNotFoundError(
            "Multiple CSV/Excel files were found. Please specify the raw price file "
            "explicitly with --file. Candidates: " + str(candidates)
        )

    raise FileNotFoundError(
        "No data file was found. Put " + DEFAULT_DATA_FILE +
        " in the current working directory or in the same folder as this script, "
        "or specify a path with --file."
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _detect_date_column(df: pd.DataFrame, user_date_col: Optional[str]) -> Optional[str]:
    if user_date_col:
        col = user_date_col.strip().lower()
        if col in [str(c).strip().lower() for c in df.columns]:
            return col
        raise ValueError(f"The specified date column is not in the data. Current columns: {list(df.columns)}")

    for c in DATE_CANDIDATES:
        if c in df.columns:
            return c

    first_col = df.columns[0]
    try:
        parsed = pd.to_datetime(df[first_col], errors='coerce')
        if parsed.notna().mean() > 0.8:
            return first_col
    except Exception:
        pass

    return None


def load_data(path: str, user_date_col: Optional[str], close_col: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext == '.csv':
        df = pd.read_csv(path)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

    df = _normalize_columns(df)
    date_col = _detect_date_column(df, user_date_col)

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).set_index(date_col)
    else:
        try:
            tmp = pd.read_csv(path, index_col=0) if ext == '.csv' else pd.read_excel(path, index_col=0)
            tmp = _normalize_columns(tmp)
            tmp.index = pd.to_datetime(tmp.index, errors='coerce')
            tmp = tmp[~tmp.index.isna()]
            df = tmp
        except Exception:
            raise ValueError("Could not identify the date column. Please specify it with --date-col.")

    df.index.name = 'date'

    close_col = close_col.strip().lower()
    if close_col not in df.columns:
        raise ValueError(f"The file must contain the close-price column {close_col}. Current columns: {list(df.columns)}")

    if 'rf' not in df.columns:
        df['rf'] = 0.0

    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')].copy()
    return df


def compute_features(df: pd.DataFrame, close_col: str) -> pd.DataFrame:
    """Compute model returns and investable asset returns.

    log_return is used for HMM/HSMM estimation.
    simple_return is used for portfolio backtesting and wealth calculations.
    """
    out = df.copy()
    out['log_return'] = np.log(out[close_col]).diff()
    out['simple_return'] = out[close_col].pct_change()
    out = out.dropna(subset=['log_return', 'simple_return']).copy()
    return out


def descriptive_stats(returns: pd.Series) -> pd.DataFrame:
    s = returns.describe().to_frame().T
    s['skew'] = returns.skew()
    s['kurtosis'] = returns.kurt()
    return s


def runs_from_states(states: np.ndarray) -> List[Tuple[int, int, int, int]]:
    states = np.asarray(states, dtype=int)
    if len(states) == 0:
        return []

    runs = []
    start = 0
    cur = states[0]

    for i in range(1, len(states)):
        if states[i] != cur:
            runs.append((int(cur), start, i - 1, i - start))
            start = i
            cur = states[i]

    runs.append((int(cur), start, len(states) - 1, len(states) - start))
    return runs


def hmm_num_params(k: int) -> int:
    return k * (k - 1) + (k - 1) + k + k


def compute_aic_bic(logL: float, n_params: int, n_obs: int) -> Tuple[float, float]:
    aic = 2 * n_params - 2 * logL
    bic = math.log(n_obs) * n_params - 2 * logL
    return aic, bic


def correctly_decoded_states(returns: pd.Series, states: np.ndarray, state_means: np.ndarray) -> float:
    eps = 1e-8
    rr = returns.values
    runs = runs_from_states(states)

    if not runs:
        return np.nan

    ok = 0
    stdev = float(np.std(rr)) if len(rr) else 0.0

    for s, a, b, _ in runs:
        seg_sum = rr[a:b + 1].sum()
        mu = state_means[s]

        if mu > eps and seg_sum > 0:
            ok += 1
        elif mu < -eps and seg_sum < 0:
            ok += 1
        elif abs(mu) <= eps and abs(seg_sum) <= stdev:
            ok += 1

    return ok / len(runs)


def fit_one_hmm(returns: pd.Series, k: int, seed: int) -> FitResult:
    X = returns.values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=k,
        covariance_type='diag',
        n_iter=HMM_N_ITER,
        random_state=seed
    )

    model.fit(X)

    if hasattr(model, 'monitor_') and not model.monitor_.converged:
        print(
            f"[WARN] HMM did not converge: K={k}, seed={seed}, "
            f"iterations={getattr(model.monitor_, 'iter', 'unknown')}"
        )

    logL = model.score(X)
    states = model.predict(X)
    posterior = model.predict_proba(X)
    means = model.means_.reshape(-1)
    vars_ = model.covars_.reshape(-1)

    aic, bic = compute_aic_bic(logL, hmm_num_params(k), len(X))
    cd = correctly_decoded_states(returns, states, means)

    return FitResult(
        'HMM',
        k,
        seed,
        float(logL),
        float(aic),
        float(bic),
        float(cd),
        None,
        model,
        states,
        posterior,
        means,
        vars_
    )


def run_hmm_model_search(returns: pd.Series, seeds_per_k: int) -> Tuple[pd.DataFrame, Dict[int, FitResult]]:
    all_rows = []
    fits: List[FitResult] = []

    for k in STATE_LIST:
        for seed in range(seeds_per_k):
            try:
                fr = fit_one_hmm(returns, k, seed)
                fits.append(fr)
                all_rows.append({
                    'model': fr.model_name,
                    'k_states': fr.k_states,
                    'seed': fr.seed,
                    'logL': fr.logL,
                    'AIC': fr.aic,
                    'BIC': fr.bic,
                    'CD': fr.cd
                })
            except Exception as e:
                print(f"[WARN] HMM in-sample fit failed: K={k}, seed={seed}, error={e}")

    ranking_df = pd.DataFrame(all_rows)
    valid = ranking_df.dropna(subset=['AIC', 'CD']).copy()

    if valid.empty:
        return ranking_df, {}

    valid['rank_AIC'] = valid.groupby('k_states')['AIC'].rank(method='average', ascending=True)
    valid['rank_CD'] = valid.groupby('k_states')['CD'].rank(method='average', ascending=False)
    valid['avg_rank'] = (valid['rank_AIC'] + valid['rank_CD']) / 2.0

    chosen: Dict[int, FitResult] = {}

    for k in STATE_LIST:
        sub = valid[valid['k_states'] == k].sort_values(['avg_rank', 'AIC'])
        if len(sub) == 0:
            continue

        best_seed = int(sub.iloc[0]['seed'])
        best_fit = next(f for f in fits if f.k_states == k and f.seed == best_seed)
        best_fit.avg_rank = float(sub.iloc[0]['avg_rank'])
        chosen[k] = best_fit

    ranking_df = ranking_df.merge(
        valid[['k_states', 'seed', 'rank_AIC', 'rank_CD', 'avg_rank']],
        on=['k_states', 'seed'],
        how='left'
    )

    return ranking_df, chosen


def run_single_k_search(returns: pd.Series, k: int, seeds: int) -> Optional[FitResult]:
    """Speed up out-of-sample estimation by searching only for the best initialization for a specified K."""
    fits = []

    for seed in range(seeds):
        try:
            fits.append(fit_one_hmm(returns, k, seed))
        except Exception as e:
            print(f"[WARN] HMM rolling fit failed: K={k}, seed={seed}, error={e}")
            continue

    if not fits:
        return None

    # For out-of-sample estimation, use the model with the lowest AIC to reduce statistical overfitting.
    fits.sort(key=lambda x: x.aic)
    return fits[0]


def summarize_regimes(returns: pd.Series, states: np.ndarray, state_means: np.ndarray) -> pd.DataFrame:
    """Summarize decoded regimes.

    The model is estimated on raw log returns, while percentage columns are added
    for direct reporting in the dissertation tables.
    """
    rows = []
    rr = returns.values

    for s in sorted(np.unique(states)):
        mask = states == s
        durations = [length for st, _, _, length in runs_from_states(states) if st == s]
        mean_raw = float(state_means[s])
        vol_raw = float(np.std(rr[mask])) if mask.sum() > 1 else np.nan

        rows.append({
            'state': int(s) + 1,  # one-based label for reporting
            'mean_return': mean_raw,
            'volatility': vol_raw,
            'mean_daily_pct': mean_raw * 100.0,
            'vol_daily_pct': vol_raw * 100.0 if pd.notna(vol_raw) else np.nan,
            'occupancy': float(mask.mean()),
            'occupancy_pct': float(mask.mean()) * 100.0,
            'occurrences': int(sum(1 for st, _, _, _ in runs_from_states(states) if st == s)),
            'avg_duration': float(np.mean(durations)) if durations else np.nan,
            'n_obs': int(mask.sum()),
        })

    return pd.DataFrame(rows).sort_values('state').reset_index(drop=True)


def state_expected_moments(model: GaussianHMM, posterior_t: np.ndarray) -> Tuple[float, float]:
    means = model.means_.reshape(-1)
    vars_ = model.covars_.reshape(-1)

    mu = float(np.sum(posterior_t * means))
    second = float(np.sum(posterior_t * (vars_ + means ** 2)))
    var = max(second - mu ** 2, 1e-8)

    return mu, var


def bounded_weight(mu: float, var: float, gamma: float = RISK_AVERSION) -> float:
    w = (mu / var) / gamma
    return float(np.clip(w, MIN_LEVERAGE, MAX_LEVERAGE))


def perf_stats(rets: pd.Series, name: str, avg_exposure: float = 1.0, avg_turnover: float = 0.0) -> pd.DataFrame:
    """Performance statistics for simple portfolio returns.

    Ann_Return is CAGR, consistent with terminal wealth.
    Ann_Mean_Return is the arithmetic annualised mean, provided for reference.
    """
    rets = pd.Series(rets).dropna()

    if len(rets) == 0:
        return pd.DataFrame([{'Strategy': name}])

    wealth = (1 + rets).cumprod()
    ann_ret = wealth.iloc[-1] ** (252 / len(rets)) - 1
    ann_mean_ret = rets.mean() * 252
    ann_vol = rets.std(ddof=1) * np.sqrt(252) if len(rets) > 1 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    dd = wealth / wealth.cummax() - 1

    return pd.DataFrame([{
        'Strategy': name,
        'Ann_Return_CAGR': ann_ret,
        'Ann_Mean_Return': ann_mean_ret,
        'Ann_Vol': ann_vol,
        'Sharpe': sharpe,
        'Max_Drawdown': dd.min(),
        'Terminal_Wealth': wealth.iloc[-1],
        'Avg_Exposure': avg_exposure,
        'Avg_Turnover_per_Day': avg_turnover
    }])


def compute_benchmarks(data: pd.DataFrame, initial_n: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Construct Benchmark 1 and a monthly-rebalanced rolling mean-variance benchmark.

    Benchmark 1 is buy-and-hold in the CSI 300 Index.

    Benchmark 2 uses the same rebalancing grid as the HMM strategies. The initial
    training window contains observations data.iloc[:initial_n]. The first
    out-of-sample return is data.iloc[initial_n], so the first rebalancing date is
    the last in-sample observation, data.iloc[initial_n - 1].

    The rolling mean-variance benchmark estimates the signal from the previous
    252 available log returns and applies transaction costs only when the target
    weight changes.
    """
    if initial_n <= 0 or initial_n >= len(data):
        raise ValueError(
            f"initial_n must be in [1, len(data)-1]. Got initial_n={initial_n}, len(data)={len(data)}."
        )

    asset_returns = data['simple_return']
    signal_returns = data['log_return']
    rf = data['rf']
    dates = data.index

    b2_ret_net = pd.Series(index=dates, dtype=float)
    b2_w = pd.Series(index=dates, dtype=float)
    b2_turnover = pd.Series(0.0, index=dates, dtype=float)

    prev_w = 1.0
    train_end = initial_n - 1

    for t in range(train_end, len(data) - 1, ROLLING_REBALANCE_STEP):
        # Previous 252 trading-day empirical mean-variance signal, including
        # the latest available training observation at index t.
        hist = signal_returns.iloc[max(0, t - 251):t + 1].dropna()

        if len(hist) < 20:
            w = prev_w
        else:
            mu = float(hist.mean())
            var = float(hist.var())
            if not np.isfinite(var) or var <= 0:
                var = 1e-8
            w = bounded_weight(mu, var, gamma=RISK_AVERSION)

        end_t = min(t + ROLLING_REBALANCE_STEP, len(data) - 1)

        for j in range(t + 1, end_t + 1):
            turnover = abs(w - prev_w) if j == t + 1 else 0.0
            tc = turnover * TRANSACTION_COST_BP / 10000.0
            b2_ret_net.iloc[j] = w * asset_returns.iloc[j] + (1 - w) * rf.iloc[j] - tc
            b2_w.iloc[j] = w
            b2_turnover.iloc[j] = turnover

        prev_w = w

    return asset_returns, b2_ret_net, b2_w, b2_turnover


def backtest_all_strategies(data: pd.DataFrame, out_dir: str):
    """Run expanding-window out-of-sample HMM backtests.

    The in-sample selected full-sample models are deliberately not used here,
    because doing so would introduce look-ahead bias. Each HMM specification is
    re-estimated at each rebalancing date using only data available up to that
    date.
    """
    model_returns = data['log_return']      # used to estimate HMM parameters
    asset_returns = data['simple_return']   # used to compute investable portfolio returns
    rf = data['rf']
    dates = data.index

    # The initial window contains data.iloc[:initial_n]. The first test return is
    # data.iloc[initial_n]. This aligns the HMM Python backtest with the R/HSMM
    # convention date > initial_window_end.
    initial_n = max(INITIAL_TRAIN_WINDOW, int(len(data) * 0.2))
    if initial_n >= len(data) - 1:
        raise ValueError(
            f"Initial training window is too large for the sample: initial_n={initial_n}, len(data)={len(data)}."
        )

    train_end = initial_n - 1
    test_dates = dates[initial_n:]

    # Compute benchmark return series using the same monthly rebalancing grid as the HMM strategies.
    bh_ret, b2_ret_net, b2_w, b2_to = compute_benchmarks(data, initial_n)

    all_perf_rows = []
    all_strategy_series = []

    # Add benchmark performance first.
    bh_eval = bh_ret.loc[test_dates]
    bh_perf = perf_stats(bh_eval, 'Benchmark 1 (Buy&Hold)', 1.0, 0.0)
    b2_eval = b2_ret_net.loc[test_dates].dropna()
    b2_perf = perf_stats(
        b2_eval,
        'Benchmark 2 (Rolling M-V, monthly)',
        b2_w.loc[b2_eval.index].mean(),
        b2_to.loc[b2_eval.index].mean()
    )

    all_perf_rows.extend([bh_perf, b2_perf])

    bh_series = pd.DataFrame({
        'date': bh_eval.index,
        'strategy': 'Benchmark 1 (Buy&Hold)',
        'strategy_ret_net': bh_eval.values,
        'weight': 1.0,
        'turnover': 0.0,
    })
    bh_series['wealth_net'] = (1 + bh_series['strategy_ret_net']).cumprod()
    all_strategy_series.append(bh_series)

    b2_series = pd.DataFrame({
        'date': b2_eval.index,
        'strategy': 'Benchmark 2 (Rolling M-V, monthly)',
        'strategy_ret_net': b2_eval.values,
        'weight': b2_w.loc[b2_eval.index].values,
        'turnover': b2_to.loc[b2_eval.index].values,
    })
    b2_series['wealth_net'] = (1 + b2_series['strategy_ret_net']).cumprod()
    all_strategy_series.append(b2_series)

    plt.figure(figsize=(12, 6))

    plt.plot(
        test_dates,
        (1 + bh_eval).cumprod(),
        label='Bench 1: Buy&Hold',
        color='black',
        linewidth=1.5,
        alpha=0.7
    )

    plt.plot(
        b2_eval.index,
        (1 + b2_eval).cumprod(),
        label='Bench 2: Rolling M-V (monthly)',
        color='gray',
        linestyle='--'
    )

    # Test the HMM strategy for each K.
    for k in STATE_LIST:
        print(
            f"Running out-of-sample backtest: HMM (K={k})... "
            f"Rebalancing interval = {ROLLING_REBALANCE_STEP} trading days"
        )

        recs = []
        prev_w = 1.0

        for t in range(train_end, len(data) - 1, ROLLING_REBALANCE_STEP):
            # Expanding window: use all data available through index t.
            train_ret = model_returns.iloc[:t + 1]

            # Use a single-K search to improve computational speed.
            best_model = run_single_k_search(train_ret, k, seeds=N_RANDOM_STARTS_ROLLING)

            if best_model is None:
                w = prev_w
            else:
                # Use the one-step-ahead HMM predictive state distribution.
                # post_t is P(S_t | Y_1:t); multiplying by the transition
                # matrix gives P(S_{t+1} | Y_1:t), which is the appropriate
                # signal for the next holding period.
                post_t = np.asarray(best_model.posterior[-1], dtype=float)
                p_next = post_t @ best_model.model.transmat_
                p_next = np.asarray(p_next, dtype=float)

                # Numerical safeguard: keep the probability vector valid even
                # if very small floating-point errors appear after matrix multiplication.
                p_next = np.clip(p_next, 0.0, 1.0)
                p_sum = p_next.sum()
                if np.isfinite(p_sum) and p_sum > 0:
                    p_next = p_next / p_sum
                else:
                    p_next = post_t

                mu, var = state_expected_moments(best_model.model, p_next)
                w = bounded_weight(mu, var, gamma=RISK_AVERSION)

            end_t = min(t + ROLLING_REBALANCE_STEP, len(data) - 1)

            for j in range(t + 1, end_t + 1):
                next_ret = asset_returns.iloc[j]
                next_rf = rf.iloc[j]
                turnover = abs(w - prev_w) if j == t + 1 else 0.0
                tc = turnover * TRANSACTION_COST_BP / 10000.0
                strat_ret_net = w * next_ret + (1 - w) * next_rf - tc

                recs.append({
                    'date': dates[j],
                    'strategy_ret_net': strat_ret_net,
                    'weight': w,
                    'turnover': turnover
                })

            prev_w = w

        if len(recs) == 0:
            print(f"[WARN] No out-of-sample records were produced for HMM (K={k}).")
            continue

        bt_df = pd.DataFrame(recs).set_index('date')
        bt_df['wealth_net'] = (1 + bt_df['strategy_ret_net']).cumprod()
        bt_df['strategy'] = f'HMM (K={k})'

        hmm_perf = perf_stats(
            bt_df['strategy_ret_net'],
            f'HMM (K={k})',
            bt_df['weight'].mean(),
            bt_df['turnover'].mean()
        )

        all_perf_rows.append(hmm_perf)
        all_strategy_series.append(bt_df.reset_index()[[
            'date', 'strategy', 'strategy_ret_net', 'weight', 'turnover', 'wealth_net'
        ]])

        plt.plot(
            bt_df.index,
            bt_df['wealth_net'],
            label=f'HMM (K={k})',
            alpha=0.8
        )

    final_perf = pd.concat(all_perf_rows, ignore_index=True)
    final_perf.to_csv(
        os.path.join(out_dir, 'final_out_of_sample_performance.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    strategy_series = pd.concat(all_strategy_series, ignore_index=True)
    strategy_series.to_csv(
        os.path.join(out_dir, 'hmm_out_of_sample_strategy_series.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    plt.title('CSI 300 Daily Empirical Study: Cumulative Wealth of HMM Strategies and Benchmarks')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Wealth')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'all_strategies_wealth_curve.png'), dpi=150)
    plt.close()

    return final_perf

def plot_price_with_states(dates, prices, states, out_path, title):
    """Plot CSI 300 close price with decoded HMM state bands.

    This version adds an explicit legend for the price line and each state band.
    The state labels are one-based, so they match the tables exported by
    summarize_regimes().
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    price_line, = ax.plot(
        dates,
        prices,
        linewidth=1.2,
        color='black',
        label='CSI 300 close price'
    )

    uniq = sorted(np.unique(states))
    ymin, ymax = float(np.nanmin(prices)), float(np.nanmax(prices))

    # Use a fixed colormap so the state legend is visible and reproducible.
    cmap = plt.get_cmap('tab10' if len(uniq) <= 10 else 'tab20')
    legend_handles = [price_line]

    for idx, s in enumerate(uniq):
        mask = states == s
        color = cmap(idx % cmap.N)

        ax.fill_between(
            dates,
            ymin,
            ymax,
            where=mask,
            facecolor=color,
            alpha=0.16,
            linewidth=0
        )

        # A rectangle proxy gives a clean legend entry for each state band.
        legend_handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color,
                alpha=0.16,
                edgecolor='none',
                label=f'State {int(s) + 1}'
            )
        )

    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')

    # Put the legend below the plot so it does not cover the state bands.
    ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(len(legend_handles), 4),
        frameon=True
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    global N_RANDOM_STARTS_INSAMPLE, N_RANDOM_STARTS_ROLLING

    args = parse_args()

    if args.quick:
        N_RANDOM_STARTS_INSAMPLE = 4
        N_RANDOM_STARTS_ROLLING = 1

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    print('1/5 Loading and cleaning data')

    data_file = find_data_file(args.file)
    raw = load_data(data_file, args.date_col, args.close_col)
    data = compute_features(raw, args.close_col.strip().lower())

    data_overview = pd.DataFrame([{
        'data_file': data_file,
        'n_obs': len(data),
        'start_date': data.index.min(),
        'end_date': data.index.max(),
        'mean_daily_log_return': data['log_return'].mean(),
        'vol_daily_log_return': data['log_return'].std(ddof=1),
        'mean_daily_simple_return': data['simple_return'].mean(),
        'vol_daily_simple_return': data['simple_return'].std(ddof=1),
    }])
    data_overview.to_csv(
        os.path.join(out_dir, 'data_overview_hmm.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    backtest_settings = pd.DataFrame([{
        'initial_train_window': max(INITIAL_TRAIN_WINDOW, int(len(data) * 0.2)),
        'rolling_rebalance_step': ROLLING_REBALANCE_STEP,
        'transaction_cost_bp': TRANSACTION_COST_BP,
        'max_leverage': MAX_LEVERAGE,
        'min_leverage': MIN_LEVERAGE,
        'risk_aversion': RISK_AVERSION,
        'n_random_starts_insample': N_RANDOM_STARTS_INSAMPLE,
        'n_random_starts_rolling': N_RANDOM_STARTS_ROLLING,
        'hmm_n_iter': HMM_N_ITER,
        'state_list': ','.join(map(str, STATE_LIST)),
    }])
    backtest_settings.to_csv(
        os.path.join(out_dir, 'backtest_settings_hmm.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    # Following the reference paper, in-sample model comparison uses the full available sample.
    train = data.copy()

    print('2/5 Running full-sample in-sample HMM regime search (K=2 to K=7)')

    ranking_df, chosen = run_hmm_model_search(
        train['log_return'],
        seeds_per_k=N_RANDOM_STARTS_INSAMPLE
    )

    ranking_df.to_csv(
        os.path.join(out_dir, 'all_model_starts_ranked_hmm.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    model_table_rows = []
    regime_tables = []

    for k, best in chosen.items():
        model_table_rows.append({
            'model': 'HMM',
            'k_states': k,
            'best_seed': best.seed,
            'avg_rank': best.avg_rank,
            'logL': best.logL,
            'AIC': best.aic,
            'CD': best.cd
        })

        reg = summarize_regimes(train['log_return'], best.states, best.state_means)
        reg.insert(0, 'k_states', k)
        regime_tables.append(reg)

        plot_price_with_states(
            train.index,
            train[args.close_col.strip().lower()].values,
            best.states,
            os.path.join(out_dir, f'train_state_bands_k{k}.png'),
            f'In-Sample: HMM State Bands (K={k})'
        )

    model_table = pd.DataFrame(model_table_rows).sort_values('k_states')

    model_table.to_csv(
        os.path.join(out_dir, 'in_sample_model_comparison_hmm.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    regime_df = pd.concat(regime_tables, ignore_index=True) if regime_tables else pd.DataFrame()

    regime_df.to_csv(
        os.path.join(out_dir, 'in_sample_regime_summary_hmm.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    if model_table.empty:
        print('3/5 In-sample estimation produced no valid HMM fits. Please inspect warning messages above.')
    else:
        print(
            f"3/5 In-sample estimation complete. "
            f"The state count with the best AIC is: K={int(model_table.sort_values('AIC').iloc[0]['k_states'])}"
        )

    print('4/5 Running full-coverage out-of-sample backtest, including Bench 1 and Bench 2')

    # Run the optimized routine covering both benchmarks and all K values.
    perf = backtest_all_strategies(data, out_dir)

    print("\n--- Final Performance Summary ---")
    print(perf[['Strategy', 'Ann_Return_CAGR', 'Sharpe', 'Max_Drawdown', 'Avg_Exposure']])

    print('\n5/5 Outputs complete!')
    print(f'Please check the [{out_dir}] folder for the complete report outputs, including all figures and charts.')


if __name__ == '__main__':
    main()