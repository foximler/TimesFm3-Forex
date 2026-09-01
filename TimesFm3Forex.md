```python
import torch
print(torch.xpu.is_available(), torch.xpu.get_device_name(0))
```

    True Intel(R) Arc(TM) Pro B70 Graphics



```python
import modin.pandas as mpd
import numpy as np
import datetime as dt
import ray
colnames=['Date', 'Time', 'Open', 'High','Low','Close','Vol'] 
colnames=['Date', 'Time', 'Open', 'High','Low','Close','Vol'] 


df_usdjpy = pd.read_csv('/home/fox/Ctrader_hist_data/USDJPY-Minute15.csv',names=colnames, header=None)
df_usdjpy['DateTime'] = df_usdjpy['Date'] +"-"+ df_usdjpy["Time"]
df_usdjpy['DateTime'] = df_usdjpy['DateTime'].apply(pd.to_datetime)
df_usdjpy['EMA12'] = df_usdjpy['Close'].ewm(span=12, adjust=False).mean()
df_usdjpy['EMA26'] = df_usdjpy['Close'].ewm(span=26, adjust=False).mean()
df_usdjpy['MACD'] = df_usdjpy['EMA12'] - df_usdjpy['EMA26']
df_usdjpy['signalLine'] = df_usdjpy['MACD'].ewm(span=9, adjust=False).mean()
df_usdjpy['percentChangeHigh'] = df_usdjpy['High'].pct_change(periods = 1)
df_usdjpy['percentChangeLow'] = df_usdjpy['Low'].pct_change(periods = 1)
df_usdjpy['percentChangeClose'] = df_usdjpy['Close'].pct_change(periods = 1)
df_usdjpy['percentChangeOpen'] = df_usdjpy['Open'].pct_change(periods = 1)
df_usdjpy['percentVolSpan'] = abs((df_usdjpy['Close']-df_usdjpy['Open'])/df_usdjpy['Open'])

df_usdjpy = df_usdjpy.drop(["Date","Time"],axis=1).set_index("DateTime")
df_usdjpy = df_usdjpy.iloc[1:]
df_usdjpy = df_usdjpy.dropna()
df_usdjpy['time'] = df_usdjpy.index
df_usdjpy['time'] = df_usdjpy['time'].astype(np.int64) // 10**9
#df_usdjpy = df_usdjpy.replace('.',0)

minute = 60
hour = 60*60
day = 24*60*60
year = (365.2425)*day
df_usdjpy['Hour sin'] = np.sin(df_usdjpy['time'] * (2 * np.pi / hour))
df_usdjpy['Hour cos'] = np.cos(df_usdjpy['time'] * (2 * np.pi / hour))
df_usdjpy['Day sin'] = np.sin(df_usdjpy['time'] * (2 * np.pi / day))
df_usdjpy['Day cos'] = np.cos(df_usdjpy['time'] * (2 * np.pi / day))
df_usdjpy['Year sin'] = np.sin(df_usdjpy['time'] * (2 * np.pi / year))
df_usdjpy['Year cos'] = np.cos(df_usdjpy['time'] * (2 * np.pi / year))

df_usdjpy = df_usdjpy.drop(columns = ['time'])
```

    UserWarning: `Series.ewm` is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: `Series.ewm` is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: `Series.ewm` is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: <function DataFrame.pct_change> is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: <function DataFrame.pct_change> is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: <function DataFrame.pct_change> is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: <function DataFrame.pct_change> is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: __array_ufunc__ is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: __array_ufunc__ is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: __array_ufunc__ is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: __array_ufunc__ is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: __array_ufunc__ is not currently supported by PandasOnRay, defaulting to pandas implementation.
    UserWarning: __array_ufunc__ is not currently supported by PandasOnRay, defaulting to pandas implementation.



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[15], line 45
         41 pdf = df_usdjpy._to_pandas()  # or df.modin.to_pandas() depending on your Modin version
         43 # one series per forecast call
         44 series_list = [group["value"].to_numpy(dtype=np.float32)
    ---> 45                for _, group in pdf.groupby("series_id")]


    File ~/anaconda3/lib/python3.13/site-packages/pandas/core/frame.py:9210, in DataFrame.groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
       9207 if level is None and by is None:
       9208     raise TypeError("You have to supply one of 'by' and 'level'")
    -> 9210 return DataFrameGroupBy(
       9211     obj=self,
       9212     keys=by,
       9213     axis=axis,
       9214     level=level,
       9215     as_index=as_index,
       9216     sort=sort,
       9217     group_keys=group_keys,
       9218     observed=observed,
       9219     dropna=dropna,
       9220 )


    File ~/anaconda3/lib/python3.13/site-packages/pandas/core/groupby/groupby.py:1331, in GroupBy.__init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
       1328 self.dropna = dropna
       1330 if grouper is None:
    -> 1331     grouper, exclusions, obj = get_grouper(
       1332         obj,
       1333         keys,
       1334         axis=axis,
       1335         level=level,
       1336         sort=sort,
       1337         observed=False if observed is lib.no_default else observed,
       1338         dropna=self.dropna,
       1339     )
       1341 if observed is lib.no_default:
       1342     if any(ping._passed_categorical for ping in grouper.groupings):


    File ~/anaconda3/lib/python3.13/site-packages/pandas/core/groupby/grouper.py:1043, in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
       1041         in_axis, level, gpr = False, gpr, None
       1042     else:
    -> 1043         raise KeyError(gpr)
       1044 elif isinstance(gpr, Grouper) and gpr.key is not None:
       1045     # Add key to exclusions
       1046     exclusions.add(gpr.key)


    KeyError: 'series_id'



```python
df = df_usdjpy._to_pandas().sort_index()
```


```python
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Vol</th>
      <th>EMA12</th>
      <th>EMA26</th>
      <th>MACD</th>
      <th>signalLine</th>
      <th>percentChangeHigh</th>
      <th>percentChangeLow</th>
      <th>percentChangeClose</th>
      <th>percentChangeOpen</th>
      <th>percentVolSpan</th>
      <th>Hour sin</th>
      <th>Hour cos</th>
      <th>Day sin</th>
      <th>Day cos</th>
      <th>Year sin</th>
      <th>Year cos</th>
    </tr>
    <tr>
      <th>DateTime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-02-08 08:30:00</th>
      <td>82.061</td>
      <td>82.079</td>
      <td>81.982</td>
      <td>82.015</td>
      <td>234</td>
      <td>82.053077</td>
      <td>82.056667</td>
      <td>-0.003590</td>
      <td>-0.000718</td>
      <td>0.000000</td>
      <td>-0.000829</td>
      <td>-0.000548</td>
      <td>-0.000073</td>
      <td>0.000561</td>
      <td>6.154732e-11</td>
      <td>-1.000000e+00</td>
      <td>0.793353</td>
      <td>-0.608761</td>
      <td>0.613739</td>
      <td>0.789509</td>
    </tr>
    <tr>
      <th>2011-02-08 08:45:00</th>
      <td>82.015</td>
      <td>82.070</td>
      <td>82.005</td>
      <td>82.030</td>
      <td>242</td>
      <td>82.049527</td>
      <td>82.054691</td>
      <td>-0.005165</td>
      <td>-0.001607</td>
      <td>-0.000110</td>
      <td>0.000281</td>
      <td>0.000183</td>
      <td>-0.000561</td>
      <td>0.000183</td>
      <td>-1.000000e+00</td>
      <td>-1.223183e-10</td>
      <td>0.751840</td>
      <td>-0.659346</td>
      <td>0.613880</td>
      <td>0.789399</td>
    </tr>
    <tr>
      <th>2011-02-08 09:00:00</th>
      <td>82.031</td>
      <td>82.089</td>
      <td>82.011</td>
      <td>82.084</td>
      <td>198</td>
      <td>82.054830</td>
      <td>82.056862</td>
      <td>-0.002032</td>
      <td>-0.001692</td>
      <td>0.000232</td>
      <td>0.000073</td>
      <td>0.000658</td>
      <td>0.000195</td>
      <td>0.000646</td>
      <td>-1.830893e-10</td>
      <td>1.000000e+00</td>
      <td>0.707107</td>
      <td>-0.707107</td>
      <td>0.614022</td>
      <td>0.789289</td>
    </tr>
    <tr>
      <th>2011-02-08 09:15:00</th>
      <td>82.087</td>
      <td>82.110</td>
      <td>82.047</td>
      <td>82.068</td>
      <td>178</td>
      <td>82.056856</td>
      <td>82.057687</td>
      <td>-0.000831</td>
      <td>-0.001520</td>
      <td>0.000256</td>
      <td>0.000439</td>
      <td>-0.000195</td>
      <td>0.000683</td>
      <td>0.000231</td>
      <td>1.000000e+00</td>
      <td>-2.218010e-10</td>
      <td>0.659346</td>
      <td>-0.751840</td>
      <td>0.614163</td>
      <td>0.789179</td>
    </tr>
    <tr>
      <th>2011-02-08 09:30:00</th>
      <td>82.068</td>
      <td>82.098</td>
      <td>82.050</td>
      <td>82.096</td>
      <td>114</td>
      <td>82.062878</td>
      <td>82.060525</td>
      <td>0.002353</td>
      <td>-0.000745</td>
      <td>-0.000146</td>
      <td>0.000037</td>
      <td>0.000341</td>
      <td>-0.000231</td>
      <td>0.000341</td>
      <td>-1.610299e-10</td>
      <td>-1.000000e+00</td>
      <td>0.608761</td>
      <td>-0.793353</td>
      <td>0.614304</td>
      <td>0.789069</td>
    </tr>
  </tbody>
</table>
</div>




```python
from timesfm3 import TimesFM3Evaluator, ModelConfig
import inspect

TARGET_COL = "percentChangeClose"

# Columns that are only ever known historically — derived from the
# target itself, so they can't be extended into the forecast horizon
PAST_ONLY_COLS = [
    "Open", "High", "Low", "Vol",
    "EMA12", "EMA26", "MACD", "signalLine",
    "percentChangeHigh", "percentChangeLow",
    "percentChangeClose", "percentChangeOpen", "percentVolSpan",
]

# Pure calendar features — computable for any timestamp, past or future
FUTURE_KNOWN_COLS = [
    "Hour sin", "Hour cos", "Day sin", "Day cos", "Year sin", "Year cos",
]

# ---------------------------------------------------------------
# 2. Train / test split (time-ordered — no shuffling for time series)
# ---------------------------------------------------------------
HORIZON = 1          # how many bars into the future to forecast/test on
CONTEXT_LEN = 128     # how much history to feed as context

assert len(df) > CONTEXT_LEN + HORIZON, "Not enough rows for this context+horizon"

test_df  = df.iloc[-HORIZON:]
train_df = df.iloc[-(CONTEXT_LEN + HORIZON):-HORIZON]   # context window immediately before test

# ---------------------------------------------------------------
# 3. Build target + covariate arrays
# ---------------------------------------------------------------
target_context = train_df[TARGET_COL].to_numpy(dtype=np.float32)
target_actual  = test_df[TARGET_COL].to_numpy(dtype=np.float32)   # held out — used only for scoring

past_covariates = {
    col: train_df[col].to_numpy(dtype=np.float32)
    for col in PAST_ONLY_COLS
}

# future-known covariates need context + horizon concatenated
future_covariates = {
    col: np.concatenate([
        train_df[col].to_numpy(dtype=np.float32),
        test_df[col].to_numpy(dtype=np.float32),   # true future values — legitimate since these are calendar-derived
    ])
    for col in FUTURE_KNOWN_COLS
}

# ---------------------------------------------------------------
# 4. Initialize the model
# ---------------------------------------------------------------
config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=32,
    device="xpu",     # your B70 — fall back to "cpu" if this errors
)
forecaster = TimesFM3Evaluator(config)

# ---------------------------------------------------------------
# 5. Confirm the real covariate call signature before using it
# ---------------------------------------------------------------
print(inspect.signature(forecaster.predict_batch))
# ^ run this once and check the actual parameter names for covariates
#   before trusting the call below — they may not match my guess.

# ---------------------------------------------------------------
# 6. Forecast (adjust kwarg names once step 5 confirms them)
# ---------------------------------------------------------------
outputs = list(forecaster.predict_batch(
    [target_context],
    horizon=HORIZON,
    return_quantiles=True,
    past_covariates=[past_covariates],       # <- confirm this kwarg name
    future_covariates=[future_covariates],   # <- confirm this kwarg name
))

result = outputs[0]
point_forecast = result.forecast          # shape (HORIZON,)
quantiles = result.quantiles              # shape (HORIZON, 9): q10..q90

# ---------------------------------------------------------------
# 7. Evaluate against the held-out test set
# ---------------------------------------------------------------
mae  = np.mean(np.abs(target_actual - point_forecast))
rmse = np.sqrt(np.mean((target_actual - point_forecast) ** 2))
mape = np.mean(np.abs((target_actual - point_forecast) / target_actual)) * 100

q10, q90 = quantiles[:, 0], quantiles[:, 8]
coverage_80 = np.mean((target_actual >= q10) & (target_actual <= q90)) * 100

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"80% PI coverage: {coverage_80:.1f}%")
```

    (contexts: 'list[np.ndarray]', horizon: 'int', past_only_covariates: 'list[np.ndarray | None] | None' = None, past_future_covariates: 'list[np.ndarray | None] | None' = None, ts_ids: 'list[str] | None' = None, return_quantiles: 'bool' = True, use_symmetric_averaging: 'bool' = True, make_positive: 'bool' = True, sort_quantiles: 'bool' = True, use_znorm: 'bool' = False, padding_mode: 'str' = 'none', univariate: 'bool' = False) -> 'Iterator[ForecastOutput]'



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[67], line 71
         64 print(inspect.signature(forecaster.predict_batch))
         65 # ^ run this once and check the actual parameter names for covariates
         66 #   before trusting the call below — they may not match my guess.
         67 
         68 # ---------------------------------------------------------------
         69 # 6. Forecast (adjust kwarg names once step 5 confirms them)
         70 # ---------------------------------------------------------------
    ---> 71 outputs = list(forecaster.predict_batch(
         72     [target_context],
         73     horizon=HORIZON,
         74     return_quantiles=True,
         75     past_covariates=[past_covariates],       # <- confirm this kwarg name
         76     future_covariates=[future_covariates],   # <- confirm this kwarg name
         77 ))
         79 result = outputs[0]
         80 point_forecast = result.forecast          # shape (HORIZON,)


    TypeError: TimesFM3Evaluator.predict_batch() got an unexpected keyword argument 'past_covariates'. Did you mean 'past_only_covariates'?



```python
HORIZON = 4
CONTEXT_LEN = 128
N_TESTS = 200

actuals = []
medians = []
q_lo = []
q_hi = []

for i in range(N_TESTS):
    end = len(df) - i
    start = end - CONTEXT_LEN - HORIZON

    if start < 0:
        print(f"Ran out of data after {i} tests")
        break

    ctx = df[TARGET_COL].iloc[start:end - HORIZON].to_numpy(dtype=np.float32)
    actual = df[TARGET_COL].iloc[end - 1]   # the bar HORIZON steps after context ends

    out = list(forecaster.predict_batch(
        contexts=[ctx],
        horizon=HORIZON,
        return_quantiles=True,
    ))[0]

    actuals.append(actual)
    medians.append(out.forecast[-1])       # last step of the horizon, not first
    q_lo.append(out.quantiles[-1, 0])
    q_hi.append(out.quantiles[-1, 8])

actuals = np.array(actuals)
medians = np.array(medians)
q_lo = np.array(q_lo)
q_hi = np.array(q_hi)

mae  = np.mean(np.abs(actuals - medians))
rmse = np.sqrt(np.mean((actuals - medians) ** 2))
coverage_80 = np.mean((actuals >= q_lo) & (actuals <= q_hi)) * 100

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"80% PI coverage: {coverage_80:.1f}%")
```

    MAE:  0.0005
    RMSE: 0.0007
    80% PI coverage: 79.0%



```python
naive_zero_mae = np.mean(np.abs(actuals))           # predicting "no change"
naive_zero_rmse = np.sqrt(np.mean(actuals ** 2))

naive_prev_pred = df[TARGET_COL].shift(HORIZON).iloc[-N_TESTS:].to_numpy()
naive_prev_mae = np.mean(np.abs(actuals - naive_prev_pred[::-1]))

print(f"Naive (zero return)     MAE={naive_zero_mae:.6f}")
print(f"Naive (repeat last ret) MAE={naive_prev_mae:.6f}")
```

    Naive (zero return)     MAE=0.000504
    Naive (repeat last ret) MAE=0.000711



```python
def run_backtest(use_past_only=False, use_past_future=False, horizon=4, n_tests=200, context_len=128, debug=True):
    actuals, medians, q_lo, q_hi = [], [], [], []

    for i in range(n_tests):
        end = len(df) - i
        start = end - context_len - horizon
        if start < 0:
            break

        train_slice = df.iloc[start:end - horizon]
        test_slice  = df.iloc[end - horizon:end]

        ctx = train_slice[TARGET_COL].to_numpy(dtype=np.float32)
        actual = df[TARGET_COL].iloc[end - 1]

        if debug and i == 0:
            print("Context last timestamp:", train_slice.index[-1])
            print("Context last value:    ", ctx[-1])
            print("Target timestamp:      ", df.index[end - 1])
            print("Target value:          ", actual)
            print("Gap (bars) between context end and target:",
                  df.index.get_loc(df.index[end - 1]) - df.index.get_loc(train_slice.index[-1]))

        kwargs = dict(contexts=[ctx], horizon=horizon, return_quantiles=True)

        if use_past_only:
            kwargs["past_only_covariates"] = [np.stack(
                [train_slice[c].to_numpy(dtype=np.float32) for c in PAST_ONLY_COLS], axis=0
            )]
        if use_past_future:
            kwargs["past_future_covariates"] = [np.stack(
                [np.concatenate([train_slice[c].to_numpy(dtype=np.float32),
                                  test_slice[c].to_numpy(dtype=np.float32)])
                 for c in FUTURE_KNOWN_COLS], axis=0
            )]

        out = list(forecaster.predict_batch(**kwargs))[0]

        actuals.append(actual)
        medians.append(out.forecast[-1])
        q_lo.append(out.quantiles[-1, 0])
        q_hi.append(out.quantiles[-1, 8])

    actuals, medians = np.array(actuals), np.array(medians)
    q_lo, q_hi = np.array(q_lo), np.array(q_hi)

    mae = np.mean(np.abs(actuals - medians))
    rmse = np.sqrt(np.mean((actuals - medians) ** 2))
    coverage = np.mean((actuals >= q_lo) & (actuals <= q_hi)) * 100
    return mae, rmse, coverage

results = {
    "none":         run_backtest(False, False),
    "past_only":    run_backtest(True,  False),
    "past_future":  run_backtest(False, True),
    "both":         run_backtest(True,  True),
}


```

    Context last timestamp: 2022-07-06 22:30:00
    Context last value:     0.00030180567
    Target timestamp:       2022-07-06 23:30:00
    Target value:           -0.0002940722388453665
    Gap (bars) between context end and target: 4
    Context last timestamp: 2022-07-06 22:30:00
    Context last value:     0.00030180567
    Target timestamp:       2022-07-06 23:30:00
    Target value:           -0.0002940722388453665
    Gap (bars) between context end and target: 4
    Context last timestamp: 2022-07-06 22:30:00
    Context last value:     0.00030180567
    Target timestamp:       2022-07-06 23:30:00
    Target value:           -0.0002940722388453665
    Gap (bars) between context end and target: 4
    Context last timestamp: 2022-07-06 22:30:00
    Context last value:     0.00030180567
    Target timestamp:       2022-07-06 23:30:00
    Target value:           -0.0002940722388453665
    Gap (bars) between context end and target: 4
    none          MAE=0.0005  RMSE=0.0007  coverage=79.0%
    past_only     MAE=0.0005  RMSE=0.0007  coverage=77.5%
    past_future   MAE=0.0005  RMSE=0.0007  coverage=81.5%
    both          MAE=0.0005  RMSE=0.0007  coverage=78.5%



```python
for name, (mae, rmse, cov) in results.items():
    print(f"{name:12s}  MAE={mae:.8f}  RMSE={rmse:.8f}  coverage={cov:.1f}%")

```

    none          MAE=0.00050937  RMSE=0.00070099  coverage=79.0%
    past_only     MAE=0.00050614  RMSE=0.00069800  coverage=77.5%
    past_future   MAE=0.00051203  RMSE=0.00070732  coverage=81.5%
    both          MAE=0.00050587  RMSE=0.00069872  coverage=78.5%



```python
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
 

```


```python
TARGET_COL = "percentChangeClose"
CONTEXT_LEN = 128
HORIZON = 4
N_TESTS = 200
 
DEVICE = "xpu" if torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
 
# Reference numbers from your TimesFM-3 runs, for the final table
TIMESFM3_RESULTS = {
    "TimesFM-3 (none)":        dict(mae=0.00050937, rmse=0.00070099),
    "TimesFM-3 (past_only)":   dict(mae=0.00050614, rmse=0.00069800),
    "TimesFM-3 (past_future)": dict(mae=0.00051203, rmse=0.00070732),
    "TimesFM-3 (both)":        dict(mae=0.00050587, rmse=0.00069872),
    "Naive (zero return)":     dict(mae=0.000504,   rmse=None),
    "Naive (repeat last)":     dict(mae=0.000711,   rmse=None),
}
 

series = df[TARGET_COL].to_numpy(dtype=np.float32)
n = len(series)
 
# Reserve the same last N_TESTS + HORIZON + CONTEXT_LEN rows as your
# TimesFM-3 test region — LSTM training must not touch this range.
test_region_start = n - (N_TESTS + HORIZON + CONTEXT_LEN)
if test_region_start < CONTEXT_LEN + HORIZON:
    raise ValueError("Not enough data for this CONTEXT_LEN/HORIZON/N_TESTS combination")
 
train_series = series[:test_region_start]
print(f"Total rows: {n} | Training rows: {len(train_series)} | Reserved test region: {n - test_region_start}")
 
# -----------------------------------------------------------------
# 2. Build sliding windows (context -> horizon-ahead single target)
#    Matches TimesFM-3's setup: predict the value HORIZON steps after
#    the context window ends.
# -----------------------------------------------------------------
def make_windows(arr, context_len, horizon):
    X, y = [], []
    for i in range(len(arr) - context_len - horizon + 1):
        X.append(arr[i:i + context_len])
        y.append(arr[i + context_len + horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
 
X_all, y_all = make_windows(train_series, CONTEXT_LEN, HORIZON)
print(f"Built {len(X_all)} training windows")
 
# Time-ordered train/val split — last 15% of windows held out for
# early stopping, no shuffling (would leak future into training).
val_frac = 0.15
split_idx = int(len(X_all) * (1 - val_frac))
X_train, y_train = X_all[:split_idx], y_all[:split_idx]
X_val, y_val = X_all[split_idx:], y_all[split_idx:]
 
X_train_t = torch.from_numpy(X_train).unsqueeze(-1).to(DEVICE)   # (N, seq_len, 1)
y_train_t = torch.from_numpy(y_train).unsqueeze(-1).to(DEVICE)
X_val_t = torch.from_numpy(X_val).unsqueeze(-1).to(DEVICE)
y_val_t = torch.from_numpy(y_val).unsqueeze(-1).to(DEVICE)
 
print(f"Train windows: {len(X_train)} | Val windows: {len(X_val)}")
 
# -----------------------------------------------------------------
# 3. Model — kept small and regularized on purpose. Given how close
#    to the noise floor this series has been in every prior test,
#    a large model is much more likely to overfit than to find signal.
# -----------------------------------------------------------------
class SimpleLSTM(nn.Module):
    def __init__(self, n_features=1, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
 
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))
 
model = SimpleLSTM().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()
 
# -----------------------------------------------------------------
# 4. Train with early stopping on validation loss
# -----------------------------------------------------------------
BATCH_SIZE = 2048
MAX_EPOCHS = 100
PATIENCE = 10
 
best_val_loss = float("inf")
best_state = None
patience_counter = 0
 
n_train = len(X_train_t)
 
for epoch in range(MAX_EPOCHS):
    model.train()
    perm = torch.randperm(n_train)
    train_loss = 0.0
 
    for i in range(0, n_train, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb, yb = X_train_t[idx], y_train_t[idx]
 
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(idx)
 
    train_loss /= n_train
 
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = loss_fn(val_pred, y_val_t).item()
 
    if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.8f} | val_loss={val_loss:.8f}")
 
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.8f})")
            break
 
model.load_state_dict(best_state)
print(f"Loaded best model (val_loss={best_val_loss:.8f})")
 
# -----------------------------------------------------------------
# 5. Walk-forward backtest on the reserved test region — identical
#    structure to your TimesFM-3 loop, walking backward from the end.
# -----------------------------------------------------------------
model.eval()
actuals, preds = [], []
 
for i in range(N_TESTS):
    end = n - i
    start = end - CONTEXT_LEN - HORIZON
    if start < test_region_start - CONTEXT_LEN - HORIZON:
        # safety guard — shouldn't trigger given how test_region_start was chosen
        break
 
    ctx = series[start:end - HORIZON]
    actual = series[end - 1]
 
    with torch.no_grad():
        x = torch.from_numpy(ctx).unsqueeze(0).unsqueeze(-1).to(DEVICE)
        pred = model(x).item()
 
    actuals.append(actual)
    preds.append(pred)
 
actuals = np.array(actuals)
preds = np.array(preds)
 
lstm_mae = np.mean(np.abs(actuals - preds))
lstm_rmse = np.sqrt(np.mean((actuals - preds) ** 2))
 
per_sample_abs_err = np.abs(actuals - preds)
per_sample_abs_err_naive_zero = np.abs(actuals)  # naive-zero error = |actual|
 
wilcoxon_stat, wilcoxon_p = stats.wilcoxon(per_sample_abs_err, per_sample_abs_err_naive_zero)
 
# -----------------------------------------------------------------
# 6. Final comparison
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print(f"{'Model':28s} {'MAE':>12s} {'RMSE':>12s}")
print("-" * 60)
print(f"{'LSTM (this run)':28s} {lstm_mae:12.8f} {lstm_rmse:12.8f}")
for name, r in TIMESFM3_RESULTS.items():
    rmse_str = f"{r['rmse']:.8f}" if r["rmse"] is not None else "     n/a"
    print(f"{name:28s} {r['mae']:12.8f} {rmse_str:>12s}")
print("=" * 60)
 
print(f"\nWilcoxon signed-rank test, LSTM |error| vs naive-zero |error|:")
print(f"  statistic={wilcoxon_stat:.2f}  p-value={wilcoxon_p:.4f}")
if wilcoxon_p < 0.05:
    winner = "LSTM" if lstm_mae < np.mean(per_sample_abs_err_naive_zero) else "naive-zero"
    print(f"  -> statistically significant difference (p<0.05), {winner} is better on this test set")
else:
    print(f"  -> no statistically significant difference from naive-zero (p>=0.05)")

```

    Using device: xpu
    Total rows: 283716 | Training rows: 283384 | Reserved test region: 332
    Built 283253 training windows
    Train windows: 240765 | Val windows: 42488
    Epoch   0 | train_loss=0.00007082 | val_loss=0.00000019
    Epoch   5 | train_loss=0.00000034 | val_loss=0.00000019
    Epoch  10 | train_loss=0.00000035 | val_loss=0.00000018
    Early stopping at epoch 11 (best val_loss=0.00000018)
    Loaded best model (val_loss=0.00000018)
    
    ============================================================
    Model                                 MAE         RMSE
    ------------------------------------------------------------
    LSTM (this run)                0.00050491   0.00069207
    TimesFM-3 (none)               0.00050937   0.00070099
    TimesFM-3 (past_only)          0.00050614   0.00069800
    TimesFM-3 (past_future)        0.00051203   0.00070732
    TimesFM-3 (both)               0.00050587   0.00069872
    Naive (zero return)            0.00050400          n/a
    Naive (repeat last)            0.00071100          n/a
    ============================================================
    
    Wilcoxon signed-rank test, LSTM |error| vs naive-zero |error|:
      statistic=9605.00  p-value=0.5871
      -> no statistically significant difference from naive-zero (p>=0.05)



```python

```
