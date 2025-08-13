!pip install yfinance pandas --quiet

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# === USER SETTINGS ===
# Leave as "" to use last available trading day automatically
target_date = "2025-08-12"  # e.g. "2025-08-11" for manual mode

# Your tickers
tickers = [
    "HEIA.AS","FDP.AQ","FLTRL.XC","GNCL.XC","GFTUL.XC","TSCOL.XC","BSN.F","GVR.IR","UPR.IR",
    "RYA.IR","PTSB.IR","OIZ.IR","MLC.IR","KRX.IR","KRZ.IR","KMR.IR","IRES.IR","IR5B.IR","HSW.IR",
    "GRP.IR","GL9.IR","EG7.IR","DQ7A.IR","DHG.IR","C5H.IR","A5G.IR","BIRG.IR","VOD.L","DCC.L",
    "HVO.L","POLB.L","ABRONXX","ICON","SBUX","PEP","META","MSFT","INTC","EBAY","COKE","AAPL",
    "AMGN","ADI","GOOG","STT","PFE","ORCL","NVS","MRK","JNJ","HPQ","GE","LLY","BSX","ABBV","ABT",
    "CRH","SW"
]

# === DETERMINE TARGET DATE ===
if not target_date.strip():
    # Find last available trading date from first ticker
    sample_ticker = yf.Ticker(tickers[0])
    hist_sample = sample_ticker.history(period="7d")
    target_dt = hist_sample.index[-1]  # last available date
else:
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")

target_date_str = target_dt.strftime("%Y-%m-%d")
five_days_ago = (target_dt - timedelta(days=7)).strftime("%Y-%m-%d")
day_after_target = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")

print(f"Using target date: {target_date_str}")

# === DATA FETCH ===
rows = []
for ticker in tickers:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=five_days_ago, end=day_after_target)

        if not hist.empty and target_date_str in hist.index.strftime("%Y-%m-%d").tolist():
            close_price = hist.loc[hist.index.strftime("%Y-%m-%d") == target_date_str, "Close"].iloc[0]

            start_close = hist["Close"].iloc[0]
            change_5d = ((close_price - start_close) / start_close) * 100 if start_close else None

            rows.append([
                ticker,
                t.info.get("shortName", ""),
                round(close_price, 4),
                round(change_5d, 2) if change_5d is not None else None
            ])
        else:
            rows.append([ticker, t.info.get("shortName", ""), None, None])
    except Exception:
        rows.append([ticker, "ERROR", None, None])

# === SAVE OUTPUT ===
df = pd.DataFrame(rows, columns=["Ticker", "Company", f"Close ({target_date_str})", "5-Day % Change"])
csv_filename = f"closing_prices_and_5day_change_{target_date_str}.csv"
df.to_csv(csv_filename, index=False)

# Download link for Colab
from google.colab import files
files.download(csv_filename)

df
