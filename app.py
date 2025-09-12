import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import sqlite3
import io
import csv
import os
from typing import Optional

# --- HTTP (requests preferred; fallback to stdlib urllib) ---
try:
    import requests
    _HTTP_LIB = "requests"
except Exception:
    import urllib.request, urllib.parse
    _HTTP_LIB = "urllib"

# --- Timezone helper (ZoneInfo on Python 3.9+) ---
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # fallback to UTC if not available

# --- Exchange calendars (optional) ---
try:
    import exchange_calendars as xcals
    _HAS_XCALS = True
except Exception:
    xcals = None
    _HAS_XCALS = False

DB_PATH = "stocks.db"

# -----------------------------
# Optional S3 backup/restore for DB
# -----------------------------
def _s3_client_from_secrets():
    """
    Creates a boto3 S3 client if secrets are present.
    Expected st.secrets:
      S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, (optional) S3_REGION
    Returns (s3_client, bucket) or (None, None) if not configured / boto3 missing.
    """
    try:
        import boto3
        if "S3_BUCKET" not in st.secrets:
            return None, None
        bucket = st.secrets["S3_BUCKET"]
        ak = st.secrets.get("AWS_ACCESS_KEY_ID")
        sk = st.secrets.get("AWS_SECRET_ACCESS_KEY")
        region = st.secrets.get("S3_REGION", None)
        if not (ak and sk and bucket):
            return None, None
        session = boto3.session.Session(
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            region_name=region,
        )
        return session.client("s3"), bucket
    except Exception:
        return None, None

def restore_db_from_s3():
    s3, bucket = _s3_client_from_secrets()
    if not s3:
        return False
    key = "stocks.db"
    try:
        s3.download_file(bucket, key, DB_PATH)
        return True
    except Exception:
        return False

def backup_db_to_s3():
    s3, bucket = _s3_client_from_secrets()
    if not s3 or not os.path.exists(DB_PATH):
        return False
    key = "stocks.db"
    try:
        s3.upload_file(DB_PATH, bucket, key)
        return True
    except Exception:
        return False

# -----------------------------
# SQLite helpers
# -----------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db_with_defaults():
    conn = get_conn()
    cur = conn.cursor()
    # Stocks table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            ticker TEXT PRIMARY KEY,
            name   TEXT NOT NULL,
            region TEXT NOT NULL,   -- Ireland | UK | Europe | US
            currency TEXT NOT NULL  -- EUR | GBp | USD | DKK | CHF
        )
    """)
    # Manual YTD baselines (one per ticker+year)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reference_prices (
            ticker TEXT NOT NULL,
            year   INTEGER NOT NULL,
            price  REAL NOT NULL,
            date   TEXT,            -- optional yyyy-mm-dd (for your reference)
            series TEXT,            -- optional: 'close'|'adjclose'
            notes  TEXT,            -- optional free text
            PRIMARY KEY (ticker, year)
        )
    """)

    # Seed defaults WITHOUT overwriting user entries
    defaults = [
        # --- US ---
        ("STT","State Street Corporation","US","USD"),
        ("PFE","Pfizer Inc.","US","USD"),
        ("SBUX","Starbucks Corporation","US","USD"),
        ("PEP","PepsiCo, Inc.","US","USD"),
        ("ORCL","Oracle Corporation","US","USD"),
        ("NVS","Novartis AG","US","USD"),
        ("META","Meta Platforms, Inc.","US","USD"),
        ("MSFT","Microsoft Corporation","US","USD"),
        ("MRK","Merck & Co., Inc.","US","USD"),
        ("JNJ","Johnson & Johnson","US","USD"),
        ("INTC","Intel Corporation","US","USD"),
        ("ICON","Icon Energy Corp.","US","USD"),
        ("HPQ","HP Inc.","US","USD"),
        ("GE","GE Aerospace","US","USD"),
        ("LLY","Eli Lilly and Company","US","USD"),
        ("EBAY","eBay Inc.","US","USD"),
        ("COKE","Coca-Cola Consolidated, Inc.","US","USD"),
        ("BSX","Boston Scientific Corporation","US","USD"),
        ("AAPL","Apple Inc.","US","USD"),
        ("AMGN","Amgen Inc.","US","USD"),
        ("ADI","Analog Devices, Inc.","US","USD"),
        ("ABBV","AbbVie Inc.","US","USD"),
        ("GOOG","Alphabet Inc.","US","USD"),
        ("ABT","Abbott Laboratories","US","USD"),
        ("CRH","CRH plc","US","USD"),
        ("SW","Smurfit Westrock Plc","US","USD"),
        ("DEO","Diageo","US","USD"),
        ("AER","AerCap Holdings","US","USD"),
        ("FLUT","Flutter Entertainment plc","US","USD"),

        # --- Europe (non-UK, non-Ireland) ---
        ("HEIA.AS","Heineken N.V.","Europe","EUR"),
        ("BSN.F","Danone S.A.","Europe","EUR"),
        ("BKT.MC","Bankinter","Europe","EUR"),
        # Added primaries:
        ("IBE.MC","Iberdrola S.A.","Europe","EUR"),    # Madrid (primary)
        ("ORSTED.CO","Orsted A/S","Europe","DKK"),     # Copenhagen (primary)
        ("ROG.SW","Roche Holding AG","Europe","CHF"),  # SIX Swiss (primary)
        ("SAN.PA","Sanofi","Europe","EUR"),            # Paris (primary)

        # --- UK ---
        ("VOD.L","Vodafone Group","UK","GBp"),
        ("DCC.L","DCC plc","UK","GBp"),
        ("GNCL.XC","Greencore Group plc","UK","GBp"),
        ("GFTUL.XC","Grafton Group plc","UK","GBp"),
        ("HVO.L","hVIVO plc","UK","GBp"),
        ("POLB.L","Poolbeg Pharma PLC","UK","GBp"),
        ("TSCOL.XC","Tesco plc","UK","GBp"),
        ("BRBY.L","Burberry","UK","GBp"),
        ("SSPG.L","SSP Group","UK","GBp"),
        ("ABF.L","Associated British Foods","UK","GBp"),
        ("GWMO.L","Great Western Mining Corp","UK","GBp"),

        # --- Ireland ---
        ("GVR.IR","Glenveagh Properties PLC","Ireland","EUR"),
        ("UPR.IR","Uniphar plc","Ireland","EUR"),
        ("RYA.IR","Ryanair Holdings plc","Ireland","EUR"),
        ("PTSB.IR","Permanent TSB Group Holdings plc","Ireland","EUR"),
        ("OIZ.IR","Origin Enterprises plc","Ireland","EUR"),
        ("MLC.IR","Malin Corporation plc","Ireland","EUR"),
        ("KRX.IR","Kingspan Group plc","Ireland","EUR"),
        ("KRZ.IR","Kerry Group plc","Ireland","EUR"),
        ("KMR.IR","Kenmare Resources plc","Ireland","EUR"),
        ("IRES.IR","Irish Residential Properties REIT Plc","Ireland","EUR"),
        ("IR5B.IR","Irish Continental Group plc","Ireland","EUR"),
        ("HSW.IR","Hostelworld Group plc","Ireland","EUR"),
        ("GRP.IR","Greencoat Renewables","Ireland","EUR"),
        ("GL9.IR","Glanbia plc","Ireland","EUR"),
        ("EG7.IR","FBD Holdings plc","Ireland","EUR"),
        ("DQ7A.IR","Donegal Investment Group plc","Ireland","EUR"),
        ("DHG.IR","Dalata Hotel Group plc","Ireland","EUR"),
        ("C5H.IR","Cairn Homes plc","Ireland","EUR"),
        ("A5G.IR","AIB Group plc","Ireland","EUR"),
        ("BIRG.IR","Bank of Ireland Group plc","Ireland","EUR"),
        ("YZA.IR","Arytza","Ireland","EUR"),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO stocks (ticker,name,region,currency) VALUES (?,?,?,?)",
        defaults
    )
    conn.commit()
    conn.close()

def db_all_stocks():
    conn = get_conn()
    df = pd.read_sql_query("SELECT ticker,name,region,currency FROM stocks", conn)
    conn.close()
    return df

def db_add_stock(ticker, name, region, currency):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO stocks (ticker,name,region,currency) VALUES (?,?,?,?)",
                (ticker.strip(), name.strip(), region, currency))
    conn.commit()
    conn.close()

def db_remove_stocks(tickers):
    if not tickers:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany("DELETE FROM stocks WHERE ticker = ?", [(t,) for t in tickers])
    conn.commit()
    conn.close()

# ---- reference_prices helpers ----
def db_set_reference(ticker: str, year: int, price: float, date_iso: Optional[str], series: Optional[str], notes: Optional[str]):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reference_prices (ticker,year,price,date,series,notes)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(ticker,year) DO UPDATE SET price=excluded.price,date=excluded.date,series=excluded.series,notes=excluded.notes
    """, (ticker.strip(), int(year), float(price), (date_iso or None), (series or None), (notes or None)))
    conn.commit()
    conn.close()

def db_get_reference(ticker: str, year: int) -> Optional[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT price,date,series,notes FROM reference_prices WHERE ticker=? AND year=?", (ticker.strip(), int(year)))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"price": float(row[0]), "date": row[1], "series": row[2], "notes": row[3]}

def db_all_references(year: Optional[int] = None) -> pd.DataFrame:
    conn = get_conn()
    if year is None:
        df = pd.read_sql_query("SELECT ticker,year,price,date,series,notes FROM reference_prices", conn)
    else:
        df = pd.read_sql_query("SELECT ticker,year,price,date,series,notes FROM reference_prices WHERE year = ?", conn, params=(int(year),))
    conn.close()
    return df

def db_delete_references(keys):
    if not keys:
        return
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany("DELETE FROM reference_prices WHERE ticker=? AND year=?", keys)
    conn.commit()
    conn.close()

# -----------------------------
# Helpers for prices/returns
# -----------------------------
def currency_symbol(cur: str) -> str:
    return {
        "USD": "$",
        "EUR": "€",
        "GBp": "£",
        "DKK": "kr",
        "CHF": "Fr",
    }.get(cur, "")

def _col(use_price_return: bool) -> str:
    # Yahoo UI uses price return => 'Close'; total return => 'Adj Close'
    return "Close" if use_price_return else "Adj Close"

def _session_dates_index(df: pd.DataFrame) -> np.ndarray:
    """Return array of date() for each row; treat rows as local session dates."""
    idx = pd.to_datetime(df.index)
    return np.array([d.date() for d in idx], dtype=object)

def last_close_on_or_before_date(df: pd.DataFrame, target_date: date, use_price_return: bool):
    """Get last session's price on/before target_date using chosen column."""
    if df.empty:
        return None, None
    dates = _session_dates_index(df)
    mask = dates <= target_date
    if not mask.any():
        return None, None
    pos = np.where(mask)[0][-1]
    return float(df.iloc[pos][_col(use_price_return)]), pos

def close_n_trading_days_ago_by_pos(df: pd.DataFrame, pos: int, n: int, use_price_return: bool):
    if df.empty or pos is None:
        return None
    ref_pos = pos - n
    if ref_pos < 0:
        return None
    return float(df.iloc[ref_pos][_col(use_price_return)])

# -----------------------------
# Yahoo chart endpoint for exact YTD
# -----------------------------
def _http_get_json(url: str, params: dict, timeout: float = 10.0) -> Optional[dict]:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        if _HTTP_LIB == "requests":
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        else:
            full = f"{url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(full, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                import json
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

def yahoo_ytd_via_chart(symbol: str, year: int, on_date: date, use_live_when_today: bool = True) -> Optional[float]:
    """
    Compute YTD % using Yahoo's own chart data (daily 'close' series).
    Baseline: last close BEFORE Jan 1 of `year` (local to exchange).
    Numerator: last close ON/BEFORE `on_date` (or live price if today & requested).
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "2y", "interval": "1d", "includePrePost": "false", "events": "div,splits"}
    data = _http_get_json(url, params)
    if not data:
        return None
    try:
        result = data["chart"]["result"][0]
        meta = result.get("meta", {})
        tzname = meta.get("exchangeTimezoneName", "UTC")
        tz = ZoneInfo(tzname) if ZoneInfo else None

        stamps = result.get("timestamp", []) or []
        closes = (result.get("indicators", {}).get("quote", [{}])[0].get("close", []) or [])
        if not stamps or not closes:
            return None

        dcs = []
        for t, c in zip(stamps, closes):
            if c is None:
                continue
            dt = datetime.fromtimestamp(t, tz) if tz else datetime.utcfromtimestamp(t)
            dcs.append((dt.date(), float(c)))
        if not dcs:
            return None

        jan1 = date(year, 1, 1)
        prior = [c for d, c in dcs if d < jan1]
        if not prior:
            in_year = [c for d, c in dcs if d >= jan1]
            if not in_year:
                return None
            base = in_year[0]
        else:
            base = prior[-1]

        last_vals = [c for d, c in dcs if d <= on_date]
        if not last_vals:
            return None
        last_close = last_vals[-1]

        if use_live_when_today and on_date == date.today():
            try:
                fi = yf.Ticker(symbol).fast_info
                live = fi.get("last_price") or fi.get("regular_market_price")
                if live is not None:
                    last_close = float(live)
            except Exception:
                pass

        if base == 0:
            return None
        return (last_close - base) / base * 100.0
    except Exception:
        return None

# -----------------------------
# OFFICIAL EXCHANGE CALENDAR helpers (Option B)
# -----------------------------
# Map Yahoo suffix -> exchange calendar code
CAL_BY_SUFFIX = {
    "IR": "XDUB",  # Dublin
    "PA": "XPAR",  # Paris
    "AS": "XAMS",  # Amsterdam
    "BR": "XBRU",  # Brussels
    "LS": "XLIS",  # Lisbon
    "L":  "XLON",  # London
    "MC": "XMAD",  # Madrid (BME)
    "CO": "XCSE",  # Copenhagen (Nasdaq Nordic)
    "SW": "XSWX",  # SIX Swiss
    "DE": "XETR",  # Xetra (Frankfurt)
    "F":  "XFRA",  # Frankfurt floor
    "MI": "XMIL",  # Milan
}

def _suffix(sym: str) -> str:
    return sym.split(".")[-1].upper() if "." in sym else ""

def ticker_calendar_code(ticker: str) -> Optional[str]:
    suf = _suffix(ticker)
    return CAL_BY_SUFFIX.get(suf)

def official_prev_year_last_session(ticker: str, year: int) -> Optional[date]:
    """Return the official last trading SESSION DATE < Jan 1 of `year` for ticker's venue."""
    if not _HAS_XCALS:
        return None
    cal_code = ticker_calendar_code(ticker)
    if not cal_code:
        return None
    try:
        cal = xcals.get_calendar(cal_code)
        sched = cal.schedule(start=f"{year-1}-12-01", end=f"{year}-01-10")
        idx = sched.index
        if len(idx) == 0:
            return None
        prev = idx[idx.date < date(year, 1, 1)]
        return prev[-1].date() if len(prev) else None
    except Exception:
        return None

def baseline_from_hist_on_or_before(hist: pd.DataFrame, session_date: date, use_price_return: bool) -> Optional[float]:
    """Pick baseline from history at the requested session date (or the last available <= date)."""
    if hist is None or hist.empty:
        return None
    dates = _session_dates_index(hist)
    mask = dates <= session_date
    if not mask.any():
        return None
    pos = np.where(mask)[0][-1]
    try:
        return float(hist.iloc[pos][_col(use_price_return)])
    except Exception:
        return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📊 Stock Dashboard")
st.caption("Last price, 5-day % change, and YTD % change. YTD can use official exchange calendars for the baseline (Europe) or Yahoo's chart feed.")

# Toggles
use_price_return = st.toggle(
    "Match Yahoo style for returns (use Close; live price if today)",
    value=True,
    help="ON = price return (Close). OFF = total return (Adj Close). Live price used for today's numerator."
)
exact_yahoo_mode = st.toggle(
    "Exact Yahoo YTD (chart feed)",
    value=True,
    help="ON = compute YTD from Yahoo's chart endpoint to match their baseline/calendar."
)
use_manual_baselines = st.toggle(
    "Use manual YTD baselines when available",
    value=True,
    help="If a manual baseline exists for (ticker, year), it overrides the automatic YTD baseline."
)
use_official_calendars = st.toggle(
    "Use official exchange calendars for YTD baseline (Europe)",
    value=True,
    help="Computes the baseline as the last official session < Jan 1 per venue (e.g., XDUB/XPAR/XAMS/XMAD/XCSE/XSWX). Falls back if calendar unavailable."
)
# Rounding
round_two_dp = st.toggle(
    "Round to 2 decimal places (off = 1 dp)",
    value=False,
    help="Switch between rounding numbers to 1 or 2 decimal places across Price and % columns."
)
DP = 2 if round_two_dp else 1
price_fmt = f"{{:.{DP}f}}"
pct_fmt   = f"{{:.{DP}f}}"

# Indices
show_indices = st.toggle(
    "Show index 5-day trends (ISEQ, FTSE 100, S&P 500, DAX)",
    value=True
)
show_index_charts = st.checkbox(
    "Mini charts for indices (last ~10 sessions)",
    value=False
)

# ---- Restore DB from cloud (if configured) BEFORE we touch it
if restore_db_from_s3():
    st.info("✅ Restored database from cloud backup.")

init_db_with_defaults()
stocks_df = db_all_stocks()

colA, colB = st.columns([1,1])
with colA:
    selected_date = st.date_input("Select date", value=date.today())
with colB:
    st.write(" ")
    run = st.button("Run")

# Editor: add/remove stocks
with st.expander("➕ Add or ➖ remove stocks (saved to SQLite)"):
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("**Add a stock**")
        a_ticker = st.text_input("Ticker (e.g., AAPL, ORSTED.CO)")
        a_name   = st.text_input("Company name")
        a_region = st.selectbox("Region", ["Ireland", "UK", "Europe", "US"])
        a_curr   = st.selectbox("Currency", ["EUR", "GBp", "USD", "DKK", "CHF"])
        if st.button("Add / Update"):
            if a_ticker and a_name:
                db_add_stock(a_ticker, a_name, a_region, a_curr)
                st.success(f"Saved {a_name} ({a_ticker})")
                if backup_db_to_s3():
                    st.info("☁️ Cloud backup updated.")
            else:
                st.warning("Please provide at least Ticker and Company name.")
    with c2:
        st.markdown("**Remove stocks**")
        rem_choices = [f"{r['name']} ({r['ticker']})" for _, r in stocks_df.sort_values("name").iterrows()]
        rem_sel = st.multiselect("Select to remove", rem_choices, [])
        if st.button("Remove selected"):
            tickers = [s[s.rfind("(")+1:-1] for s in rem_sel]
            db_remove_stocks(tickers)
            st.success(f"Removed {len(tickers)} stock(s)")
            if backup_db_to_s3():
                st.info("☁️ Cloud backup updated.")

    st.markdown("---")
    st.markdown("**Backup & restore (DB)**")
    if st.button("☁️ Backup DB now"):
        if backup_db_to_s3():
            st.success("Cloud backup updated.")
        else:
            st.warning("Cloud backup not configured (add S3_* to st.secrets).")

    st.markdown("**Export / import stock list**")
    # Export current stocks to CSV
    try:
        stocks_now = db_all_stocks().sort_values("name")
        out = io.StringIO()
        stocks_now.to_csv(out, index=False)
        st.download_button("⬇️ Download stocks CSV", out.getvalue(), "stocks.csv", "text/csv")
    except Exception as e:
        st.warning(f"Could not export stocks: {e}")

    # Import stocks from CSV (tolerant)
    up_stocks = st.file_uploader("Upload stocks CSV", type=["csv"], key="stocks_csv")
    if up_stocks is not None:
        try:
            df_imp = pd.read_csv(up_stocks, encoding="utf-8-sig", keep_default_na=False)
            # Accept flexible header cases
            cols = {c.strip().lower(): c for c in df_imp.columns}
            required = {"ticker","name","region","currency"}
            if not required.issubset(set(cols.keys())):
                st.error("CSV must include columns: ticker, name, region, currency")
            else:
                tcol, ncol, rcol, ccol = cols["ticker"], cols["name"], cols["region"], cols["currency"]
                count = 0
                for _, r in df_imp.iterrows():
                    t = str(r[tcol]).strip()
                    n = str(r[ncol]).strip()
                    rg = str(r[rcol]).strip()
                    cu = str(r[ccol]).strip()
                    if t and n and rg and cu:
                        db_add_stock(t, n, rg, cu)
                        count += 1
                st.success(f"Imported/updated {count} stock(s).")
                if backup_db_to_s3():
                    st.info("☁️ Cloud backup updated.")
                st.rerun()
        except Exception as e:
            st.exception(e)

# ---- Manual YTD baseline manager
with st.expander("🧭 Manual YTD baselines (set once at start of year)"):
    cur_year = st.number_input("Year", min_value=2000, max_value=2100, value=selected_date.year, step=1)
    st.caption("Each row defines the **baseline price** used for YTD % for that ticker in this year. Price should match the series you want to mirror (Yahoo typically uses Close).")

    # Quick add form
    c1, c2, c3, c4 = st.columns([1.2, 0.8, 0.8, 1])
    with c1:
        b_ticker = st.text_input("Ticker (exact)", placeholder="A5G.IR")
    with c2:
        b_price = st.text_input("Baseline price", placeholder="e.g. 4.25")
    with c3:
        b_series = st.selectbox("Series", ["close","adjclose"])
    with c4:
        b_date = st.text_input("Baseline date (optional, yyyy-mm-dd)", placeholder="2024-12-27")

    b_notes = st.text_input("Notes (optional)", placeholder="e.g. Dec 27 close from Yahoo")
    if st.button("Add / Update baseline"):
        try:
            price_val = float(b_price)
            db_set_reference(b_ticker, int(cur_year), price_val, b_date.strip() or None, b_series, b_notes.strip() or None)
            st.success(f"Baseline saved for {b_ticker} ({cur_year}): {price_val}")
            if backup_db_to_s3():
                st.info("☁️ Cloud backup updated.")
        except Exception as e:
            st.error(f"Could not save baseline: {e}")

    # CSV import/export (tolerant importer)
    st.markdown("**Bulk import / export**")
    st.caption("CSV columns: ticker, year, price, date (optional), series (close|adjclose, optional), notes (optional)")
    up = st.file_uploader("Upload CSV to import/update baselines", type=["csv"])
    if up is not None:
        try:
            # handle BOM, flexible headers, blanks
            df_imp = pd.read_csv(up, encoding="utf-8-sig", keep_default_na=False)
            # normalize header names (lower + strip)
            name_map = {c: c.strip().lower() for c in df_imp.columns}
            inv = {v: k for k, v in name_map.items()}
            required = {"ticker", "year", "price"}
            if not required.issubset(set(name_map.values())):
                st.error(f"CSV must include columns: {', '.join(sorted(required))}")
            else:
                tick = df_imp[inv["ticker"]].astype(str).str.strip()
                yr   = pd.to_numeric(df_imp[inv["year"]], errors="coerce").astype("Int64")
                pr   = pd.to_numeric(df_imp[inv["price"]], errors="coerce")
                dt   = df_imp[inv["date"]] if "date" in name_map.values() else ""
                ser  = df_imp[inv["series"]] if "series" in name_map.values() else ""
                nts  = df_imp[inv["notes"]] if "notes" in name_map.values() else ""

                norm = pd.DataFrame({
                    "ticker": tick,
                    "year": yr,
                    "price": pr,
                    "date": dt,
                    "series": ser,
                    "notes": nts,
                })

                # drop invalids with a clear message
                bad = norm[norm[["ticker","year","price"]].isna().any(axis=1) | (norm["ticker"] == "")]
                if not bad.empty:
                    st.warning(f"Dropped {len(bad)} invalid row(s) (missing ticker/year/price).")

                norm = norm[(norm["ticker"] != "") & norm["year"].notna() & norm["price"].notna()]
                ok = 0
                for _, r in norm.iterrows():
                    db_set_reference(
                        r["ticker"], int(r["year"]), float(r["price"]),
                        (None if pd.isna(r["date"]) or str(r["date"]).strip()=="" else str(r["date"])),
                        (None if pd.isna(r["series"]) or str(r["series"]).strip()=="" else str(r["series"])),
                        (None if pd.isna(r["notes"]) or str(r["notes"]).strip()=="" else str(r["notes"]))
                    )
                    ok += 1
                st.success(f"Imported/updated {ok} baseline(s).")
                if ok > 0 and backup_db_to_s3():
                    st.info("☁️ Cloud backup updated.")
        except Exception as e:
            # full traceback to actually see what's wrong
            st.exception(e)

    refs_df = db_all_references(cur_year).sort_values(["ticker","year"])
    st.dataframe(refs_df, use_container_width=True)
    if not refs_df.empty:
        out_csv = io.StringIO()
        refs_df.to_csv(out_csv, index=False)
        st.download_button("⬇️ Download current year's baselines CSV", data=out_csv.getvalue(), file_name=f"ytd_baselines_{cur_year}.csv", mime="text/csv")

    # Delete selected
    if not refs_df.empty:
        del_opts = [f"{r['ticker']} ({r['year']})" for _, r in refs_df.iterrows()]
        del_sel = st.multiselect("Delete baselines", del_opts, [])
        if st.button("Delete selected baselines"):
            keys = []
            for s in del_sel:
                t = s[:s.rfind("(")].strip()
                y = int(s[s.rfind("(")+1:-1])
                keys.append((t,y))
            db_delete_references(keys)
            st.success(f"Deleted {len(keys)} baseline(s).")
            if backup_db_to_s3():
                st.info("☁️ Cloud backup updated.")

# Stock selection for this run
stocks_df = db_all_stocks()
stock_options = {f"{r['name']} ({r['ticker']})": dict(r) for _, r in stocks_df.iterrows()}
sel_labels = st.multiselect(
    "Stocks to include in this run:",
    list(stock_options.keys()),
    default=list(stock_options.keys())
)
selected_stocks = [stock_options[label] for label in sel_labels]

# -----------------------------
# Run calculation
# -----------------------------
if run:
    rows = []
    target_dt = pd.to_datetime(selected_date)
    target_date = target_dt.date()
    today_date = date.today()

    # --------- Stocks ----------
    for s in selected_stocks:
        tkr = s["ticker"]
        try:
            # Pull enough history for 5D calculation and display; session-date indexing
            hist = yf.download(
                tkr,
                start=f"{selected_date.year-1}-12-15",
                end=selected_date + timedelta(days=7),
                progress=False,
                auto_adjust=False,
            )
            if hist.empty:
                continue

            # Last session on/before selected date (for display & 5D reference)
            price_eod, pos = last_close_on_or_before_date(hist, target_date, use_price_return)
            if pos is None:
                continue

            # Live price behavior (only when matching Yahoo style AND date is today)
            use_live = use_price_return and (target_date == today_date)
            live_price = None
            if use_live:
                try:
                    fi = yf.Ticker(tkr).fast_info
                    live_price = fi.get("last_price") or fi.get("regular_market_price")
                except Exception:
                    live_price = None

            # Numerator price (today's live if available; else EOD close)
            price_num = float(live_price) if (live_price is not None) else float(price_eod)

            # 5D change uses n sessions back from the EOD position
            c_5ago = close_n_trading_days_ago_by_pos(hist, pos, 5, use_price_return)
            chg_5d = None
            if c_5ago is not None and c_5ago != 0:
                chg_5d = (price_num - c_5ago) / c_5ago * 100.0

            # YTD %  (precedence: Manual → Official calendars → Yahoo chart → fallback)
            manual_used = False
            chg_ytd = None

            # 1) Manual baseline
            manual_ref = db_get_reference(tkr, selected_date.year) if use_manual_baselines else None
            if manual_ref is not None:
                base = float(manual_ref["price"])
                chg_ytd = (price_num - base) / base * 100.0
                manual_used = True
            else:
                # 2) Official exchange calendars
                base_val = None
                baseline_session = None
                if use_official_calendars and _HAS_XCALS:
                    baseline_session = official_prev_year_last_session(tkr, selected_date.year)
                    if baseline_session is not None:
                        base_val = baseline_from_hist_on_or_before(hist, baseline_session, use_price_return)
                # 3) If official available, use it; else 4) Yahoo chart; else 5) fallback
                if base_val is not None and base_val != 0:
                    chg_ytd = (price_num - float(base_val)) / float(base_val) * 100.0
                else:
                    if exact_yahoo_mode:
                        chg_ytd = yahoo_ytd_via_chart(tkr, selected_date.year, target_date, use_live_when_today=use_price_return)
                    else:
                        dates = _session_dates_index(hist)
                        mask_prev = dates <= date(selected_date.year - 1, 12, 31)
                        base_fallback = float(hist.iloc[np.where(mask_prev)[0][-1]][_col(use_price_return)]) if mask_prev.any() else None
                        chg_ytd = ((price_num - base_fallback) / base_fallback * 100.0) if base_fallback else None

            rows.append({
                "Company": s["name"],
                "Manual": "🧭" if manual_used else "",
                "Region": s["region"],
                "Currency": s["currency"],
                "Price": round(price_num, DP),
                "5D % Change": round(chg_5d, DP) if chg_5d is not None else None,
                "YTD % Change": round(chg_ytd, DP) if chg_ytd is not None else None,
            })
        except Exception:
            continue

    # --------- Indices ----------
    if show_indices:
        idx_defs = [
            {"name": "ISEQ All-Share", "ticker": "^ISEQ"},
            {"name": "FTSE 100",       "ticker": "^FTSE"},
            {"name": "S&P 500",        "ticker": "^GSPC"},
            {"name": "DAX",            "ticker": "^GDAXI"},
        ]
        chart_cols = st.columns(len(idx_defs)) if show_index_charts else None
        idx_rows = []
        for i, info in enumerate(idx_defs):
            try:
                h = yf.download(
                    info["ticker"],
                    start=selected_date - timedelta(days=30),
                    end=selected_date + timedelta(days=7),
                    progress=False,
                    auto_adjust=False,
                )
                if h is None or h.empty:
                    continue

                last_lvl, pos_lvl = last_close_on_or_before_date(h, target_date, use_price_return=True)
                if pos_lvl is None:
                    continue

                lvl_5ago = close_n_trading_days_ago_by_pos(h, pos_lvl, 5, use_price_return=True)
                chg_5d_idx = None
                if lvl_5ago is not None and lvl_5ago != 0:
                    chg_5d_idx = (last_lvl - lvl_5ago) / lvl_5ago * 100.0

                idx_rows.append({
                    "Index": info["name"],
                    "Level": round(last_lvl, DP),
                    "5D % Change": round(chg_5d_idx, DP) if chg_5d_idx is not None else None,
                })

                if show_index_charts:
                    series = h["Close"].dropna().tail(10)
                    with chart_cols[i]:
                        st.caption(info["name"])
                        st.line_chart(series)
            except Exception:
                continue

        if idx_rows:
            st.subheader("Major indices — 5-day trend")
            st.dataframe(pd.DataFrame(idx_rows), use_container_width=True)

    # --------- Stocks table / CSV ----------
    if not rows:
        st.warning("No stock data available for that date.")
    else:
        # build & sort the result table
        df = (
            pd.DataFrame(rows)
              .sort_values(by=["Region", "Company"])
              .reset_index(drop=True)
        )

        region_order = ["Ireland", "UK", "Europe", "US"]
        df["Region"] = pd.Categorical(df["Region"], categories=region_order, ordered=True)
        df = df.sort_values(["Region", "Company"])

        # Show a small badge column for manual baselines
        display_cols = ["Company","Manual","Price","5D % Change","YTD % Change"]

        for region in region_order:
            g = df[df["Region"] == region]
            if g.empty:
                continue
            currs = g["Currency"].unique().tolist()
            curr_label = " / ".join(currency_symbol(c) for c in currs if currency_symbol(c))
            header = f"{region} ({curr_label})" if curr_label else region
            st.subheader(header)
            st.dataframe(g[display_cols], use_container_width=True)

        # CSV export (unchanged headers; values formatted using chosen decimal places)
        REGION_LABELS = {
            "Ireland": f"Ireland ({currency_symbol('EUR')})",
            "UK":      f"UK ({currency_symbol('GBp')})",
            "Europe":  f"Europe ({currency_symbol('EUR')})",
            "US":      f"US ({currency_symbol('USD')})",
        }

        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

        for region in region_order:
            g = df[df["Region"] == region]
            if g.empty:
                continue
            writer.writerow([REGION_LABELS[region], "Last price", "5D %change", "YTD % change"])
            for _, row in g.iterrows():
                company = (row["Company"] or "").replace(",", "")
                price = (price_fmt.format(row['Price'])) if pd.notnull(row["Price"]) else ""
                c5 = (pct_fmt.format(row['5D % Change'])) if pd.notnull(row["5D % Change"]) else ""
                cy = (pct_fmt.format(row['YTD % Change'])) if pd.notnull(row["YTD % Change"]) else ""
                writer.writerow([company, price, c5, cy])

        csv_bytes = "\ufeff" + output.getvalue()
        st.download_button("💾 Download CSV", csv_bytes, "stock_data.csv", "text/csv")
