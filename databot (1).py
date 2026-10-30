"""
DataBot - ask plain-English questions about ANY table (CSV or Excel).  ONE file.

    pip install pandas matplotlib            (+ openpyxl if you use .xlsx files)
    python databot.py mydata.csv             # chat
    python databot.py mydata.csv --ask "average salary by department"

How it works (no training needed):
  1. It profiles your table: which columns are numbers, dates, categories, names/ids.
  2. It turns your question into a small, safe "query plan" (what to calculate,
     which groups, which filters) using built-in language rules. These are instant,
     free and need no internet.
  3. If the rules cannot understand a question, an optional AI (free local Ollama, or an
     Anthropic key) writes the same kind of plan. The AI never sees your rows, only column
     names and a few example values, and never runs code. pandas does all the maths.
  4. You get a direct, specific answer first. Charts are drawn only if you ask for one
     ("plot", "chart", "graph").
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import urllib.error
import urllib.request
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # charts are optional
    plt = None

CHART_DIR = Path("charts")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# --------------------------------------------------------------------------- language data
STOP = set(
    """a an the of in on at to for from with and or is are was were be been being do does did what whats
    which who whom whose how many much show me tell give list find get can could would you please i we us
    our my your it its this that these those there their them they than then as by per each every across
    over under above below between among within into about vs versus compare has have had having made make
    sold sell selling earn earns earned paid pay all any some also just only too very really value comes come coming made""".split()
)
GENERIC = {"id", "no", "number", "name", "date", "type", "code", "count", "value", "the", "of", "total", "amount"}
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}
MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
               "oct": 10, "nov": 11, "dec": 12})
MAIN_METRIC_HINTS = ["revenue", "sales", "amount", "income", "salary", "profit", "price", "cost", "score",
                      "marks", "gpa", "value", "total"]
ALIAS = {  # words people use -> regex for the real column name
    "money": r"revenue|sales|amount|income|salary|price|cost|total",
    "sale": r"revenue|sales|amount",
    "turnover": r"revenue|sales|amount",
    "income": r"revenue|income|salary|earning",
    "earning": r"salary|income|earning|revenue",
    "pay": r"salary|pay|wage|income", "wage": r"salary|wage|pay", "paid": r"salary|wage|pay",
    "cost": r"cost|price|expense", "expense": r"expense|cost",
    "mark": r"marks|score|grade|gpa|result", "score": r"score|marks|grade|gpa|rating",
    "grade": r"grade|gpa|marks|score", "result": r"result|marks|score|gpa",
    "old": r"\bage\b|^age", "age": r"age",
}
NON_ADDITIVE = re.compile(r"age|score|rating|price|rate|margin|gpa|percent|pct|ratio|temperature|marks|grade|"
                          r"salary|attendance|experience|performance|discount|weight|height|speed", re.I)
ID_NAME = re.compile(r"(?:^|[_\s])(id|phone|mobile|cnic|zip|postal|code|sku|barcode|account|ref)(?:$|[_\s])|"
                     r"(?:^|[_\s])(no|number)$|roll", re.I)
TIME_PART = re.compile(r"^(year|yr|fy|month|quarter|week|day|semester|term|grade|class)$", re.I)

R_COUNT = r"\b(how many|count|number of|no of|headcount)\b"
R_MEAN = r"\b(average|avg|mean|typical|on average)\b"
R_MEDIAN = r"\bmedian\b"
R_STD = r"\b(std|standard deviation|variance|spread|variability|volatility)\b"
R_SUM = r"\b(total|sum|overall|altogether|combined|cumulative)\b"
R_SHARE = r"(%|\bpercent\w*|\bshare\b|\bproportion\b|\bcontribut\w*|\bratio\b|\bfraction\b)"
R_HIGH = r"\b(highest|largest|biggest|greatest|maximum|max|most|top|best|leading|peak|richest|oldest|longest|" \
         r"tallest|heaviest|costliest|priciest|strongest)\b"
R_LOW = r"\b(lowest|smallest|minimum|min|least|bottom|worst|weakest|poorest|youngest|shortest|cheapest|fewest|" \
        r"lightest)\b"
R_TREND = r"\b(trend\w*|over time|growing|growth|declin\w*|increas\w*|decreas\w*|evolution|progress\w*|" \
          r"going up|going down|rising|falling|improv\w*)\b"
R_CHART = r"\b(chart|plot|graph|visuali[sz]e|visuali[sz]ation|draw|diagram|pie|histogram|bar chart|line chart)\b"
GRAINS = {"day": "day", "days": "day", "daily": "day", "week": "week", "weeks": "week", "weekly": "week",
          "month": "month", "months": "month", "monthly": "month", "quarter": "quarter", "quarters": "quarter",
          "quarterly": "quarter", "year": "year", "years": "year", "yearly": "year", "annual": "year",
          "annually": "year", "weekday": "weekday", "weekdays": "weekday"}
BY_MARK = {"by", "per", "each", "every", "across", "which", "what", "top", "bottom", "who", "grouped",
           "breakdown", "among", "highest", "lowest", "best", "worst", "most", "least"}


# --------------------------------------------------------------------------- helpers
def st(t):
    """Very small singulariser so 'products' == 'product'."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith(("ches", "shes", "sses", "xes")):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def words_of(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
    return re.findall(r"[a-z0-9]+", s.lower())


def nice(col):
    return str(col).replace("_", " ")


def parse_num(txt):
    t = txt.lower().replace(",", "").strip()
    mult = 1
    if t.endswith("%"):
        t = t[:-1]
    if t and t[-1] in "kmb":
        mult = {"k": 1e3, "m": 1e6, "b": 1e9}[t[-1]]
        t = t[:-1]
    return float(t) * mult


def fmtn(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, (np.integer, int)) and not pct:
        return f"{int(x):,}"
    x = float(x)
    if pct:
        return f"{x:,.1f}%"
    if x.is_integer() and abs(x) < 1e15:
        return f"{int(x):,}"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:,.1f}"
    return f"{x:,.2f}"


@dataclass
class Plan:
    kind: str = "scalar"  # scalar group rows corr trend summary describe columns quality distinct outliers
    agg: str = "sum"  # sum mean median min max count nunique std share
    metric: str | None = None
    by: list = field(default_factory=list)
    grain: str | None = None  # day week month quarter year weekday monthname
    filters: list = field(default_factory=list)  # {"col","op","value","label"}
    sort_desc: bool = True
    limit: int | None = None
    chart: bool = False
    cols2: list = field(default_factory=list)
    ranked: bool = False
    note: str = ""
    noun: str = "records"
    ignored: list = field(default_factory=list)


def render_table(t):
    t = t.copy()
    for c in t.columns:
        if str(c) == "share %":
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}%")
        elif pd.api.types.is_datetime64_any_dtype(t[c]):
            t[c] = t[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_numeric_dtype(t[c]):
            t[c] = t[c].map(lambda v: "" if pd.isna(v) else fmtn(v))
        else:
            t[c] = t[c].astype(str).str.slice(0, 32)
    return t.to_string(index=False)


@dataclass
class Answer:
    text: str
    table: pd.DataFrame | None = None
    chart: str | None = None
    show_table: bool = True
    notes: list = field(default_factory=list)

    def __str__(self):
        out = self.text
        if self.show_table and self.table is not None and len(self.table) > 0:
            out += "\n\n" + render_table(self.table)
        for n in self.notes:
            out += f"\n\n({n})"
        return out


# =========================================================================== the bot
class DataBot:
    def __init__(self, source, sheet=None, api_key=None, ai="auto"):
        self.name = "data"
        self.df = self._load(source, sheet)
        self._profile()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.provider = None
        if ai != "off":
            if self.api_key:
                self.provider = "anthropic"
            elif self._ollama_running():
                self.provider = "ollama"
        self.ollama_model = OLLAMA_MODEL
        self.last_plan = None
        self.last_table = None
        self.last_chart = None

    # ------------------------------------------------------------------ loading
    def _load(self, source, sheet):
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            p = Path(source)
            self.name = p.stem
            if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
                df = pd.read_excel(p, sheet_name=sheet or 0)
            else:
                try:
                    df = pd.read_csv(p)
                except UnicodeDecodeError:
                    df = pd.read_csv(p, encoding="latin-1")
                if df.shape[1] == 1:  # maybe ; or tab separated
                    df = pd.read_csv(p, sep=None, engine="python")
        df = df.dropna(how="all").dropna(axis=1, how="all")
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        seen, cols = {}, []
        for c in df.columns:
            c = str(c).strip() or "column"
            seen[c] = seen.get(c, 0) + 1
            cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
        df.columns = cols
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ profiling
    def _profile(self):
        df = self.df
        self.roles, self.pct_cols = {}, set()
        self.date_note = ""
        n = len(df)
        for c in df.columns:
            s = df[c]
            idname = bool(ID_NAME.search(c))
            if pd.api.types.is_bool_dtype(s):
                self.roles[c] = "category"
            elif pd.api.types.is_datetime64_any_dtype(s):
                self.roles[c] = "date"
            elif pd.api.types.is_numeric_dtype(s):
                nn = s.dropna()
                if idname or (nn.nunique() == len(nn) and len(nn) > 20 and nn.dtype.kind in "iu"):
                    self.roles[c] = "entity"
                elif TIME_PART.match(c) and nn.nunique() <= 40:
                    self.roles[c] = "category"
                else:
                    self.roles[c] = "number"
            else:
                role = self._text_role(c, s, idname)
                self.roles[c] = role
        self.num_cols = [c for c, r in self.roles.items() if r == "number"]
        self.cat_cols = [c for c, r in self.roles.items() if r == "category"]
        self.ent_cols = [c for c, r in self.roles.items() if r == "entity"]
        dates = [c for c, r in self.roles.items() if r == "date"]
        pref = [c for c in dates if re.search(r"date|time|created|order", c, re.I)]
        self.date_col = (pref or dates or [None])[0]
        self.anchor = df[self.date_col].max() if self.date_col else None
        self.metric = None
        for h in MAIN_METRIC_HINTS:
            hit = next((c for c in self.num_cols if h in c.lower()), None)
            if hit:
                self.metric = hit
                break
        if not self.metric and self.num_cols:
            self.metric = self.num_cols[0]
        # value index for filters: "lahore" -> [(region, "Lahore")]
        self.index = defaultdict(list)
        for c in self.cat_cols + self.ent_cols:
            if df[c].nunique() > 6000:
                continue
            for v in df[c].dropna().unique():
                key = " ".join(words_of(v))
                if not key or len(key) < 2:
                    continue
                if key.isdigit() and not (len(key) == 4 and re.search(r"year|yr|fy", c, re.I)):
                    continue
                for k in {key, " ".join(key.split()[:-1] + [st(key.split()[-1])])}:
                    if (c, v) not in self.index[k]:
                        self.index[k].append((c, v))
        # column words
        self.col_words = {c: [st(w) for w in words_of(c)] for c in df.columns}
        self.tok2cols = defaultdict(set)
        for c, ws in self.col_words.items():
            for w in ws:
                if w not in GENERIC:
                    self.tok2cols[w].add(c)

    def _text_role(self, c, s, idname):
        nn = s.dropna().astype(str)
        if nn.empty:
            return "category"
        if not idname:
            t = nn.str.strip().str.replace(r"[,\s]", "", regex=True).str.replace(
                r"^(Rs\.?|PKR|USD|\$|£|€)", "", regex=True, case=False)
            pct = t.str.endswith("%")
            num = pd.to_numeric(t.str.rstrip("%"), errors="coerce")
            if num.notna().mean() >= 0.9:
                self.df[c] = pd.to_numeric(self.df[c].astype(str).str.strip().str.replace(
                    r"[,\s]", "", regex=True).str.replace(r"^(Rs\.?|PKR|USD|\$|£|€)", "", regex=True,
                                                           case=False).str.rstrip("%"), errors="coerce")
                if pct.mean() > 0.5:
                    self.pct_cols.add(c)
                return "number"
            if nn.str.contains(r"\d").mean() > 0.9 and nn.str.len().mean() < 30:
                dayfirst = False
                m = nn.str.extract(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})")
                if m.notna().all(axis=1).mean() > 0.9:
                    a, b = pd.to_numeric(m[0], errors="coerce"), pd.to_numeric(m[1], errors="coerce")
                    if a.max() > 12:
                        dayfirst = True
                    elif b.max() > 12:
                        dayfirst = False
                    else:
                        dayfirst = True
                        self.date_note = (f"Dates in '{c}' like {nn.iloc[0]} are read as day/month/year "
                                          "(they are ambiguous).")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        parsed = pd.to_datetime(self.df[c], errors="coerce", format="mixed", dayfirst=dayfirst)
                    except (TypeError, ValueError):
                        parsed = pd.to_datetime(self.df[c], errors="coerce", dayfirst=dayfirst)
                if parsed.notna().sum() / len(nn) >= 0.9:
                    self.df[c] = parsed
                    return "date"
        k = nn.nunique()
        ratio = k / len(nn)
        if idname or (k > 30 and re.search(r"(^|[_\s])name$|^name", c, re.I)):
            return "entity"
        if k <= 50 or ratio <= 0.2:
            return "category"
        if ratio >= 0.6:
            return "entity"
        return "category" if k <= 300 else "entity"

    # ------------------------------------------------------------------ AI settings
    @staticmethod
    def _ollama_running():
        try:
            with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=1.2):
                return True
        except Exception:
            return False

    def set_key(self, key=None):
        self.api_key = key or getpass.getpass("Paste your API key (hidden): ").strip()
        self.provider = "anthropic" if self.api_key else self.provider
        print("AI backup is ON (Claude)." if self.api_key else "No key entered.")

    def use_ollama(self, model=None):
        if model:
            self.ollama_model = model
        if not self._ollama_running():
            print("Ollama is not running. Install it from ollama.com and open it first.")
            return
        self.provider = "ollama"
        print(f"AI backup is ON (free local model: {self.ollama_model}).")

    # ================================================================== PUBLIC API
    def ask(self, question):
        """Return the answer as text (and remember the table / chart for later)."""
        return str(self.answer(question))

    def say(self, question):
        """Print the answer; show the chart inside Jupyter if one was requested."""
        ans = self.answer(question)
        print(ans)
        if ans.chart:
            try:
                from IPython.display import Image, display

                display(Image(ans.chart))
            except Exception:
                pass
        return None

    def answer(self, question):
        q = (question or "").strip()
        if not q:
            return Answer("Ask me something about your data. Type 'examples' to see ideas.")
        ql = q.lower()
        if ql in {"help", "?", "examples", "example", "what can you do", "what can i ask"}:
            return Answer(self.help_text())
        plan, how = self._interpret(q)
        if plan is None:
            return Answer(how)
        try:
            ans = self._run(plan)
        except Exception as e:  # never crash the chat
            return Answer(f"Sorry, I could not calculate that ({type(e).__name__}: {e}).")
        self.last_plan = plan
        if plan.chart:
            ans.chart = self._chart(ans, plan)
            self.last_chart = ans.chart
            if ans.chart:
                ans.notes.append(f"chart saved: {ans.chart}")
        self.last_table = ans.table
        if any(f.get("rel") for f in plan.filters):
            ans.notes.append(f"'last' / 'this' are measured from the latest date in your data ({self.anchor:%Y-%m-%d})")
        if how:
            ans.notes.append(how)
        return ans

    def show_chart(self):
        if not self.last_chart:
            print("No chart yet. Ask with the word 'chart', for example: revenue by month chart")
            return
        from IPython.display import Image, display

        display(Image(self.last_chart))

    def columns(self):
        lines = [f"{len(self.df):,} rows x {len(self.df.columns)} columns"]
        names = {"number": "numbers", "category": "categories", "entity": "names / ids", "date": "dates"}
        for role in ("number", "category", "date", "entity"):
            cols = [c for c, r in self.roles.items() if r == role]
            if cols:
                lines.append(f"  {names[role]:<12}: {', '.join(cols)}")
        if self.date_note:
            lines.append("  note: " + self.date_note)
        return "\n".join(lines)

    def examples(self):
        ex = []
        m = self.metric
        cat = next((c for c in self.cat_cols if 2 <= self.df[c].nunique() <= 15), None)
        cat2 = next((c for c in self.cat_cols if c != cat and 2 <= self.df[c].nunique() <= 15), None)
        ent = self.ent_cols[0] if self.ent_cols else None
        if m and cat:
            agg = "average" if NON_ADDITIVE.search(m) else "total"
            ex.append(f"{agg} {nice(m)} by {nice(cat)}")
            v = self.df[cat].dropna().astype(str).value_counts().index[0]
            ex.append(f"how many records are in {v}?")
            ex.append(f"which {nice(cat)} has the highest {nice(m)}?")
            ex.append(f"{agg} {nice(m)} for {v}")
        if m and self.date_col:
            ex.append(f"{nice(m)} trend by month")
        if len(self.num_cols) >= 2:
            ex.append(f"is there a relationship between {nice(self.num_cols[0])} and {nice(self.num_cols[1])}?")
        if m and ent and self.df[ent].is_unique:
            ex.append(f"who has the highest {nice(m)}?")
        if m:
            ex.append(f"list the top 5 by {nice(m)}")
            ex.append(f"how many have {nice(m)} above average?")
        if cat2:
            ex.append(f"count by {nice(cat2)}")
        ex.append("any missing values?")
        return ex[:10]

    def help_text(self):
        ex = "\n".join(f"  - {e}" for e in self.examples())
        return (f"{self.columns()}\n\nTry questions like:\n{ex}\n\n"
                "Add the word 'chart' to any question if you want a graph.")

    # ================================================================== INTERPRET
    def _interpret(self, q):
        plan, conf = self._parse_rules(q)
        lp = self.last_plan
        if plan is not None and plan.note == "profile" and lp is not None and lp.kind in ("scalar", "group", "trend", "rows"):
            import copy

            new = copy.deepcopy(lp)
            cols = {f["col"] for f in plan.filters}
            new.filters = [f for f in new.filters if f["col"] not in cols] + plan.filters
            new.note = ""
            return new, "following on from your previous question"
        if plan is not None and plan.ignored and plan.kind == "scalar" and plan.agg == "count" and not plan.filters:
            conf = min(conf, 0.4)
            if not self.provider:
                return None, (f"I did not recognise '{', '.join(plan.ignored)}', so I am not sure what to count. "
                              "Try naming a column or value from your data.\n\n" + self.help_text())
        if plan is not None and conf >= 0.55:
            if plan.ignored and not self.provider:
                return plan, "I did not recognise: " + ", ".join(plan.ignored) + ". If the answer looks off, rephrase using your column names"
            return plan, ""
        if self.provider:
            try:
                p2 = self._llm_plan(q)
                if p2:
                    p2.chart = p2.chart or bool(re.search(R_CHART, q.lower()))
                    return p2, "understood with the AI backup"
            except RuntimeError as e:
                if plan is not None:
                    return plan, f"AI backup unavailable: {e}"
                why_ = "the AI backup could not read it" if "not understood" in str(e) else f"the AI backup is unavailable: {e}"
                return None, f"I could not understand that ({why_}). Try rephrasing with your column names.\n\n" + self.help_text()
        if plan is not None:
            return plan, "I was not fully sure what you meant, so this is my best reading"
        return None, "I could not work out what to calculate.\n\n" + self.help_text()

    # ------------------------------------------------------------------ rule parser
    def _parse_rules(self, q):
        ql = q.lower().replace("’", "'")
        ql = re.sub(r"\bq([1-4])\b", r"quarter\1", ql)
        toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"[a-z0-9]+(?:[.,][0-9]+)*%?", ql)]
        if not toks:
            return None, 0
        T = [t for t, _, _ in toks]
        S = [st(t) for t in T]
        used = [False] * len(T)
        plan = Plan()
        plan.chart = bool(re.search(R_CHART, ql))
        for i, t in enumerate(T):
            if re.fullmatch(R_CHART, t) or t in {"show", "me", "bar", "line", "pie"}:
                used[i] = True

        # ---- 1. values mentioned (Lahore, Laptop, Finance ...)
        value_hits = []
        for n in (4, 3, 2, 1):
            for i in range(len(T) - n + 1):
                if any(used[i:i + n]):
                    continue
                gram = " ".join(T[i:i + n])
                gram2 = " ".join(T[i:i + n - 1] + [st(T[i + n - 1])])
                hit = self.index.get(gram) or self.index.get(gram2)
                if not hit:
                    continue
                if n == 1 and (T[i] in STOP or T[i] in GROUP_WORDS):
                    continue
                if n == 1 and re.fullmatch(r"\d+", T[i]) and not any(
                        re.search(r"year|yr|fy", c, re.I) for c, _ in hit):
                    continue
                pick = next((h for h in hit if self.roles[h[0]] == "category"), hit[0])
                value_hits.append((i, pick[0], pick[1]))
                for j in range(i, i + n):
                    used[j] = True
        value_hits.sort()

        # ---- 2. numeric comparisons (salary above 100000, age under 30 ...)
        num_filters = self._parse_comparisons(ql, toks, used)

        # ---- 3. dates
        date_filters, grain_from_dates = self._parse_dates(ql, T, used)

        # ---- 4. column mentions
        mentions = self._mentions(S, used)

        # ---- 5. intent words
        count_q = bool(re.search(R_COUNT, ql))
        distinct_q = bool(re.search(r"\b(unique|distinct|different)\b", ql))
        has_mean = bool(re.search(R_MEAN, ql))
        has_median = bool(re.search(R_MEDIAN, ql))
        has_std = bool(re.search(R_STD, ql))
        has_sum = bool(re.search(R_SUM, ql))
        has_share = bool(re.search(R_SHARE, ql))
        high = bool(re.search(R_HIGH, ql))
        low = bool(re.search(R_LOW, ql))
        rank_n = None
        m = re.search(r"\b(?:top|bottom|first|last|best|worst|highest|lowest)\s+(\d+)\b(?!\s*(?:day|week|month|year))", ql)
        if m:
            rank_n = int(m.group(1))
            for i, t in enumerate(T):
                if t == m.group(1):
                    used[i] = True
        for rx in (R_COUNT, R_MEAN, R_MEDIAN, R_STD, R_SUM, R_SHARE, R_TREND,
                   r"\b(correlat\w*|relationship|related|relate|affect\w*|impact\w*|influence\w*|depend\w*|associat\w*|linked|connection|drives?)\b",
                   r"\b(outlier\w*|anomal\w*|unusual|abnormal|weird|strange|suspicious|missing|null|blank|empty|duplicate\w*|quality|summary|summari[sz]e|overview|describe|profile)\b",
                   r"\b(compare|comparison|versus|vs|difference|differ|against|unique|distinct|different|list|display|print|fetch|select|find|get|give|show|who|whom|whose|people|records|rows|data|please|percentage|percent|distribution|histogram|statistics|stats|spread|range)\b"):
            for mt in re.finditer(rx, ql):
                self._mark(toks, used, mt.start(), mt.end())
        who = bool(re.search(r"\b(who|whom|whose)\b", ql))
        listing = bool(re.search(r"\b(list|display|print|fetch|select|find|get|show|give)\b", ql)) and not (
                count_q or has_mean or has_sum or has_median)
        for i, t in enumerate(T):
            if t in GRAINS or t in BY_MARK or t in STOP or re.fullmatch(R_HIGH, t) or re.fullmatch(R_LOW, t):
                used[i] = True

        mcols = {mm["col"]: mm for mm in mentions}
        num_m = [mm for mm in mentions if self.roles[mm["col"]] == "number"]
        dim_m = [mm for mm in mentions if self.roles[mm["col"]] in ("category", "entity", "date")]
        cmp_cols = {f["col"] for f in num_filters}

        # ---- 6. group-by columns
        by = []
        strong = {"by", "per", "each", "every", "across", "which", "top", "bottom", "who", "grouped", "breakdown", "among"}
        for mm in dim_m:
            before = {S[j] for j in range(max(0, mm["pos"] - 3), mm["pos"])} | {T[j] for j in
                                                                              range(max(0, mm["pos"] - 3), mm["pos"])}
            after = T[mm["pos"] + mm["n"]: mm["pos"] + mm["n"] + 1]
            markers = strong if self.roles[mm["col"]] == "entity" else BY_MARK
            if (before & markers) or after == ["wise"]:
                if mm["col"] not in by:
                    by.append(mm["col"])
        # time grain
        grain = grain_from_dates
        if self.date_col:
            for i, t in enumerate(T):
                if t in GRAINS:
                    marker = bool({S[j] for j in range(max(0, i - 3), i)} & (BY_MARK | {"over", "in", "for"})) or \
                             t in {"monthly", "weekly", "daily", "quarterly", "yearly", "annually", "annual"}
                    # month/year used as a plain unit ("last year") is already consumed by the date parser
                    if marker and not self._is_column_word(t):
                        grain = GRAINS[t]
                        if self.date_col not in by:
                            by.append(self.date_col)
                        break
            if self.date_col in by and not grain:
                span = (self.df[self.date_col].max() - self.df[self.date_col].min()).days
                grain = "day" if span <= 62 else "month"
        # two values of the same column + compare words -> compare them
        cmp_word = bool(re.search(r"\b(compare|comparison|versus|vs|difference|differ|against)\b", ql))
        by_filter_cols = defaultdict(list)
        for _, c, v in value_hits:
            by_filter_cols[c].append(v)
        for c, vs in by_filter_cols.items():
            if len(vs) >= 2 and c not in by:
                by.append(c)
        if cmp_word and len(by_filter_cols) == 1 and not by:
            by = [next(iter(by_filter_cols))]
        if len(date_filters_years(date_filters)) >= 2 and self.date_col and not by:
            by, grain = [self.date_col], "year"

        # yes/no columns mentioned without a value ("how many employees left?") -> filter on the "yes" value
        neg = bool(re.search(r"\b(not|never|without|didn't|didnt|no longer|haven't|hasn't)\b", ql))
        relation_q = bool(re.search(r"\b(correlat\w*|relationship|related|relate|affect\w*|impact\w*|influence\w*|"
                                    r"depend\w*|associat\w*|compare|versus|vs|difference)\b", ql))
        have = {c for _, c, _ in value_hits}
        for mm in mentions:
            c = mm["col"]
            if relation_q or c in by or c in have or self.roles[c] != "category":
                continue
            vals = {str(v).lower(): v for v in self.df[c].dropna().unique()}
            if 1 <= len(vals) <= 2 and set(vals) <= {"yes", "no", "true", "false", "y", "n", "1", "0"}:
                pos = next((v for k, v in vals.items() if k in ("yes", "true", "y", "1")), None)
                negv = next((v for k, v in vals.items() if k in ("no", "false", "n", "0")), None)
                want = negv if neg else pos
                if want is not None:
                    value_hits.append((mm["pos"], c, want))
                    have.add(c)
        # "how many <noun>": a category noun ("how many departments") means distinct values
        noun = "records"
        mcount = re.search(r"\b(?:how many|number of|count of|count)\s+([a-z]+)", ql)
        if count_q and mcount:
            w = mcount.group(1)
            if w not in STOP and w not in GRAINS and w not in {"different", "unique", "distinct", "records", "rows"}:
                col = next(iter(self.tok2cols.get(st(w), [])), None) if len(self.tok2cols.get(st(w), [])) == 1 else None
                if col and self.roles[col] == "category" and col not in by and not value_hits \
                        and not any(f["col"] == col for f in num_filters):
                    plan.kind, plan.metric = "distinct", col
                    return plan, 0.95
                noun = w
                if st(w) in self.tok2cols:
                    nxt = re.search(rf"\b{re.escape(w)}\s+([a-z]+)", ql)
                    if nxt and nxt.group(1) not in STOP:
                        noun = nxt.group(1)
        plan.noun = noun
        # ---- 7. metric
        metric = None
        for mm in num_m:
            if mm["col"] not in cmp_cols or (len(num_m) == 1 and not count_q and (has_mean or has_sum or has_median or has_std)):
                metric = mm["col"]
                break
        if metric is None and not count_q and not distinct_q:
            for i, s in enumerate(S):
                if s in ALIAS:
                    cand = self._alias_col(s, avoid=cmp_cols)
                    if cand:
                        metric = cand
                        break
        if metric is None and not count_q and not distinct_q and re.search(r"\b(sold|selling|sell)\b", ql):
            metric = next((c for c in self.num_cols if re.search(r"quant|units|qty", c, re.I) and c not in cmp_cols), None)
        # "best month", "worst region", "top product": a ranking with no metric means the main value column,
        # unless it is about counting things ("which department has the most employees")
        if metric is None and by and not count_q and not distinct_q and (high or low) and agg_free(ql):
            metric = self.metric
        # ---- 8. kind
        filters = []
        by_col_vals = defaultdict(list)
        for _, c, v in value_hits:
            by_col_vals[c].append(v)
        for c, vs in by_col_vals.items():
            filters.append({"col": c, "op": "==" if len(vs) == 1 else "in", "value": vs[0] if len(vs) == 1 else vs})
        filters += num_filters + date_filters
        plan.filters = filters
        plan.by = by
        plan.grain = grain

        content = [T[i] for i in range(len(T)) if not used[i] and len(T[i]) >= 3 and not re.fullmatch(r"[\d.,%]+", T[i])]
        explained_cols = {mm["col"] for mm in mentions}
        unexplained = [w for i, w in enumerate(T) if not used[i] and st(w) not in ALIAS and st(w) not in self.tok2cols and len(w) >= 3
                       and not re.fullmatch(r"[\d.,%]+", w) and not any(mm["pos"] <= i < mm["pos"] + mm["n"] for mm in mentions)]
        conf = max(0.0, 1.0 - 0.18 * len(unexplained))
        plan.ignored = [w for w in unexplained if len(w) >= 4]

        gen_words = set(T)
        if re.search(r"\b(columns?|fields?|headers?|variables?|attributes)\b", ql) and len(T) <= 8 and not mentions:
            plan.kind = "columns"
            return plan, 1.0
        if re.search(r"\b(summary|summari[sz]e|overview|describe|profile|about this)\b", ql) and not mentions \
                and not value_hits:
            plan.kind = "summary"
            return plan, 1.0
        if re.search(r"\b(missing|null|blank|empty|duplicate\w*|data quality|clean|nan)\b", ql):
            plan.kind = "quality"
            return plan, 1.0
        if re.search(r"\boutlier\w*|anomal\w*|unusual|abnormal|weird|strange|suspicious\b", ql):
            plan.kind = "outliers"
            plan.metric = metric or self.metric
            return plan, 0.9 if plan.metric else 0.3
        if re.search(r"\b(correlat\w*|relationship|related|relate|affect\w*|impact\w*|influence\w*|depend\w*|"
                     r"associat\w*|linked|connection|drives?)\b", ql):
            cols = [mm["col"] for mm in mentions if self.roles[mm["col"]] in ("number", "category")]
            if len(cols) >= 2:
                plan.kind, plan.cols2 = "corr", cols[:2]
                return plan, 0.95
        if distinct_q and (dim_m or mentions):
            plan.kind = "distinct"
            plan.metric = (dim_m or mentions)[0]["col"]
            return plan, 0.95
        if re.search(R_TREND, ql) and self.date_col and not has_mean:
            plan.kind, plan.metric = "trend", metric or self.metric
            plan.by = [self.date_col]
            plan.grain = grain or self._auto_grain()
            plan.agg = self._default_agg(plan.metric, has_mean, has_median, has_sum)
            return plan, conf if plan.metric else 0.3

        # aggregation choice
        agg = None
        if count_q:
            agg = "count"
        elif has_mean:
            agg = "mean"
        elif has_median:
            agg = "median"
        elif has_std:
            agg = "std"
        elif has_sum:
            agg = "sum"
        elif has_share:
            agg = "share"

        # entity lookup rows ("who has the highest salary", "list employees in finance earning > ...")
        ent_by = [c for c in by if self.roles[c] == "entity" and self.df[c].nunique() >= 0.9 * len(self.df)]
        if re.search(r"\b(distribution|histogram|statistics|stats|spread|range)\b", ql) and (metric or self.metric) \
                and not by:
            plan.kind, plan.metric = "describe", metric or self.metric
            return plan, 0.9
        if who or ent_by or (rank_n and not by and (metric or self.metric)) or \
                (listing and not by and (filters or mentions or "all" in T)):
            if metric is None and (high or low):
                metric = self.metric
            plan.kind = "rows"
            plan.metric = metric
            plan.by = []
            plan.sort_desc = not low
            plan.ranked = bool((high or low) and metric)
            plan.limit = rank_n or (1 if (who and plan.ranked) else 10)
            plan.note = "|".join(mm["col"] for mm in mentions)
            return plan, conf

        if agg is None and metric is None and not by:
            if filters:
                if num_filters:
                    plan.kind, plan.limit = "rows", 10
                    return plan, conf
                plan.kind, plan.agg = "scalar", "count"
                plan.note = "profile"
                return plan, conf
            return None, 0.0

        if agg is None:
            agg = self._default_agg(metric, False, False, False) if metric else "count"
        if agg == "count" and metric and distinct_q:
            agg = "nunique"

        plan.agg = agg
        plan.metric = metric if agg != "count" else None
        if agg in ("sum", "mean", "median", "std") and plan.metric is None:
            plan.metric = self.metric
            if plan.metric is None:
                return None, 0.0
        if by:
            plan.kind = "group"
            plan.ranked = bool((high or low) or rank_n)
            plan.sort_desc = not (low and not high)
            if rank_n:
                plan.limit = rank_n
            elif plan.ranked:
                plan.limit = 1
            if (high or low) and agg in ("max", "min"):
                plan.agg = self._default_agg(metric, False, False, False)
        else:
            plan.kind = "scalar"
            if (high or low) and agg in (None, "sum") and plan.metric and not has_sum:
                plan.agg = "max" if high and not low else "min"
        if plan.agg == "share" and not by and not filters:
            plan.agg = "sum"
        return plan, conf

    def _alias_col(self, word, avoid=()):
        for alt in ALIAS[word].split("|"):
            hit = next((c for c in self.num_cols if re.search(alt, c, re.I) and c not in avoid), None)
            if hit:
                return hit
        return None

    def _is_column_word(self, t):
        return any(cw == [st(t)] for c, cw in self.col_words.items() if c != self.date_col)

    def _auto_grain(self):
        span = (self.df[self.date_col].max() - self.df[self.date_col].min()).days
        return "day" if span <= 62 else "month" if span <= 1100 else "quarter" if span <= 3650 else "year"

    def _default_agg(self, metric, mean, median, summ):
        if mean:
            return "mean"
        if median:
            return "median"
        if summ:
            return "sum"
        if metric and NON_ADDITIVE.search(metric):
            return "mean"
        return "sum"

    # ---- mentions of columns
    def _mentions(self, S, used):
        found = {}
        for col, cw in self.col_words.items():
            n = len(cw)
            if not n:
                continue
            for i in range(len(S) - n + 1):
                if S[i:i + n] == cw and not any(used[i:i + n]):
                    found[col] = {"col": col, "pos": i, "n": n, "score": 3}
                    break
        for col, cw in self.col_words.items():
            if col in found or len(cw) < 2:
                continue
            pos = [S.index(w) for w in cw if w in S and not used[S.index(w)]]
            if len(pos) == len(cw):
                found[col] = {"col": col, "pos": min(pos), "n": 1, "score": 2}
        for i, s in enumerate(S):
            if used[i]:
                continue
            cols = self.tok2cols.get(s)
            if cols and len(cols) == 1:
                col = next(iter(cols))
                found.setdefault(col, {"col": col, "pos": i, "n": 1, "score": 1})
            elif len(s) >= 5 and not cols and s not in STOP:
                close = get_close_matches(s, list(self.tok2cols), n=1, cutoff=0.84)
                if close and len(self.tok2cols[close[0]]) == 1:
                    col = next(iter(self.tok2cols[close[0]]))
                    found.setdefault(col, {"col": col, "pos": i, "n": 1, "score": 0.8})
        return sorted(found.values(), key=lambda m: m["pos"])

    # ---- numeric comparisons
    def _parse_comparisons(self, ql, toks, used):
        out = []
        pats = [
            (r"between\s+(-?[\d,.]+[kmb]?)\s+(?:and|to)\s+(-?[\d,.]+[kmb]?)", "between"),
            (r"(?:at least|minimum of|no less than|>=)\s*(-?[\d,.]+[kmb]?%?)", ">="),
            (r"(?:at most|maximum of|no more than|up to|<=)\s*(-?[\d,.]+[kmb]?%?)", "<="),
            (r"(?:more than|greater than|higher than|bigger than|larger than|above|over|exceeding|exceeds|older than|>)"
             r"\s*(-?[\d,.]+[kmb]?%?)", ">"),
            (r"(?:less than|fewer than|below|under|lower than|smaller than|younger than|<)\s*(-?[\d,.]+[kmb]?%?)", "<"),
            (r"(?:equal to|equals|exactly|=)\s*(-?[\d,.]+[kmb]?%?)", "=="),
        ]
        avg_pat = r"(more than|greater than|higher than|above|over|below|under|less than|lower than)\s+" \
                  r"(?:the\s+)?(average|mean|median)"
        S = [st(t) for t, _, _ in toks]
        mentions = self._mentions(S, [False] * len(S))
        num_mentions = [(mm, toks[mm["pos"]][1]) for mm in mentions if self.roles[mm["col"]] == "number"]

        def nearest_col(start, ql_):
            before = [mm for mm, cs in num_mentions if cs < start]
            if before:
                return before[-1]["col"]
            after = [mm for mm, cs in num_mentions if cs >= start]
            if after:
                return after[0]["col"]
            for s_ in S:
                if s_ in ALIAS:
                    cand = self._alias_col(s_)
                    if cand:
                        return cand
            if re.search(r"older|younger", ql_):
                cand = next((c for c in self.num_cols if re.search(r"age", c, re.I)), None)
                if cand:
                    return cand
            return self.metric

        for m in re.finditer(avg_pat, ql):
            col = nearest_col(m.start(), ql)
            if col is None:
                continue
            ref = self.df[col].mean() if m.group(2) != "median" else self.df[col].median()
            op = ">" if m.group(1) in ("more than", "greater than", "higher than", "above", "over") else "<"
            out.append({"col": col, "op": op, "value": float(ref), "label": f"{nice(col)} {op} {m.group(2)} ({fmtn(ref)})"})
            self._mark(toks, used, m.start(), m.end())
        for pat, op in pats:
            for m in re.finditer(pat, ql):
                if any(used[i] for i, (_, a, b) in enumerate(toks) if a < m.end() and b > m.start()):
                    continue
                col = nearest_col(m.start(), ql)
                if col is None:
                    continue
                try:
                    if op == "between":
                        v = [parse_num(m.group(1)), parse_num(m.group(2))]
                        lab = f"{nice(col)} between {fmtn(v[0])} and {fmtn(v[1])}"
                    else:
                        v = parse_num(m.group(1))
                        lab = f"{nice(col)} {op} {fmtn(v)}"
                except ValueError:
                    continue
                out.append({"col": col, "op": op, "value": v, "label": lab})
                self._mark(toks, used, m.start(), m.end())
        return out

    @staticmethod
    def _mark(toks, used, a, b):
        for i, (_, s, e) in enumerate(toks):
            if s < b and e > a:
                used[i] = True

    # ---- dates
    def _parse_dates(self, ql, T, used):
        if not self.date_col:
            return [], None
        col, anchor = self.date_col, self.anchor
        filters, grain = [], None
        D = lambda y, m=1, d=1: pd.Timestamp(year=y, month=m, day=d)

        def rng(lo, hi, label):
            return {"col": col, "op": "between", "value": [lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d")],
                    "label": label}

        def mark_words(*ws):
            for i, t in enumerate(T):
                if t in ws:
                    used[i] = True

        # relative phrases (relative to the LATEST date in the data)
        m = re.search(r"\b(?:last|past|previous)\s+(\d+)\s+(day|week|month|year)s?\b", ql)
        if m:
            n, u = int(m.group(1)), m.group(2)
            lo = anchor - {"day": pd.Timedelta(days=n), "week": pd.Timedelta(weeks=n),
                           "month": pd.DateOffset(months=n), "year": pd.DateOffset(years=n)}[u]
            filters.append(rng(lo + pd.Timedelta(days=1), anchor, f"last {n} {u}s up to {anchor:%Y-%m-%d}"))
            mark_words(m.group(1), "last", "past", "previous", m.group(2), m.group(2) + "s")
        rel_before = len(filters)
        for pat, fn in [
            (r"\b(?:this|current)\s+year\b", lambda: rng(D(anchor.year), anchor, f"{anchor.year}")),
            (r"\b(?:last|previous)\s+year\b", lambda: rng(D(anchor.year - 1), D(anchor.year - 1, 12, 31),
                                                          f"{anchor.year - 1}")),
            (r"\b(?:this|current)\s+month\b", lambda: rng(D(anchor.year, anchor.month), anchor,
                                                           anchor.strftime("%B %Y"))),
            (r"\b(?:last|previous)\s+month\b", lambda: self._prev_month(anchor, rng)),
            (r"\b(?:this|current)\s+quarter\b", lambda: self._quarter(anchor.year, (anchor.month - 1) // 3 + 1, rng)),
            (r"\b(?:last|previous)\s+quarter\b", lambda: self._prev_quarter(anchor, rng)),
            (r"\bytd\b|\byear to date\b", lambda: rng(D(anchor.year), anchor, f"{anchor.year} year to date")),
            (r"\b(?:last|previous)\s+week\b", lambda: rng(anchor - pd.Timedelta(days=6), anchor, "last 7 days")),
        ]:
            if re.search(pat, ql):
                filters.append(fn())
                mark_words("this", "current", "last", "previous", "year", "month", "quarter", "week", "ytd", "to", "date")
                break
        if filters:
            for f in filters[rel_before:] if False else filters:
                f["rel"] = True
            return filters, grain

        years = [(i, int(t)) for i, t in enumerate(T) if re.fullmatch(r"(?:19|20)\d\d", t) and not used[i]]
        months = [(i, MONTHS[t]) for i, t in enumerate(T) if t in MONTHS and not used[i]
                  and not (t == "may" and not years)]
        quarters = [(i, int(t[-1])) for i, t in enumerate(T) if re.fullmatch(r"quarter[1-4]", t)]
        since = re.search(r"\b(since|from|after)\s+(?:\w+\s+)?((?:19|20)\d\d)\b", ql)
        until = re.search(r"\b(before|until|till|prior to)\s+(?:\w+\s+)?((?:19|20)\d\d)\b", ql)
        if since and years:
            y = int(since.group(2)) + (1 if since.group(1) == "after" else 0)
            filters.append(rng(D(y), self.df[col].max(), f"{'after ' + str(y - 1) if since.group(1) == 'after' else 'since ' + str(y)}"))
            for i, _ in years:
                used[i] = True
            mark_words("since", "from", "after")
            return filters, grain
        if until and years:
            y = int(until.group(2))
            filters.append(rng(self.df[col].min(), D(y - 1, 12, 31), f"before {y}"))
            for i, _ in years:
                used[i] = True
            mark_words("before", "until", "till", "prior")
            return filters, grain
        if quarters:
            y = years[0][1] if years else anchor.year
            filters.append(self._quarter(y, quarters[0][1], rng))
            for i, _ in quarters + years:
                used[i] = True
            return filters, grain
        pairs = [(mi, mn, yv) for mi, mn in months for yi, yv in years if yi == mi + 1]
        if len(months) >= 2 and len(pairs) == len(months):
            rngs, labels = [], []
            for mi, mn, yv in pairs:
                lo = D(yv, mn)
                rngs.append([lo.strftime("%Y-%m-%d"), (lo + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")])
                labels.append(lo.strftime("%b %Y"))
                used[mi] = used[mi + 1] = True
            filters.append({"col": col, "op": "ranges", "value": rngs, "label": " + ".join(labels)})
            return filters, "month"
        if months:
            if len(months) == 1 and len(years) <= 1:
                mnum = months[0][1]
                if years:
                    y = years[0][1]
                    lo = D(y, mnum)
                    filters.append(rng(lo, lo + pd.offsets.MonthEnd(0), lo.strftime("%B %Y")))
                else:
                    filters.append({"col": col, "op": "month_is", "value": [mnum],
                                    "label": f"{pd.Timestamp(2000, mnum, 1):%B} (all years)"})
            else:
                filters.append({"col": col, "op": "month_is", "value": sorted({m for _, m in months}),
                                "label": "months " + ", ".join(f"{pd.Timestamp(2000, m, 1):%b}" for _, m in months)})
                grain = "monthname"
            for i, _ in months + years:
                used[i] = True
            return filters, grain
        if years:
            ys = sorted({y for _, y in years})
            filters.append(rng(D(ys[0]), D(ys[-1], 12, 31), f"{ys[0]}" if len(ys) == 1 else f"{ys[0]} to {ys[-1]}"))
            if len(ys) > 1:
                filters[-1]["years"] = ys
            for i, _ in years:
                used[i] = True
        return filters, grain

    @staticmethod
    def _prev_month(anchor, rng):
        first = pd.Timestamp(year=anchor.year, month=anchor.month, day=1)
        lo = first - pd.DateOffset(months=1)
        return rng(lo, first - pd.Timedelta(days=1), lo.strftime("%B %Y"))

    @staticmethod
    def _quarter(y, qn, rng):
        lo = pd.Timestamp(year=y, month=3 * (qn - 1) + 1, day=1)
        return rng(lo, lo + pd.DateOffset(months=3) - pd.Timedelta(days=1), f"Q{qn} {y}")

    def _prev_quarter(self, anchor, rng):
        qn = (anchor.month - 1) // 3 + 1
        y, qn = (anchor.year, qn - 1) if qn > 1 else (anchor.year - 1, 4)
        return self._quarter(y, qn, rng)

    # ================================================================== AI planner
    def _schema_for_ai(self):
        lines = []
        for c in self.df.columns:
            r = self.roles[c]
            if r == "number":
                s = self.df[c]
                lines.append(f"- {c} (number, range {fmtn(s.min())} to {fmtn(s.max())})")
            elif r == "date":
                s = self.df[c]
                lines.append(f"- {c} (date, {s.min():%Y-%m-%d} to {s.max():%Y-%m-%d})")
            elif r == "category":
                vals = ", ".join(map(str, self.df[c].dropna().unique()[:12]))
                lines.append(f"- {c} (category, e.g. {vals})")
            else:
                lines.append(f"- {c} (name/id text)")
        return "\n".join(lines)

    def _llm_plan(self, q):
        system = (
            "You convert a question about a table into a JSON query plan. Reply with ONLY one JSON object.\n"
            "TABLE COLUMNS:\n" + self._schema_for_ai() + "\n\n"
            'Fields: "kind": scalar|group|rows|corr|trend|summary|describe|columns|quality|distinct|outliers; '
            '"agg": sum|mean|median|min|max|count|nunique|std|share; "metric": numeric column or null; '
            '"by": list of up to 2 columns to group by; "grain": day|week|month|quarter|year|weekday|monthname|null '
            "(only when grouping by the date column); "
            '"filters": list of {"col","op","value"} with op one of == in != > >= < <= between '
            "(for dates use between with [\"YYYY-MM-DD\",\"YYYY-MM-DD\"]; for 'in' value is a list); "
            '"sort_desc": true/false; "limit": integer or null; "chart": true only if the user asked for a chart; '
            '"cols2": two columns for corr.\n'
            "Use scalar for one number, group for a breakdown, rows to list records (who/list/which one), "
            "trend for change over time. Use only the column names above. "
            "Relative dates ('last year') are relative to the END of the data's date range."
        )
        text = self._call_llm(system, q)
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b < 0:
            raise RuntimeError("the AI reply was not understood")
        return self._validate(json.loads(text[a:b + 1]))

    def _col(self, name, roles=None):
        if not name:
            return None
        for c in self.df.columns:
            if c.lower() == str(name).lower() or nice(c).lower() == str(name).lower():
                if roles is None or self.roles[c] in roles:
                    return c
        return None

    def _validate(self, d):
        kinds = {"scalar", "group", "rows", "corr", "trend", "summary", "describe", "columns", "quality",
                 "distinct", "outliers"}
        aggs = {"sum", "mean", "median", "min", "max", "count", "nunique", "std", "share"}
        p = Plan(kind=d.get("kind") if d.get("kind") in kinds else "scalar",
                 agg=d.get("agg") if d.get("agg") in aggs else "sum")
        p.metric = self._col(d.get("metric"), ("number",))
        p.by = [c for c in (self._col(x) for x in (d.get("by") or [])[:2]) if c]
        p.grain = d.get("grain") if d.get("grain") in {"day", "week", "month", "quarter", "year", "weekday",
                                                       "monthname"} else None
        p.sort_desc = bool(d.get("sort_desc", True))
        try:
            p.limit = max(1, min(int(d["limit"]), 100)) if d.get("limit") else None
        except (TypeError, ValueError):
            p.limit = None
        p.chart = bool(d.get("chart"))
        p.cols2 = [c for c in (self._col(x) for x in (d.get("cols2") or [])[:2]) if c]
        for f in d.get("filters") or []:
            c = self._col(f.get("col"))
            op = f.get("op")
            if not c or op not in {"==", "in", "!=", ">", ">=", "<", "<=", "between"}:
                continue
            p.filters.append({"col": c, "op": op, "value": f.get("value")})
        if p.kind in ("scalar", "group", "trend", "describe", "outliers") and p.agg != "count" and not p.metric:
            p.metric = self.metric
        if p.kind == "group" and not p.by:
            p.kind = "scalar"
        if p.kind == "trend" and self.date_col:
            p.by, p.grain, p.metric = [self.date_col], p.grain or self._auto_grain(), p.metric or self.metric
        if p.agg in ("sum", "mean", "median", "std", "share", "min", "max") and not p.metric and p.kind != "rows":
            return None
        return p

    def _call_llm(self, system, user):
        if self.provider == "ollama":
            payload = {"model": self.ollama_model, "stream": False, "format": "json",
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                       "options": {"temperature": 0, "num_predict": 400}}
            req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(payload).encode(),
                                         headers={"content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.load(r).get("message", {}).get("content", "")
            except urllib.error.HTTPError as e:
                raise RuntimeError(f"model '{self.ollama_model}' not found; run: ollama pull {self.ollama_model}"
                                   if e.code == 404 else f"Ollama error {e.code}")
            except Exception:
                raise RuntimeError("Ollama is not running or too slow")
        body = json.dumps({"model": ANTHROPIC_MODEL, "max_tokens": 400, "system": system,
                           "messages": [{"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(ANTHROPIC_URL, data=body, headers={
            "content-type": "application/json", "x-api-key": self.api_key or "", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError("the API key was rejected" if e.code == 401 else f"API error {e.code}")
        except Exception:
            raise RuntimeError("no internet connection")
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    # ================================================================== EXECUTE
    def _apply(self, df, filters):
        desc = []
        for f in filters:
            c, op, v = f["col"], f["op"], f["value"]
            s = df[c]
            if self.roles.get(c) == "date":
                if op == "between":
                    lo, hi = pd.Timestamp(v[0]), pd.Timestamp(v[1]) + pd.Timedelta(days=1)
                    m = (s >= lo) & (s < hi)
                elif op == "month_is":
                    m = s.dt.month.isin(v)
                elif op == "ranges":
                    m = pd.Series(False, index=df.index)
                    for lo_, hi_ in v:
                        m |= (s >= pd.Timestamp(lo_)) & (s < pd.Timestamp(hi_) + pd.Timedelta(days=1))
                else:
                    m = pd.Series(True, index=df.index)
            elif op == "==":
                m = s.astype(str).str.lower() == str(v).lower()
            elif op == "!=":
                m = s.astype(str).str.lower() != str(v).lower()
            elif op == "in":
                m = s.astype(str).str.lower().isin([str(x).lower() for x in v])
            elif op == "between":
                m = (s >= v[0]) & (s <= v[1])
            else:
                x = float(v)
                m = {">": s > x, ">=": s >= x, "<": s < x, "<=": s <= x}[op]
            df = df[m]
            desc.append((f"{nice(c)}: {f['label']}" if self.roles.get(c) == "date" and f.get("label") else None)
                        or (f"{nice(c)}: {v[0]} to {v[1]}" if self.roles.get(c) == "date" and op == "between" else None)
                        or f.get("label") or (f"{nice(c)} = {v}" if op == "==" else
                                           f"{nice(c)} in {', '.join(map(str, v))}" if op == "in" else
                                           f"{nice(c)} {op} {v}"))
        return df, desc

    def _no_rows(self, plan, desc):
        msg = f"No records match: {'; '.join(desc)}."
        for f in plan.filters:
            if f["op"] in ("==", "in") and f["col"] in self.cat_cols + self.ent_cols:
                ex = ", ".join(map(str, self.df[f["col"]].dropna().unique()[:8]))
                msg += f"\nValues in '{f['col']}' include: {ex}"
                break
        return Answer(msg)

    def _label(self, plan):
        a, m = plan.agg, nice(plan.metric) if plan.metric else "records"
        return {"sum": f"total {m}", "mean": f"average {m}", "median": f"median {m}", "min": f"lowest {m}",
                "max": f"highest {m}", "std": f"standard deviation of {m}", "count": "number of records",
                "nunique": f"number of different {m}", "share": f"share of {m}"}[a]

    def _run(self, plan):
        df, desc = self._apply(self.df, plan.filters)
        if df.empty and plan.kind not in ("columns",):
            return self._no_rows(plan, desc)
        k = plan.kind
        ctx = f" ({'; '.join(desc)})" if desc else ""
        if k == "columns":
            return Answer(self.columns())
        if k == "summary":
            return self._summary()
        if k == "quality":
            return self._quality()
        if k == "describe":
            return self._describe(df, plan, ctx)
        if k == "distinct":
            return self._distinct(df, plan, ctx)
        if k == "outliers":
            return self._outliers(df, plan, ctx)
        if k == "corr":
            return self._corr(df, plan, ctx)
        if k == "trend":
            return self._trend(df, plan, ctx)
        if k == "rows":
            return self._rows(df, plan, ctx)
        if k == "group":
            return self._group(df, plan, ctx, desc)
        return self._scalar(df, plan, ctx, desc)

    # ---- scalar
    def _val(self, df, agg, metric):
        if agg == "count":
            return len(df)
        s = df[metric]
        if agg == "nunique":
            return s.nunique()
        return {"sum": s.sum, "mean": s.mean, "median": s.median, "min": s.min, "max": s.max, "std": s.std,
                "share": s.sum}[agg]()

    def _scalar(self, df, plan, ctx, desc):
        agg, metric = plan.agg, plan.metric
        pct = metric in self.pct_cols
        n = len(df)
        if plan.note == "profile":
            lines = [f"{n:,} records{ctx}."]
            if self.metric:
                lines.append(f"{self._label(Plan(agg='sum' if not NON_ADDITIVE.search(self.metric) else 'mean', metric=self.metric)).capitalize()}: "
                             f"{fmtn(self._val(df, 'sum' if not NON_ADDITIVE.search(self.metric) else 'mean', self.metric), metric in self.pct_cols)}")
            return Answer("\n".join(lines))
        if agg == "share":
            date_f = [f for f in plan.filters if self.roles.get(f["col"]) == "date" or f["op"] in (">", "<", ">=", "<=", "between")]
            base, _ = self._apply(self.df, date_f)
            num = df[metric].sum() if metric else len(df)
            den = base[metric].sum() if metric else len(base)
            p = num / den * 100 if den else float("nan")
            what = nice(metric) if metric else (plan.noun or "records")
            return Answer(f"{p:.1f}% of {what}{ctx}: {fmtn(num)} of {fmtn(den)}.")
        v = self._val(df, agg, metric)
        label = self._label(plan)
        if agg == "count":
            noun = plan.noun or "records"
            text = f"{n:,} {noun}" + (f" match: {'; '.join(desc)}" if desc else " in total")
            if desc and len(df) < len(self.df):
                text += f" ({n / len(self.df):.1%} of all {len(self.df):,})"
            return Answer(text + ".")
        text = f"{label.capitalize()}{ctx}: {fmtn(v, pct and agg != 'count')}"
        extra = [f"Based on {n:,} records."]
        if agg in ("max", "min") and metric:
            row = df.loc[df[metric].idxmax() if agg == "max" else df[metric].idxmin()]
            show = [c for c in self.ent_cols[:2] + self.cat_cols[:3] if c in df.columns][:4]
            show = show or [c for c in df.columns if c != metric][:4]
            if show:
                extra.insert(0, "Found at: " + ", ".join(f"{nice(c)} = {row[c]}" for c in show) + ".")
        if desc and agg in ("sum", "count") and len(df) < len(self.df):
            tot = len(self.df) if agg == "count" else self.df[metric].sum()
            if tot:
                extra.append(f"That is {v / tot:.1%} of the overall {'records' if agg == 'count' else 'total'} "
                             f"({fmtn(tot)}).")
        if desc and agg == "mean":
            extra.append(f"Overall average (all records): {fmtn(self.df[metric].mean(), pct)}.")
        return Answer(text + "\n" + " ".join(extra))

    # ---- group
    def _key(self, df, col, grain):
        s = df[col]
        if self.roles.get(col) == "date":
            d = s
            if grain == "year":
                return d.dt.year.astype("Int64").astype(str)
            if grain == "quarter":
                return d.dt.year.astype("Int64").astype(str) + "-Q" + d.dt.quarter.astype("Int64").astype(str)
            if grain == "week":
                return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
            if grain == "day":
                return d.dt.strftime("%Y-%m-%d")
            if grain == "weekday":
                return d.dt.day_name()
            if grain == "monthname":
                return d.dt.strftime("%b")
            return d.dt.strftime("%Y-%m")
        return s.astype(object).where(s.notna(), "(missing)").astype(str)

    def _group_table(self, df, plan):
        by, agg, metric = plan.by[:2], plan.agg, plan.metric
        keys = []
        for c in by:
            k = self._key(df, c, plan.grain if self.roles.get(c) == "date" else None)
            k.name = f"{nice(c)} ({plan.grain})" if self.roles.get(c) == "date" and plan.grain else nice(c)
            keys.append(k)
        mask = pd.concat([k.notna() for k in keys], axis=1).all(axis=1)
        df, keys = df[mask], [k[mask] for k in keys]
        g = df.groupby(keys, sort=False)
        if agg == "count" or (agg == "share" and not metric):
            s = g.size()
        elif agg == "nunique":
            s = g[metric].nunique()
        elif agg == "share":
            s = g[metric].sum()
        else:
            s = getattr(g[metric], agg)()
        name = self._valname(plan)
        res = s.rename(name).reset_index()
        timeish = any(self.roles.get(c) == "date" for c in by) or plan.grain in ("monthname", "weekday")
        key_cols = [k.name for k in keys]
        if timeish:
            order = None
            if plan.grain == "weekday":
                order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            elif plan.grain == "monthname":
                order = [pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)]
            if order:
                res["_o"] = res[key_cols[0]].map({v: i for i, v in enumerate(order)})
                res = res.sort_values("_o").drop(columns="_o")
            else:
                res = res.sort_values(key_cols)
        else:
            res = res.sort_values(name, ascending=not plan.sort_desc)
        return res.reset_index(drop=True), key_cols, name

    def _valname(self, plan):
        m = nice(plan.metric) if plan.metric else "records"
        return {"count": "records", "share": "share_%", "mean": f"average {m}", "median": f"median {m}",
                "std": f"std dev of {m}", "nunique": f"different {m}", "min": f"lowest {m}", "max": f"highest {m}",
                "sum": m}[plan.agg]

    def _group(self, df, plan, ctx, desc):
        res, keys, name = self._group_table(df, plan)
        if res.empty:
            return Answer("No data to group.")
        pct = plan.metric in self.pct_cols
        label = self._label(plan)
        timeish = any(self.roles.get(c) == "date" for c in plan.by) or plan.grain in ("monthname", "weekday")
        missing_row = None
        if not timeish and len(res) > 1:
            mm_ = (res[keys] == "(missing)").any(axis=1)
            if mm_.any() and not mm_.all():
                missing_row, res = res[mm_], res[~mm_].reset_index(drop=True)
        vals = res[name]
        total = vals.sum() + (missing_row[name].sum() if missing_row is not None else 0) if plan.agg in ("sum", "count") else None
        if plan.agg in ("sum", "count") and len(keys) == 1 and not timeish:
            res["share %"] = (vals / total * 100).round(1) if total else np.nan
            if missing_row is not None:
                missing_row = missing_row.assign(**{"share %": (missing_row[name] / total * 100).round(1)})
        if missing_row is not None:
            res = pd.concat([res, missing_row], ignore_index=True)
            vals = res[name].iloc[: len(res) - len(missing_row)]
        k0 = res[keys[0]].astype(str)
        label_keys = " x ".join(keys)

        def row_label(i):
            return " / ".join(str(res.loc[i, c]) for c in keys)

        if timeish:
            return self._time_answer(res, keys, name, plan, label, ctx, pct)
        hi, lo = vals.idxmax(), vals.idxmin()
        miss_note = f" {int(missing_row['records'].sum() if 'records' in missing_row else 0):,} records have no {nice(keys[0])}." if (
            missing_row is not None and plan.agg == "count") else (" Some records have no value for " + nice(keys[0]) + " and are listed last." if missing_row is not None else "")
        if plan.ranked:
            n = plan.limit or 1
            order = res.head(max(n, 5))
            lead = row_label(0)
            word = "highest" if plan.sort_desc else "lowest"
            txt = f"{lead} has the {word} {label}{ctx}: {fmtn(res.loc[0, name], pct)}"
            if total and len(res) > 1:
                txt += f" ({res.loc[0, name] / total:.0%} of the total {fmtn(total)})"
            txt += "."
            if n > 1:
                names = ", ".join(f"{row_label(i)} ({fmtn(res.loc[i, name], pct)})" for i in range(min(n, len(res))))
                txt = f"{'Top' if plan.sort_desc else 'Bottom'} {min(n, len(res))} by {label}{ctx}: {names}."
            elif len(res) > 1:
                gap = res.loc[1, name]
                if gap and res.loc[0, name] and plan.sort_desc and res.loc[0, name] > 0 and gap > 0:
                    d_ = res.loc[0, name] / gap - 1
                    txt += f" Next is {row_label(1)} ({fmtn(gap, pct)}), {d_:.1%} behind." if d_ < 0.1 else \
                        f" Next is {row_label(1)} ({fmtn(gap, pct)}), {d_:.0%} behind."
            return Answer(txt + miss_note, pd.concat([order, missing_row]) if missing_row is not None else order)
        if len(res) == 2 and plan.agg in ("sum", "mean", "median", "count") and res[name].min() > 0:
            a, b = res.loc[0], res.loc[1]
            diff = a[name] / b[name] - 1
            dtxt = f"{diff:.1%}" if diff < 0.1 else f"{diff:.0%}"
            txt = (f"{label.capitalize()}{ctx}: {row_label(0)} = {fmtn(a[name], pct)}, {row_label(1)} = {fmtn(b[name], pct)}. "
                   f"{row_label(0)} is {dtxt} higher.")
            return Answer(txt, res)
        shown = res.head(15)
        txt = (f"{label.capitalize()} by {label_keys}{ctx}: highest {row_label(hi)} ({fmtn(vals[hi], pct)}), "
               f"lowest {row_label(lo)} ({fmtn(vals[lo], pct)}); {len(res)} groups")
        if total:
            txt += f", total {fmtn(total)}"
        elif plan.agg in ("mean", "median"):
            txt += f", overall {plan.agg} {fmtn(self._val(df, plan.agg, plan.metric), pct)}"
        txt += "."
        if len(res) > 15:
            txt += f" (showing the first 15 of {len(res)})"
        return Answer(txt + miss_note, shown)

    def _complete_periods(self, res, keys, plan):
        """Drop an incomplete first/last period so trends are not distorted."""
        if plan.grain not in ("month", "quarter", "year", "week") or len(res) < 4:
            return res, ""
        lo, hi = self.df[self.date_col].min(), self.df[self.date_col].max()
        note = ""
        k = res[keys[0]].astype(str)
        last, first = k.iloc[-1], k.iloc[0]
        try:
            if plan.grain == "month":
                ends = pd.Period(last, "M").end_time
                starts = pd.Period(first, "M").start_time
            elif plan.grain == "year":
                ends, starts = pd.Timestamp(int(last), 12, 31), pd.Timestamp(int(first), 1, 1)
            elif plan.grain == "quarter":
                ends = pd.Period(last.replace("-", ""), "Q").end_time
                starts = pd.Period(first.replace("-", ""), "Q").start_time
            else:
                return res, ""
        except Exception:
            return res, ""
        out = res
        if hi.normalize() < ends.normalize() - pd.Timedelta(days=2):
            out, note = out.iloc[:-1], f"the last period ({last}) is incomplete (data ends {hi:%Y-%m-%d}) so it is left out of the comparison"
        if lo.normalize() > starts.normalize() + pd.Timedelta(days=2):
            out = out.iloc[1:]
            note = (note + "; " if note else "") + f"the first period ({first}) is incomplete and left out"
        return out, note

    def _time_answer(self, res, keys, name, plan, label, ctx, pct):
        core, note = self._complete_periods(res, keys, plan)
        if len(core) < 2:
            core = res
        v = core[name]
        kk = core[keys[0]].astype(str)
        hi, lo = v.idxmax(), v.idxmin()
        txt = f"{label.capitalize()} by {plan.grain}{ctx}: peak {kk[hi]} ({fmtn(v[hi], pct)}), low {kk[lo]} ({fmtn(v[lo], pct)})"
        if plan.grain not in ("weekday", "monthname") and len(core) >= 3 and v.iloc[0]:
            ch = v.iloc[-1] / v.iloc[0] - 1 if v.iloc[0] else float("nan")
            txt += f"; {kk.iloc[0]} to {kk.iloc[-1]}: {fmtn(v.iloc[0], pct)} -> {fmtn(v.iloc[-1], pct)} ({ch:+.0%})"
        txt += "."
        if note:
            txt += f" Note: {note}."
        table = res if len(res) <= 20 else res.tail(12)
        if plan.ranked and plan.limit:
            order = core.sort_values(name, ascending=not plan.sort_desc).head(max(plan.limit, 3))
            word = "highest" if plan.sort_desc else "lowest"
            first = order.iloc[0]
            txt = (f"{first[keys[0]]} has the {word} {label}{ctx}: {fmtn(first[name], pct)}. "
                   f"Next: " + ", ".join(f"{r[keys[0]]} ({fmtn(r[name], pct)})" for _, r in order.iloc[1:3].iterrows()) + ".")
            table = order
        return Answer(txt, table)

    # ---- trend
    def _trend(self, df, plan, ctx):
        plan.kind = "group"
        plan.ranked = False
        res, keys, name = self._group_table(df, plan)
        if len(res) < 2:
            return Answer("There is not enough history to show a trend.")
        core, note = self._complete_periods(res, keys, plan)
        core = core if len(core) >= 3 else res
        v, kk = core[name].astype(float), core[keys[0]].astype(str)
        x = np.arange(len(v))
        slope = np.polyfit(x, v, 1)[0] if len(v) >= 3 else v.iloc[-1] - v.iloc[0]
        rel = slope / v.mean() if v.mean() else 0
        ch = v.iloc[-1] / v.iloc[0] - 1 if v.iloc[0] else float("nan")
        direction = "flat" if (abs(ch) < 0.05 and abs(rel) < 0.01) else ("rising" if slope > 0 else "falling")
        label = self._label(plan)
        pct = plan.metric in self.pct_cols
        txt = (f"{label.capitalize()}{ctx} is {direction}: {kk.iloc[0]} {fmtn(v.iloc[0], pct)} -> {kk.iloc[-1]} "
               f"{fmtn(v.iloc[-1], pct)} ({ch:+.0%}). Average change per {plan.grain}: {rel:+.1%}. "
               f"Peak {kk[v.idxmax()]} ({fmtn(v.max(), pct)}), low {kk[v.idxmin()]} ({fmtn(v.min(), pct)}).")
        if note:
            txt += f" Note: {note}."
        table = res if len(res) <= 20 else res.tail(12)
        return Answer(txt, table)

    # ---- rows (who / list)
    def _rows(self, df, plan, ctx):
        metric = plan.metric
        if metric:
            df = df.sort_values(metric, ascending=not plan.sort_desc) if plan.ranked else df
        note_cols = [c for c in plan.note.split("|") if c in df.columns]
        name_col = (next((c for c in self.ent_cols if re.search(r"(^|[_\s])name$|^name", c, re.I)), None)
                    or next((c for c in self.ent_cols if re.search(r"customer|employee|student|person|client", c, re.I)), None)
                    or (self.ent_cols[0] if self.ent_cols else None))
        cols = []
        for c in ([name_col] + self.ent_cols[:1] + note_cols + ([metric] if metric else []) +
                  [f["col"] for f in plan.filters if f["col"] in df.columns]):
            if c and c in df.columns and c not in cols:
                cols.append(c)
        for c in list(df.columns):
            if len(cols) >= 7:
                break
            if c not in cols:
                cols.append(c)
        n = plan.limit or 10
        out = df[cols].head(n)
        if plan.ranked and n == 1 and len(out):
            r = out.iloc[0]
            pct = metric in self.pct_cols
            head = ", ".join(f"{nice(c)}: {fmtn(r[c], c in self.pct_cols) if self.roles[c] == 'number' else r[c]}" for c in cols[:6])
            word = "highest" if plan.sort_desc else "lowest"
            return Answer(f"{word.capitalize()} {nice(metric)}{ctx}: {head}", out, show_table=False)
        total = len(df)
        if plan.ranked and metric and len(out):
            word = "highest" if plan.sort_desc else "lowest"
            who_ = ", ".join(f"{r[name_col] if name_col else i + 1} ({fmtn(r[metric], metric in self.pct_cols)})"
                             for i, (_, r) in enumerate(out.iterrows()))
            return Answer(f"Top {len(out)} by {nice(metric)} ({word} first){ctx}: {who_}.", out)
        txt = f"{total:,} matching records{ctx}" + (f"; showing the first {len(out)}." if total > n else ".")
        return Answer(txt, out)

    # ---- others
    def _describe(self, df, plan, ctx):
        s = df[plan.metric].dropna()
        st_ = {"count": len(s), "mean": s.mean(), "median": s.median(), "min": s.min(), "max": s.max(),
               "std": s.std()}
        pct = plan.metric in self.pct_cols
        counts, edges = np.histogram(s, bins=min(8, max(3, int(np.sqrt(len(s))))))
        tbl = pd.DataFrame({"range": [f"{fmtn(edges[i])} - {fmtn(edges[i + 1])}" for i in range(len(counts))],
                            "records": counts, "share %": (counts / counts.sum() * 100).round(1)})
        q1, q3 = s.quantile(.25), s.quantile(.75)
        return Answer(f"{nice(plan.metric).capitalize()}{ctx}: average {fmtn(st_['mean'], pct)}, median "
                      f"{fmtn(st_['median'], pct)}, typical middle half {fmtn(q1, pct)} to {fmtn(q3, pct)}, "
                      f"full range {fmtn(st_['min'], pct)} to {fmtn(st_['max'], pct)} ({len(s):,} records).", tbl)

    def _distinct(self, df, plan, ctx):
        c = plan.metric
        vc = df[c].dropna().astype(str).value_counts()
        top = ", ".join(f"{k} ({v})" for k, v in vc.head(8).items())
        return Answer(f"{len(vc):,} different {nice(c)} values{ctx}. Most common: {top}.",
                      vc.head(15).rename_axis(nice(c)).reset_index(name="records"))

    def _outliers(self, df, plan, ctx):
        s = df[plan.metric].dropna()
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = df[(df[plan.metric] < lo) | (df[plan.metric] > hi)]
        if out.empty:
            return Answer(f"No outliers in {nice(plan.metric)}{ctx} (normal range {fmtn(lo)} to {fmtn(hi)}).")
        cols = [c for c in (self.ent_cols[:1] + [plan.metric]) if c in df.columns]
        cols += [c for c in df.columns if c not in cols][:3]
        out = out.sort_values(plan.metric, key=lambda x: (x - s.median()).abs(), ascending=False)
        return Answer(f"{len(out):,} outliers in {nice(plan.metric)}{ctx} ({len(out) / len(s):.1%} of records). "
                      f"Normal range is {fmtn(lo)} to {fmtn(hi)}.", out[cols].head(10))

    def _corr(self, df, plan, ctx):
        a, b = plan.cols2[:2]
        ra, rb = self.roles[a], self.roles[b]
        if ra == "number" and rb == "number":
            d = df[[a, b]].dropna()
            if len(d) < 5:
                return Answer("Not enough rows to measure a relationship.")
            r = d[a].corr(d[b])
            strength = "very weak" if abs(r) < .2 else "weak" if abs(r) < .4 else "moderate" if abs(r) < .6 else \
                "strong" if abs(r) < .8 else "very strong"
            direction = "positive" if r > 0 else "negative"
            meaning = (f"higher {nice(a)} tends to go with {'higher' if r > 0 else 'lower'} {nice(b)}"
                       if abs(r) >= .2 else f"{nice(a)} and {nice(b)} are barely related")
            slope = np.polyfit(d[a], d[b], 1)[0]
            txt = (f"{nice(a).capitalize()} vs {nice(b)}{ctx}: correlation {r:+.2f} ({strength} {direction}); {meaning}. "
                   f"On average each +1 in {nice(a)} goes with {slope:+,.3g} in {nice(b)} (based on {len(d):,} records). "
                   "Correlation does not prove cause.")
            return Answer(txt, d.rename(columns=nice).head(0))
        cat, num = (a, b) if ra == "category" else (b, a)
        if self.roles[num] != "number":
            return Answer("I can relate two numeric columns, or a category with a numeric column.")
        g = df.groupby(df[cat].astype(str))[num].agg(["count", "mean"]).reset_index().sort_values("mean", ascending=False)
        g.columns = [nice(cat), "records", f"average {nice(num)}"]
        top, bot = g.iloc[0], g.iloc[-1]
        gap = top.iloc[2] / bot.iloc[2] - 1 if bot.iloc[2] else float("nan")
        ss_b = sum(len(x) * (x.mean() - df[num].mean()) ** 2 for _, x in df.groupby(df[cat].astype(str))[num])
        ss_t = ((df[num] - df[num].mean()) ** 2).sum()
        eta = ss_b / ss_t if ss_t else 0
        verdict = ("so it makes almost no difference" if eta < 0.01 else "a small effect" if eta < 0.06 else
                   "a medium effect" if eta < 0.14 else "a large effect")
        txt = (f"Average {nice(num)} by {nice(cat)}{ctx}: highest {top.iloc[0]} ({fmtn(top.iloc[2])}), lowest "
               f"{bot.iloc[0]} ({fmtn(bot.iloc[2])}), a gap of {(f"{gap:.1%}" if abs(gap) < 0.1 else f"{gap:.0%}")}. {nice(cat).capitalize()} explains about "
               f"{eta:.0%} of the variation in {nice(num)}, {verdict}.")
        return Answer(txt, g.head(15))

    def _summary(self):
        df = self.df
        lines = [f"{self.name}: {len(df):,} rows, {len(df.columns)} columns."]
        if self.date_col:
            lines.append(f"Dates ({self.date_col}): {df[self.date_col].min():%Y-%m-%d} to {df[self.date_col].max():%Y-%m-%d}.")
        for c in self.num_cols[:4]:
            lines.append(f"{nice(c)}: average {fmtn(df[c].mean(), c in self.pct_cols)}, min {fmtn(df[c].min())}, max {fmtn(df[c].max())}.")
        for c in self.cat_cols[:4]:
            vc = df[c].astype(str).value_counts()
            lines.append(f"{nice(c)}: {len(vc)} values, most common {vc.index[0]} ({vc.iloc[0] / len(df):.0%}).")
        miss = df.isna().sum()
        miss = miss[miss > 0]
        lines.append("Missing values: " + (", ".join(f"{nice(c)} ({n})" for c, n in miss.items()) if len(miss) else "none") + ".")
        return Answer("\n".join(lines))

    def _quality(self):
        df = self.df
        miss = df.isna().sum()
        miss = miss[miss > 0]
        lines = []
        lines.append("Missing values: " + (", ".join(f"{nice(c)} {n} ({n / len(df):.1%})" for c, n in miss.items()) if len(miss) else "none"))
        lines.append(f"Duplicate rows: {int(df.duplicated().sum()):,}")
        return Answer("\n".join(lines))

    # ================================================================== charts (only on request)
    def _chart(self, ans, plan):
        t = ans.table
        if plt is None or t is None or t.empty or plan.kind not in ("group", "trend", "rows", "corr"):
            return None
        CHART_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        try:
            if plan.kind == "corr" and len(plan.cols2) == 2 and all(self.roles[c] == "number" for c in plan.cols2):
                d, _ = self._apply(self.df, plan.filters)
                ax.scatter(d[plan.cols2[0]], d[plan.cols2[1]], s=12, alpha=.5)
                ax.set_xlabel(nice(plan.cols2[0]))
                ax.set_ylabel(nice(plan.cols2[1]))
                title = f"{nice(plan.cols2[1])} vs {nice(plan.cols2[0])}"
            else:
                if plan.kind == "rows":
                    return None
                key, val = t.columns[0], t.columns[1]
                timeish = plan.kind == "trend" or any(self.roles.get(c) == "date" for c in plan.by)
                x = t[key].astype(str)
                if timeish:
                    ax.plot(x, t[val], marker="o")
                    step = max(1, len(x) // 12)
                    ax.set_xticks(range(0, len(x), step))
                    ax.set_xticklabels(x.iloc[::step], rotation=45, ha="right")
                else:
                    tt = t.head(15).iloc[::-1]
                    ax.barh(tt[key].astype(str), tt[val])
                title = f"{val} by {key}"
            ax.set_title(title)
            ax.grid(alpha=.3)
            fig.tight_layout()
            path = CHART_DIR / (re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60] + ".png")
            fig.savefig(path, dpi=130)
            return str(path)
        finally:
            plt.close(fig)


GROUP_WORDS = {"total", "average", "mean", "median", "top", "bottom", "best", "worst", "highest", "lowest", "most",
               "least", "number", "count", "all", "sum"}


def agg_free(ql):
    """True when 'most/least' is NOT followed by a counted noun (so it is about a value, not a count)."""
    m = re.search(r"\b(?:most|least|fewest)\s+([a-z]+)", ql)
    if not m:
        return True
    return m.group(1) in STOP or m.group(1) in GRAINS or m.group(1) in ALIAS or m.group(1) in {"money", "profitable", "revenue"}


def date_filters_years(filters):
    out = []
    for f in filters:
        out += f.get("years", [])
    return out


# =========================================================================== command line
def main():
    ap = argparse.ArgumentParser(description="DataBot - ask questions about any CSV/Excel file")
    ap.add_argument("file")
    ap.add_argument("--sheet")
    ap.add_argument("--ask", help="ask one question and exit")
    ap.add_argument("--no-ai", action="store_true")
    args = ap.parse_args()
    bot = DataBot(args.file, sheet=args.sheet, ai="off" if args.no_ai else "auto")
    if args.ask:
        print(bot.ask(args.ask))
        return
    print("=" * 62)
    print(f" DataBot  -  {bot.name}   (AI backup: {bot.provider or 'off'})")
    print("=" * 62)
    print(bot.help_text())
    print("\nType 'exit' to quit ('ollama' = free local AI backup, 'setkey' = Claude key).\n")
    while True:
        try:
            q = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", "q", "bye"}:
            break
        if q.lower() == "ollama":
            bot.use_ollama()
            continue
        if q.lower() == "setkey":
            bot.set_key()
            continue
        print("\nBot >", bot.ask(q), "\n")


if __name__ == "__main__":
    main()
