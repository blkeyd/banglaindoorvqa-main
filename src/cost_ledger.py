"""
cost_ledger.py
BanglaIndoorVQA: THE one API cost ledger.

Every script that makes a billed API call writes one line per call to the same
ledger file in Google Drive and checks the total before each call. This is how
the 40 USD hard stop of Design 4.9 and 12.1 holds across scripts and chats.

Import this file. Do not write a second cost meter anywhere.

Ledger file (Drive only, never in Git):
    MyDrive/BanglaIndoorVQA/outputs/api_cost_ledger.jsonl
One JSON object per line. Required keys:
    ts_utc, script, call_id, model_id, mode, input_tokens, output_tokens,
    thoughts_tokens, usd
Any other key is allowed (run_id, note, price_in_per_m, price_out_per_m, ...).

Usage:
    import cost_ledger
    cost_ledger.assert_room(ledger_path, next_call_max_usd=0.02)   # before a call
    usd = cost_ledger.cost_usd(in_tok, out_tok_billed, price_in, price_out)
    cost_ledger.append(ledger_path, {... "usd": usd})              # after a call

Known limit: two Colab sessions that append at the same moment can interleave
lines. Run one billed script at a time.
"""

import json
import os
from datetime import datetime, timezone

HARD_CAP_USD = 40.0          # Design 12.1: the hard cap on total API spend

REQUIRED_KEYS = (
    "ts_utc", "script", "call_id", "model_id", "mode",
    "input_tokens", "output_tokens", "thoughts_tokens", "usd",
)


class BudgetExceeded(RuntimeError):
    """Raised when the next call could take the spend over a cap."""


class LedgerError(RuntimeError):
    """Raised when the ledger file cannot be trusted (unreadable line)."""


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_lines(path):
    """Return every ledger record as a list of dicts. Missing file: []."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                float(rec["usd"])
            except Exception as exc:  # a broken line must never be skipped
                raise LedgerError(
                    f"Ledger line {n} in {path} is unreadable ({exc}). "
                    "Fix or remove that line by hand, then run again. "
                    "The script stops so that spend is never undercounted."
                )
            records.append(rec)
    return records


def read_total(path, script=None):
    """Total USD in the ledger; with script=..., only that script's lines."""
    total = 0.0
    for rec in read_lines(path):
        if script is None or rec.get("script") == script:
            total += float(rec["usd"])
    return total


def cost_usd(input_tokens, output_tokens_billed, price_in_per_m, price_out_per_m):
    """Cost of one call. output_tokens_billed includes thinking tokens."""
    return (
        float(input_tokens) * float(price_in_per_m)
        + float(output_tokens_billed) * float(price_out_per_m)
    ) / 1_000_000.0


def assert_room(path, next_call_max_usd=0.0, cap_usd=HARD_CAP_USD):
    """Stop before a call that could take total spend over the cap."""
    total = read_total(path)
    if total + float(next_call_max_usd) > float(cap_usd):
        raise BudgetExceeded(
            f"HARD STOP: total API spend is {total:.4f} USD and the next call "
            f"could add up to {float(next_call_max_usd):.4f} USD. "
            f"The cap is {float(cap_usd):.2f} USD (Design 12.1). "
            "Do not raise the cap without a decision row."
        )
    return total


def append(path, record):
    """Append one record. Fills ts_utc if missing. Flushes to disk."""
    rec = dict(record)
    rec.setdefault("ts_utc", utc_now())
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    if missing:
        raise LedgerError(f"Ledger record is missing keys: {missing}")
    rec["usd"] = round(float(rec["usd"]), 8)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return rec


if __name__ == "__main__":
    # Self check on a temporary file. Every line must print OK.
    import tempfile

    tmp = os.path.join(tempfile.mkdtemp(), "ledger.jsonl")
    checks = []
    checks.append(("empty ledger total is 0", read_total(tmp) == 0.0))
    c = cost_usd(1_000_000, 1_000_000, 0.75, 3.75)
    checks.append(("cost of 1M in and 1M out", abs(c - 4.5) < 1e-9))
    base = {"script": "selfcheck", "call_id": "c1", "model_id": "m", "mode": "standard",
            "input_tokens": 10, "output_tokens": 5, "thoughts_tokens": 0}
    append(tmp, dict(base, usd=1.25))
    append(tmp, dict(base, usd=2.5, script="other", call_id="c2"))
    checks.append(("total after two lines", abs(read_total(tmp) - 3.75) < 1e-9))
    checks.append(("total for one script", abs(read_total(tmp, "other") - 2.5) < 1e-9))
    try:
        assert_room(tmp, next_call_max_usd=36.26)
        ok = False
    except BudgetExceeded:
        ok = True
    checks.append(("cap stops a call that would pass 40 USD", ok))
    checks.append(("cap allows a call that fits", assert_room(tmp, 36.25) == 3.75))
    try:
        append(tmp, {"usd": 1.0})
        ok = False
    except LedgerError:
        ok = True
    checks.append(("record with missing keys is refused", ok))
    with open(tmp, "a", encoding="utf-8") as f:
        f.write("not json\n")
    try:
        read_total(tmp)
        ok = False
    except LedgerError:
        ok = True
    checks.append(("broken line stops the script", ok))
    bad = 0
    for name, passed in checks:
        print(("OK   " if passed else "FAIL ") + name)
        bad += 0 if passed else 1
    print()
    print(f"{len(checks) - bad}/{len(checks)} self checks passed")
