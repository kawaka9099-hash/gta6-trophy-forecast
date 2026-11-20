"""Forecast GTA VI (PS5) Exophase trophy earn rates one week after launch.

Entry for make.fun's gta6-trophy-rarity-forecast-01. Python 3.13, standard
library only, no network, deterministic: the same case always yields the same
bytes.

How it works
------------
For each trophy the program looks at two things and blends them in logit
space (logit(p) = ln(p / (100 - p))):

1. The case's own history.  Every valid Exophase read before the checkpoint
   gives (time, players tracked, rate).  Rates are deduplicated per trophy
   (consecutive identical reads carry no signal -- Exophase refreshes on its
   own schedule), then a weighted least-squares line is fitted to
   logit(rate) against ln(1 + hours since release) over the last 24 hours of
   reads, and read at the target.  The slope is capped so a trophy that went
   0.00% -> 0.50% overnight cannot extrapolate to 90% by the target.

2. A launch-week prior, calibrated on the closest real analogues:
   GTA V's PSN-wide trophy rates (PSNProfiles: first story trophy ~43%
   lifetime, platinum ~0.1%), Red Dead Redemption 2's story pace (50-60 h
   main story, platinum a 240-300 h project), and GTA VI's own shape
   (Push Square's trophy guide puts the platinum at ~300 h; Newzoo /
   Sensor Tower put launch-week sales at roughly 37-51M copies, ~28.7M
   digital by Nov 20, so the tracked population is still growing fast and
   diluting the slow trophies).  The prior says, for each trophy class, what
   fraction of the day-7 tracked population should have reached it: early
   story chapters high, the finale single digits, 100% completion and the
   platinum near zero (nobody finishes a 300-hour platinum in seven days),
   grind/collectible trophies lagging far behind one-off activity trophies.

The blend weight follows the history: a trophy whose rate moved a lot gets
its fit trusted; a trophy whose history is flat (like the practice cases,
where every read is the same pre-refresh page) falls back to its last rate
with a wide spread, which is the honest answer when no movement has been
seen.  The spread also grows with the horizon and with fit residuals, and
is wider for rare/grindy trophies.

Structural constraints are enforced last: the story chain (Chapter 1..4,
final mission) is non-increasing, the story missions sit between Chapter 1
and the finale, 100% completion sits below the finale, and the platinum
sits below 100% completion (it requires every other trophy).  The opening
mission ("Gone Fishing") is essentially automatic, so it is forecast near
100% with a narrow spread.

Output: for every <id>.in in the input directory, one <id>.out with
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}, five
quantiles in percent, non-decreasing, each in [0.01, 99.99], rounded to
4 decimals half-up.

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would still import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release moment: GTA VI launched 2026-11-19.  The case's game.release_date
# is used when present; the fallback is the same date.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


# ---------------------------------------------------------------------------
# Launch-week priors: (day-7 median %, long-run ceiling %, extra spread)
# keyed by trophy id.  Calibrated on GTA V PSN-wide rates and RDR2's story
# pace (see the module docstring); the program's fit from the case's own
# history moves these up or down per trophy.
# ---------------------------------------------------------------------------
PRIORS = {
    1: (0.10, 2.0, 0.25),    # platinum: a ~300 h project, ~nobody in week 1
    2: (98.5, 99.99, 0.0),   # Gone Fishing: the opening mission, automatic
    3: (50.0, 65.0, 0.05),   # Chapter 1
    4: (35.0, 55.0, 0.05),   # Chapter 2
    5: (25.0, 45.0, 0.05),   # Chapter 3
    6: (16.0, 38.0, 0.05),   # Chapter 4
    7: (9.0, 30.0, 0.10),    # final mission
    8: (22.0, 35.0, 0.05),   # Breakfast Run (story mission)
    9: (19.0, 32.0, 0.05),   # The Cartel Heist
    10: (17.0, 30.0, 0.05),  # Child's Play
    11: (14.0, 28.0, 0.05),  # Headliners
    12: (12.0, 26.0, 0.05),  # VCIA Takedown
    13: (10.0, 24.0, 0.05),  # Operation Savage Glory
    14: (0.15, 8.0, 0.25),   # 100% completion
    15: (0.4, 8.0, 0.20),    # 250 stunts (grind)
    16: (8.0, 30.0, 0.05),   # try each activity once
    17: (0.8, 10.0, 0.15),   # gold in all Shooting Range challenges
    18: (1.5, 12.0, 0.15),   # drone race at every location
    19: (2.0, 15.0, 0.15),   # win 20 different races
    20: (6.0, 25.0, 0.05),   # beat Cal in a fishing challenge
    21: (12.0, 35.0, 0.05),  # complete an arcade game
    22: (4.0, 18.0, 0.10),   # final mission for Andy
    23: (3.5, 16.0, 0.10),   # final mission for Kenton
    24: (3.5, 16.0, 0.10),   # final mission for Billy
    25: (1.5, 10.0, 0.15),   # 8 stranger mission strands (silver)
    26: (3.0, 15.0, 0.10),   # final mission for Randy
    27: (3.5, 16.0, 0.10),   # final mission for Tyrone
    28: (1.0, 8.0, 0.15),    # meditate at every spot
    29: (4.0, 18.0, 0.10),   # hunt a unique animal
    30: (0.8, 8.0, 0.20),    # 3 collectible card sets
    31: (0.5, 6.0, 0.20),    # WAiNK It at 100 locations
    32: (1.5, 10.0, 0.15),   # complete 50 challenges
    33: (8.0, 28.0, 0.05),   # lose a 4-star wanted level with partner
    34: (6.0, 22.0, 0.05),   # rob 3 shops back-to-back
    35: (4.0, 18.0, 0.10),   # discover 100 vehicles
    36: (5.0, 20.0, 0.05),   # 5 enemies in one Combat Focus
    37: (6.0, 22.0, 0.05),   # unscheduled departure at VCIA
    38: (5.0, 20.0, 0.05),   # Rhinos
    39: (5.0, 20.0, 0.05),   # rob a jewelry store undetected
    40: (1.5, 10.0, 0.15),   # rob 5 bank vaults (silver)
    41: (6.0, 22.0, 0.05),   # rob an armored truck
    42: (2.5, 15.0, 0.10),   # explore all of Leonida
    43: (2.0, 12.0, 0.10),   # rob an illegal op of each gang
    44: (7.0, 25.0, 0.05),   # cocktail on Ocean Beach
    45: (5.0, 20.0, 0.05),   # smash out of a vehicle showroom
    46: (8.0, 28.0, 0.05),   # bowling
    47: (5.0, 20.0, 0.05),   # partnership with mass appeal
    48: (6.0, 22.0, 0.05),   # choices made
    49: (4.0, 18.0, 0.10),   # second time's the charm
    50: (3.0, 15.0, 0.10),   # what the hell just happened?
    51: (2.0, 12.0, 0.10),   # melee drive-by bowling
    52: (1.5, 10.0, 0.15),   # $250,000 laundering marked bills
    53: (5.0, 20.0, 0.05),   # spend $100,000 across shops
    54: (1.0, 8.0, 0.15),    # purchase all garage properties
    55: (4.0, 18.0, 0.10),   # register 10 vehicles
}

# Fallback prior (median %, ceiling %, extra spread) by tier, for any
# trophy the table above does not know.
TIER_PRIORS = {
    "platinum": (0.10, 2.0, 0.25),
    "gold": (3.0, 15.0, 0.15),
    "silver": (1.5, 10.0, 0.15),
    "bronze": (5.0, 25.0, 0.10),
}


def prior_for(trophy):
    if trophy["id"] in PRIORS:
        return PRIORS[trophy["id"]]
    return TIER_PRIORS.get(trophy.get("tier", "bronze"), (5.0, 25.0, 0.10))


def distinct_series(history, idx, release):
    """Per-trophy (hours, x, logit rate) points, consecutive duplicates removed.

    x = ln(1 + hours since release); hours is the raw age in hours.  A read
    whose rate equals the previous kept read's adds no information:
    Exophase had not refreshed.
    """
    pts = []
    prev = None
    for read in history:
        rate = read["rates"][idx]
        if rate is None:
            continue
        if prev is not None and rate == prev:
            continue
        prev = rate
        hours = (parse_time(read["at"]) - release).total_seconds() / 3600.0
        pts.append((max(0.0, hours), math.log1p(max(0.0, hours)), logit(rate)))
    return pts


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.  Returns (a, b, residual rms)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(
        sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    )
    return a, b, resid


def forecast_trophy(trophy, idx, case, release, x_target, horizon_days):
    last_rate = None
    for read in reversed(case["history"]):
        rate = read["rates"][idx]
        if rate is not None:
            last_rate = rate
            break
    if last_rate is None:
        last_rate = 0.0
    last_logit = logit(last_rate)

    prior_pct, ceiling_pct, extra = prior_for(trophy)
    # The prior is a day-7 level: never let it sit below what the trophy
    # has already reached (the fit may still pull the blend down if the
    # history shows dilution).
    prior_pct = max(prior_pct, 0.8 * last_rate)
    prior_logit = logit(prior_pct)
    ceiling_logit = logit(ceiling_pct)

    # A trophy the page shows at 100% can only fall (new tracked players
    # who have not finished the opening mission dilute it).  Its logit is
    # so large that blending against the last rate would pin it at 99.99%;
    # trust the prior instead: the opening mission is near-automatic, but
    # a week of new players pulls it a couple of points down.
    if last_rate >= 99.9:
        median_logit = prior_logit
        spread = 0.30 + 0.05 * math.sqrt(horizon_days) + extra
        spread = max(0.25, min(1.0, spread))
        qs = [round4(clamp_pct(inv_logit(median_logit + z * spread))) for z in Z]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        return {"id": trophy["id"], "q": qs, "median_logit": median_logit, "spread": spread}

    pts = distinct_series(case["history"], idx, release)
    # Fit window: the last 24 hours of reads (the house curve baseline's
    # window), so launch-night reads from a 638-player population do not
    # drag the slope; fall back to all points when there are too few.
    if pts:
        checkpoint_hours = (parse_time(case["at"]) - release).total_seconds() / 3600.0
        recent = [p for p in pts if p[0] >= checkpoint_hours - 24.0]
        if len(recent) >= 3:
            pts = recent
    fit_logit = None
    resid = 0.0
    movement = 0.0
    if len(pts) >= 2:
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        movement = max(ys) - min(ys)
        n = len(pts)
        ws = [0.5 + 0.5 * (j / max(1, n - 1)) for j in range(n)]
        a, b, resid = wls_line(xs, ys, ws)
        # Cap the slope: a trophy cannot plausibly move more than ~1.2
        # logit units per unit of ln(1 + hours).
        b = max(-1.2, min(1.2, b))
        fit_logit = a + b * x_target
        fit_logit = max(logit(0.02), min(ceiling_logit, fit_logit))

    if fit_logit is None or movement < 0.05:
        # No usable trend: the honest forecast is the last rate, with a
        # spread wide enough to cover an unseen refresh (the practice
        # cases) or a week of drift (flat trophies in the final).
        median_logit = last_logit
        # Still let the prior tug the median a little on long horizons:
        # a flat week-1 trophy is more likely to be near its class level
        # than frozen at a launch-night artefact (e.g. 4 of 638 players).
        if horizon_days >= 1.0:
            median_logit = 0.7 * last_logit + 0.3 * prior_logit
            median_logit = max(logit(0.02), min(ceiling_logit, median_logit))
        spread = 0.45 + 0.12 * math.sqrt(horizon_days) + extra
    else:
        # Blend fit and prior; the more the trophy moved, the more the
        # fit is trusted.  Cap how far the blend may sit from the prior:
        # the fit is a 2.5-day trend, the prior is the launch-week shape.
        signal = min(1.0, movement / 1.0) * min(1.0, len(pts) / 8.0)
        w_fit = 0.25 + 0.55 * signal
        median_logit = w_fit * fit_logit + (1.0 - w_fit) * prior_logit
        dev = median_logit - prior_logit
        if abs(dev) > 2.0:
            median_logit = prior_logit + math.copysign(2.0, dev)
        median_logit = max(logit(0.02), min(ceiling_logit, median_logit))
        spread = math.sqrt(
            0.30 ** 2
            + min(0.6, resid) ** 2
            + (0.12 * math.sqrt(horizon_days)) ** 2
        ) + extra

    spread = max(0.25, min(1.0, spread))
    qs = [round4(clamp_pct(inv_logit(median_logit + z * spread))) for z in Z]
    # Enforce non-decreasing (rounding can only tie, never invert, but be safe).
    for j in range(1, 5):
        if qs[j] < qs[j - 1]:
            qs[j] = qs[j - 1]
    return {"id": trophy["id"], "q": qs, "median_logit": median_logit, "spread": spread}


def enforce_structure(rows_by_id):
    """Story ordering and platinum/100% caps, on median logits."""
    def get(i):
        return rows_by_id[i]["median_logit"]

    def setm(i, v):
        rows_by_id[i]["median_logit"] = v

    # Story chain non-increasing (pool adjacent violators).  Only the
    # chain trophies actually present in the case take part.
    chain = [i for i in (3, 4, 5, 6, 7) if i in rows_by_id]
    vals = [get(i) for i in chain]
    # Simple isotonic regression (decreasing) by pooling.
    pooled = [[v] for v in vals]
    j = 0
    while j < len(pooled) - 1:
        a = sum(pooled[j]) / len(pooled[j])
        b = sum(pooled[j + 1]) / len(pooled[j + 1])
        if a < b:
            pooled[j] = pooled[j] + pooled[j + 1]
            del pooled[j + 1]
            if j > 0:
                j -= 1
        else:
            j += 1
    k = 0
    for group in pooled:
        avg = sum(group) / len(group)
        for _ in group:
            setm(chain[k], avg)
            k += 1

    # Story missions (8..13) sit between Chapter 1 and the finale.
    if 3 in rows_by_id and 7 in rows_by_id:
        lo = get(7)
        hi = get(3)
        for i in range(8, 14):
            if i in rows_by_id:
                setm(i, max(lo, min(hi, get(i))))
    # 100% completion below the finale; platinum below 100%.
    if 14 in rows_by_id and 7 in rows_by_id:
        setm(14, min(get(14), get(7)))
    if 1 in rows_by_id and 14 in rows_by_id:
        setm(1, min(get(1), get(14)))


def forecast_case(case, name):
    if len(case.get("levels", [])) != 5:
        raise ValueError(f"{name} asks for {case.get('levels')}, not five levels")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    release = RELEASE
    try:
        rd = case.get("game", {}).get("release_date")
        if rd:
            release = datetime.fromisoformat(rd).replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    target = parse_time(case["target"])
    hours_target = (target - release).total_seconds() / 3600.0
    x_target = math.log1p(max(0.0, hours_target))
    horizon_days = case["horizon_minutes"] / 1440.0

    rows_by_id = {}
    for idx, trophy in enumerate(case["trophies"]):
        rows_by_id[trophy["id"]] = forecast_trophy(
            trophy, idx, case, release, x_target, horizon_days
        )
    enforce_structure(rows_by_id)
    rows = []
    for trophy in case["trophies"]:
        r = rows_by_id[trophy["id"]]
        m = r["median_logit"]
        s = r["spread"]
        qs = [round4(clamp_pct(inv_logit(m + z * s))) for z in Z]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
