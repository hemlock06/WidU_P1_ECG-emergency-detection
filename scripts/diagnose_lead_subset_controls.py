"""Bounded historical-control diagnosis; never changes or passes the run gate."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ablation_exhaustive_lead_subsets as e

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=list(e.MODELS), default="P1_a07")
args = parser.parse_args()
x, ids, yb, ym = e.data_for(args.model)
model = e.model_for(args.model)
leads = tuple(range(12)) if args.model == "P1_a07" else (0, 1)
out = "lead_subset_control_diagnosis" + ("_reference" if ym is None else "")
rows = []
# Predeclared diagnostic settings, not a search for a passing seed.
settings = [
    (32, 42, True),
    (32, 43, True),
    (32, 44, True),
    (16, 42, True),
    (32, 42, False),
    (32, 43, False),
]
forward = model[0].forward
first = {}
for batch, seed, mask in settings:
    if mask:
        model[0].forward = forward
    else:
        model[0].forward = lambda *a, **kw: forward(*a, **dict(kw, mask=False))
    start = time.perf_counter()
    pb, pm = e.predict(model, x, leads, batch, seed)
    probabilities = pb if pm is None else pm
    key = (batch, mask)
    delta = None if key not in first else float(np.max(np.abs(first[key] - probabilities)))
    first.setdefault(key, probabilities.copy())
    row = {
        "batch": batch,
        "seed": seed,
        "feature_mask": mask,
        "seconds": time.perf_counter() - start,
        "max_probability_delta_from_first_same_batch_mask": delta,
        **e.metrics(yb, pb, ym, pm),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
    Path(f"results/{out}.json").write_text(
        json.dumps(
            {
                "purpose": "diagnosis only; no historical gate override or seed selection",
                "model": args.model,
                "leads": leads,
                "settings_predeclared": settings,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
