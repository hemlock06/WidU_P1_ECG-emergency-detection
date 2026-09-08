"""Five-record GPU/code agreement only; not held-out performance reproduction."""

import json
from pathlib import Path

import numpy as np
import torch
from ablation_exhaustive_lead_subsets import (
    CONTROLS,
    model_for,
    original_code_check,
    predict,
)
from preprocess_cpsc2018_mc import load_signal, map_label_mc, parse_dx_from_hea


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Fixed smoke IDs in class order, selected before complete raw download.
    ids = ["A0002", "A0003", "A0008", "A0001", "A0005"]
    signals = []
    for label, record in enumerate(ids):
        header = Path("data/raw/cpsc2018") / f"{record}.hea"
        if map_label_mc(parse_dx_from_hea(str(header))) != label:
            raise ValueError(f"Smoke label changed: {record}")
        signal = load_signal(str(header))
        if signal is None:
            raise ValueError(f"Smoke signal missing/invalid: {record}")
        signals.append(signal)
    x = np.stack(signals)
    ym = np.arange(5)
    yb = np.isin(ym, [1, 2]).astype(int)
    model = model_for("P1_a07")
    report = {
        "scope": "Five-record implementation smoke only, not historical metric reproduction or performance evaluation",
        "records": ids,
        "seed": 42,
        "controls": [],
    }
    for leads in CONTROLS:
        pb, pm = predict(model, x, leads, 5, seed=42)
        agreement = original_code_check(model, x, yb, ym, leads, pb, pm, 5, seed=42)
        report["controls"].append(
            {
                "leads": list(leads),
                "original_code_probability_agreement": agreement,
                "binary_shape": list(pb.shape),
                "multiclass_shape": list(pm.shape),
            }
        )
    destination = Path("results/lead_subset_four_mask_smoke.json")
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
