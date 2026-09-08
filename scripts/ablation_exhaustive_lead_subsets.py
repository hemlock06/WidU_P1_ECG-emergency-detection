"""Frozen P1 and reference-model lead subsets, gated by historical controls.

Run from the repository root. The controls stage must pass before the run stage.
No training, data reconstruction, test-set selection, or threshold tuning occurs.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from ablation_nlead_curve import (
    LEAD_NAMES,
    BinaryHead,
    MCHead,
    apply_lead_mask,
    inject_lora,
    load_ecgfm,
)
from sklearn.metrics import f1_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from verify_lead_subset_assets import sha256


class MaskedTest(Dataset):
    def __init__(self, x, yb, ym, leads):
        self.x, self.yb, self.ym, self.leads = x, yb, ym, leads

    def __len__(self):
        return len(self.yb)

    def __getitem__(self, index):
        x = torch.from_numpy(np.array(self.x[index], copy=True))
        x = apply_lead_mask(x.unsqueeze(0), self.leads).squeeze(0)
        if self.ym is None:
            return x, torch.tensor(float(self.yb[index]))
        return x, torch.tensor(float(self.yb[index])), torch.tensor(int(self.ym[index]))


def original_code_check(model, x, yb, ym, leads, pb, pm, batch, seed=42):
    """Compare four control configurations with the existing evaluation paths."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if ym is None:
        from ablation_nlead_curve import eval_leads

        observed = eval_leads(
            model[0],
            model[1],
            MaskedTest(x, yb, None, leads),
            list(range(12)),
            torch.device("cuda"),
            batch_size=batch,
        )
        if abs(observed - roc_auc_score(yb, pb)) > 1e-7:
            raise ValueError("Existing binary evaluation disagrees")
    else:
        from train_lora_multitask import evaluate

        loader = DataLoader(
            MaskedTest(x, yb, ym, leads), batch_size=batch, shuffle=False, num_workers=0
        )
        observed = evaluate(*model, loader, torch.device("cuda"))
        np.testing.assert_allclose(observed["mc_probs"], pm, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(observed["bin_probs"], pb, atol=1e-6, rtol=1e-6)
    return "passed"


CLASSES = ["nsr", "af", "ischemia", "conduction", "ectopy"]
MODELS = {
    "P1_a07": (
        "outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt",
        "cpsc2018_mc",
        936,
    ),
    "reference_iii": ("outputs/lora_multisnr/lora_multisnr_best.pt", "cpsc2018", 474),
}
CONTROLS = [tuple(range(12)), (0, 1, 7, 10), (0, 1), (1,)]
HISTORY_III = [(0.9469, 0.7734), (0.9497, 0.7875), (0.9519, 0.7762), (0.9446, 0.7564)]


def subsets():
    return [c for n in range(1, 5) for c in itertools.combinations(range(12), n)]


def sensitivity(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    # Same operating-point rule as train_lora_multitask.evaluate.
    i = np.searchsorted((1 - fpr)[::-1], 0.95)
    return float(tpr[::-1][i])


def metrics(yb, pb, ym=None, pm=None):
    result = {
        "binary_auroc": float(roc_auc_score(yb, pb)),
        "sens_at_95sp": sensitivity(yb, pb),
        "f1_at_05": float(f1_score(yb, pb >= 0.5, zero_division=0)),
        "macro_f1": "",
        "macro_sensitivity": "",
        "measurable": "binary;multiclass" if pm is not None else "binary",
    }
    for name in CLASSES:
        for metric in ("auroc", "f1", "sensitivity", "sens95"):
            result[f"{metric}_{name}"] = ""
    if pm is not None:
        pred = pm.argmax(1)
        result["macro_f1"] = float(
            f1_score(ym, pred, labels=range(5), average="macro", zero_division=0)
        )
        result["macro_sensitivity"] = float(
            recall_score(ym, pred, labels=range(5), average="macro", zero_division=0)
        )
        for c, name in enumerate(CLASSES):
            truth = (ym == c).astype(int)
            result[f"auroc_{name}"] = float(roc_auc_score(truth, pm[:, c]))
            result[f"f1_{name}"] = float(f1_score(truth, pred == c, zero_division=0))
            result[f"sensitivity_{name}"] = float(
                recall_score(truth, pred == c, zero_division=0)
            )
            result[f"sens95_{name}"] = sensitivity(truth, pm[:, c])
    if any(isinstance(v, float) and not np.isfinite(v) for v in result.values()):
        raise ValueError("Nonfinite performance metric")
    return result


def asset_identity(seed, batch):
    paths = ["checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"]
    for checkpoint, dataset, _ in MODELS.values():
        paths.append(checkpoint)
        root = f"data/processed/{dataset}/test"
        paths.extend(
            f"{root}/{name}" for name in ("signals.npy", "labels.npy", "record_ids.npy")
        )
        if dataset == "cpsc2018_mc":
            paths.append(f"{root}/labels_bin.npy")
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Required assets missing: " + ", ".join(missing))
    hashes = {p: sha256(Path(p)) for p in paths}
    for p in (
        "scripts/ablation_exhaustive_lead_subsets.py",
        "scripts/ablation_nlead_curve.py",
        "patches/lora_fairseq_signals.patch",
    ):
        hashes[p] = sha256(Path(p))
    import fairseq_signals

    package_root = Path(fairseq_signals.__file__).parent
    package_hashes = {
        p.relative_to(package_root).as_posix(): sha256(p)
        for p in sorted(package_root.rglob("*.py"))
    }
    hashes["fairseq_signals_python_source"] = hashlib.sha256(
        json.dumps(package_hashes, sort_keys=True).encode()
    ).hexdigest()
    identity = {
        "sha256": hashes,
        "seed": seed,
        "batch": batch,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "scipy",
                "scikit-learn",
                "wfdb",
                "omegaconf",
                "hydra-core",
                "fairseq_signals",
            )
        },
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "tf32": False,
        "precision": "float32",
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, digest


def data_for(name):
    _, dataset, count = MODELS[name]
    root = Path("data/processed") / dataset / "test"
    x = np.load(root / "signals.npy", mmap_mode="r")
    ids = np.load(root / "record_ids.npy", allow_pickle=True).astype(str)
    labels = np.load(root / "labels.npy")
    if x.shape != (count, 12, 5000) or x.dtype != np.float32:
        raise ValueError(f"{dataset}: unexpected signal shape/dtype")
    if labels.shape != (count,) or len(ids) != count or len(set(ids)) != count:
        raise ValueError(f"{dataset}: invalid labels/record IDs")
    if not np.isfinite(x).all():
        raise ValueError(f"{dataset}: nonfinite signals")
    if name == "P1_a07":
        yb = np.load(root / "labels_bin.npy")
        if not np.array_equal(
            np.bincount(labels, minlength=5), [130, 180, 165, 379, 82]
        ):
            raise ValueError("P1 class supports differ from historical test")
        if not np.array_equal(yb, np.isin(labels, [1, 2]).astype(yb.dtype)):
            raise ValueError("P1 binary/multiclass label mismatch")
        return x, ids, yb, labels
    if not np.array_equal(np.bincount(labels, minlength=2), [121, 353]):
        raise ValueError("Reference class supports differ from historical test")
    return x, ids, labels, None


def model_for(name):
    # Local hashes identify the explicitly restored, trusted model files.
    checkpoint = torch.load(MODELS[name][0], map_location="cpu", weights_only=False)
    bb = load_ecgfm(torch.device("cpu"))
    inject_lora(bb, rank=8, alpha=16.0, dropout=0.0)
    bb.load_state_dict(checkpoint["backbone_lora"], strict=True)
    hb = BinaryHead()
    hb.load_state_dict(
        checkpoint["head_bin_state" if name == "P1_a07" else "head_state"]
    )
    hm = None
    if name == "P1_a07":
        if checkpoint.get("epoch") != 18 or checkpoint.get("alpha") != 0.7:
            raise ValueError("Not the canonical epoch-18 alpha-0.7 model")
        hm = MCHead()
        hm.load_state_dict(checkpoint["head_mc_state"])
        hm.cuda().eval()
    elif checkpoint.get("epoch") != 30:
        raise ValueError("Not the canonical epoch-30 binary reference model")
    return bb.cuda().eval(), hb.cuda().eval(), hm


@torch.inference_mode()
def predict(model, x, leads, batch, seed=42):
    # Legacy forward(mask=True) samples NumPy feature masks even in eval mode.
    # Common random numbers keep masks identical across subsets and resumed runs.
    np.random.seed(seed)
    torch.manual_seed(seed)
    bb, hb, hm = model
    binary, multiclass = [], []
    for start in range(0, len(x), batch):
        source = torch.from_numpy(np.array(x[start : start + batch], copy=True)).cuda()
        source = apply_lead_mask(source, leads)
        z = bb(source=source, padding_mask=None, features_only=True)["x"].mean(1)
        binary.append(hb(z).sigmoid().cpu().numpy())
        if hm is not None:
            multiclass.append(hm(z).softmax(-1).cpu().numpy())
    pb = np.concatenate(binary)
    pm = np.concatenate(multiclass) if multiclass else None
    if not np.isfinite(pb).all() or (pm is not None and not np.isfinite(pm).all()):
        raise ValueError("Nonfinite predictions")
    return pb, pm


def historical_checks(name, leads, row):
    if name == "reference_iii":
        auroc, sens = HISTORY_III[CONTROLS.index(leads)]
        targets = {"binary_auroc": (auroc, 0.0021), "sens_at_95sp": (sens, 0.02)}
    elif len(leads) == 12:
        targets = {
            "binary_auroc": (0.9139, 0.0021),
            "sens_at_95sp": (0.7072, 0.02),
            "f1_at_05": (0.7887, 0.005),
            "macro_f1": (0.6858, 0.005),
        }
        for c, value in zip(CLASSES, [0.9304, 0.9671, 0.9066, 0.9577, 0.8634]):
            targets[f"auroc_{c}"] = (value, 0.0021)
    else:
        return {
            "status": "not_available",
            "reason": "No historical P1 fixed-subset value",
        }
    checks = {
        k: {
            "expected": v,
            "observed": row[k],
            "tolerance": t,
            "passed": abs(row[k] - v) <= t,
        }
        for k, (v, t) in targets.items()
    }
    return {
        "status": "passed" if all(c["passed"] for c in checks.values()) else "failed",
        "checks": checks,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["controls", "run"], default="controls")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.batch < 1 or not torch.cuda.is_available():
        raise RuntimeError("Positive batch size and CUDA are required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    identity, fingerprint = asset_identity(args.seed, args.batch)
    root = Path("results")
    root.mkdir(exist_ok=True)
    manifest = root / "exhaustive_lead_subsets_controls.json"
    if args.stage == "run":
        gate = json.loads(manifest.read_text(encoding="utf-8"))
        if gate["fingerprint"] != fingerprint or gate["status"] != "passed":
            raise RuntimeError("Matching positive controls have not passed")
    controls = {
        "fingerprint": fingerprint,
        "identity": identity,
        "rows": [],
        "status": "failed",
    }
    if args.stage == "controls":
        manifest.write_text(json.dumps(controls, indent=2), encoding="utf-8")
    output = root / (
        "exhaustive_lead_subsets_1to4.csv"
        if args.stage == "run"
        else "exhaustive_lead_subsets_controls.csv"
    )
    existing = []
    if output.exists() and args.stage == "run":
        with output.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
        if any(r["run_fingerprint"] != fingerprint for r in existing):
            raise RuntimeError("Resume identity mismatch")
    completed = {(r["model"], r["lead_indices"]) for r in existing}
    previous_rows = {(r["model"], r["lead_indices"]): r for r in existing}
    if len(completed) != len(existing):
        raise RuntimeError("Duplicate resume rows")
    prediction_dir = Path("outputs/exhaustive_lead_subsets") / fingerprint
    prediction_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    with output.open("a" if existing else "w", newline="", encoding="utf-8") as stream:
        writer = None
        for name in MODELS:
            x, ids, yb, ym = data_for(name)
            model = model_for(name)
            for leads in CONTROLS if args.stage == "controls" else subsets():
                key = ";".join(map(str, leads))
                if (name, key) in completed:
                    prediction_path = (
                        prediction_dir / f"{name}_{'-'.join(map(str, leads))}.npz"
                    )
                    if not prediction_path.is_file():
                        raise FileNotFoundError(
                            f"Missing resumed predictions: {prediction_path}"
                        )
                    if (
                        sha256(prediction_path)
                        != previous_rows[(name, key)]["prediction_sha256"]
                    ):
                        raise ValueError(f"Prediction hash mismatch: {prediction_path}")
                    skipped += 1
                    continue
                start = time.perf_counter()
                pb, pm = predict(model, x, leads, args.batch, args.seed)
                row = {
                    "model": name,
                    "n_leads": len(leads),
                    "lead_indices": key,
                    "lead_names": ";".join(LEAD_NAMES[i] for i in leads),
                    "n_samples": len(x),
                    **metrics(yb, pb, ym, pm),
                    "runtime_sec": time.perf_counter() - start,
                    "seed": args.seed,
                    "checkpoint_sha256": identity["sha256"][MODELS[name][0]],
                    "data_fingerprint": hashlib.sha256(
                        json.dumps(
                            {
                                k: v
                                for k, v in identity["sha256"].items()
                                if k.startswith(
                                    f"data/processed/{MODELS[name][1]}/test/"
                                )
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest(),
                    "run_fingerprint": fingerprint,
                }
                prediction_path = (
                    prediction_dir / f"{name}_{'-'.join(map(str, leads))}.npz"
                )
                payload = {"record_ids": ids, "labels_bin": yb, "binary_probs": pb}
                if pm is not None:
                    payload.update(labels_mc=ym, multiclass_probs=pm)
                np.savez_compressed(prediction_path, **payload)
                row["prediction_sha256"] = sha256(prediction_path)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row.keys())
                    if stream.tell() == 0:
                        writer.writeheader()
                writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
                if args.stage == "controls":
                    implementation_check = original_code_check(
                        model, x, yb, ym, leads, pb, pm, args.batch, args.seed
                    )
                    controls["rows"].append(
                        {
                            "model": name,
                            "leads": list(leads),
                            "original_code_check": implementation_check,
                            **historical_checks(name, leads, row),
                        }
                    )
                completed.add((name, key))
                print(
                    name,
                    row["lead_names"],
                    row["binary_auroc"],
                    row["macro_f1"],
                    flush=True,
                )
            del model
            torch.cuda.empty_cache()
    if args.stage == "controls":
        # Historical gates: P1 full lead plus all four reference-model configurations.
        checks = [r for r in controls["rows"] if r["status"] != "not_available"]
        controls["status"] = (
            "passed"
            if len(checks) == 5 and all(r["status"] == "passed" for r in checks)
            else "failed"
        )
        manifest.write_text(json.dumps(controls, indent=2), encoding="utf-8")
        if controls["status"] != "passed":
            raise RuntimeError(
                "Historical controls failed; exhaustive evaluation is blocked"
            )
    else:
        expected = {(m, ";".join(map(str, c))) for m in MODELS for c in subsets()}
        if completed != expected:
            raise RuntimeError("Expected exactly 793 unique subsets per model")
    status = {
        "stage": args.stage,
        "rows": len(completed),
        "resumed_skips": skipped,
        "failures": 0,
        "run_fingerprint": fingerprint,
    }
    (root / f"exhaustive_lead_subsets_{args.stage}_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure_dir = Path("results/exhaustive_lead_subsets_failures")
        failure_dir.mkdir(parents=True, exist_ok=True)
        (failure_dir / f"{time.time_ns()}.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "traceback_paths": "repository_relative",
                    "traceback": traceback.format_exc().replace(str(Path.cwd()), "."),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
