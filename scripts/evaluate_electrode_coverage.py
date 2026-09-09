"""Extend frozen P1 evaluation to all 4095 subsets with electrode provenance."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import ablation_exhaustive_lead_subsets as e
import numpy as np
import torch

ROOT = Path("results/electrode_coverage_v1")
PROTOCOL = Path("records/electrode_coverage_protocol_v1.md")
PARENT = "34060e20ca6ca6de13d114a5b1b14cf4e1fd9ffccc3f7ec4453bb00d8b5646f1"


def now():
    return datetime.now(timezone.utc).isoformat()


def contacts(combo):
    points = set()
    for i in combo:
        points.update(
            ({"RA", "LA"}, {"RA", "LL"}, {"LA", "LL"})[i]
            if i < 3
            else {"RA", "LA", "LL"}
        )
        if i >= 6:
            points.add(e.LEAD_NAMES[i])
    return tuple(sorted(points))


def all_subsets():
    values = [c for n in range(1, 13) for c in itertools.combinations(range(12), n)]
    return sorted(values, key=lambda c: (len(contacts(c)), len(c), c))


def key(combo):
    return ";".join(map(str, combo))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def event(kind, **detail):
    with (ROOT / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": now(), "event": kind, **detail}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive_run():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "run.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def identity():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parent_identity, parent = e.asset_identity(42, 32)
    if parent != PARENT:
        raise ValueError(
            "Parent environment/assets/source changed; do not mix experiments"
        )
    gate = json.loads(
        Path("results/exhaustive_lead_subsets_stochastic_controls.json").read_text()
    )
    if (
        gate["fingerprint"] != parent
        or gate["status"] != "passed_stochastic_reproduction"
    ):
        raise ValueError("Parent stochastic gate invalid")
    value = {
        "parent_fingerprint": parent,
        "parent_identity": parent_identity,
        "script_sha256": e.sha256(Path(__file__)),
        "protocol_sha256": e.sha256(PROTOCOL),
        "parent_raw_sha256": e.sha256(Path("results/exhaustive_lead_subsets_1to4.csv")),
        "parent_controls_sha256": e.sha256(
            Path("results/exhaustive_lead_subsets_controls.csv")
        ),
        "stochastic_gate_sha256": e.sha256(
            Path("results/exhaustive_lead_subsets_stochastic_controls.json")
        ),
        "model": "P1_a07",
        "seed": 42,
        "batch": 32,
        "expected_rows": 4095,
    }
    run_id = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value, run_id


def checked_prediction(row, ids, yb, ym):
    path = Path(row["prediction_path"])
    if e.sha256(path) != row["prediction_sha256"]:
        raise ValueError(f"Prediction hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as stored:
        for name, expected in (
            ("record_ids", ids),
            ("labels_bin", yb),
            ("labels_mc", ym),
        ):
            np.testing.assert_array_equal(stored[name], expected)
        pb, pm = stored["binary_probs"], stored["multiclass_probs"]
        if pb.shape != (936,) or pm.shape != (936, 5):
            raise ValueError("Prediction shape mismatch")
        for p in (pb, pm):
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ValueError("Invalid probabilities")
        np.testing.assert_allclose(pm.sum(1), 1, atol=1e-6)
        return pb, pm


def imported_rows(ids, yb, ym, run_id):
    rows = [
        r
        for r in read_rows(Path("results/exhaustive_lead_subsets_1to4.csv"))
        if r["model"] == "P1_a07"
    ]
    control = next(
        r
        for r in read_rows(Path("results/exhaustive_lead_subsets_controls.csv"))
        if r["model"] == "P1_a07" and r["n_leads"] == "12"
    )
    rows.append(control)
    if len(rows) != 794 or len({r["lead_indices"] for r in rows}) != 794:
        raise ValueError("Expected 794 unique imported configurations")
    for r in rows:
        if r["run_fingerprint"] != PARENT:
            raise ValueError("Imported parent identity mismatch")
        combo = tuple(map(int, r["lead_indices"].split(";")))
        r["prediction_path"] = (
            Path("outputs/exhaustive_lead_subsets")
            / PARENT
            / f"P1_a07_{'-'.join(map(str, combo))}.npz"
        ).as_posix()
        checked_prediction(r, ids, yb, ym)
        r.update(
            stage1_run_id=run_id,
            origin="prior_control" if len(combo) == 12 else "prior_exhaustive",
            measurement_contacts=len(contacts(combo)),
            contact_names=";".join(contacts(combo)),
            contacts_with_extra_drl=len(contacts(combo)) + 1,
        )
    return rows


def save_prediction(path, ids, yb, ym, pb, pm):
    if path.exists():
        raise FileExistsError(f"Preserve existing predictions and investigate: {path}")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        record_ids=ids,
        labels_bin=yb,
        labels_mc=ym,
        binary_probs=pb,
        multiclass_probs=pm,
    )
    temporary.replace(path)


def prepare(value, run_id, x, ids, yb, ym):
    existing = ROOT / "preflight.json"
    if existing.exists():
        prior = json.loads(existing.read_text())
        if prior["run_id"] != run_id or prior["status"] != "passed":
            raise ValueError("Existing preflight differs; preserve it and investigate")
        print("Matching preflight already passed", flush=True)
        return
    imported = imported_rows(ids, yb, ym, run_id)
    parent_controls = [
        r
        for r in read_rows(Path("results/exhaustive_lead_subsets_controls.csv"))
        if r["model"] == "P1_a07"
    ]
    model = e.model_for("P1_a07")
    pred_dir = (
        Path("outputs/electrode_coverage_v1")
        / run_id
        / "preflight"
        / str(time.time_ns())
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    for combo in e.CONTROLS + [tuple(range(6)), tuple(range(6)) + (7,)]:
        pb, pm = e.predict(model, x, combo, 32, 42)
        path = pred_dir / f"{'-'.join(map(str, combo))}.npz"
        save_prediction(path, ids, yb, ym, pb, pm)
        if combo in e.CONTROLS:
            r = next(r for r in parent_controls if r["lead_indices"] == key(combo))
            r["prediction_path"] = (
                Path("outputs/exhaustive_lead_subsets")
                / PARENT
                / f"P1_a07_{'-'.join(map(str, combo))}.npz"
            ).as_posix()
            old_pb, old_pm = checked_prediction(r, ids, yb, ym)
            np.testing.assert_allclose(pb, old_pb, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(pm, old_pm, atol=1e-6, rtol=1e-6)
            method = "full_dataset_saved_probabilities_parity"
        else:
            e.original_code_check(model, x, yb, ym, combo, pb, pm, 32, 42)
            method = "expanded_input_original_implementation_parity"
        checks.append(
            {
                "leads": key(combo),
                "status": "passed",
                "method": method,
                "prediction_path": path.as_posix(),
                "prediction_sha256": e.sha256(path),
                "metrics": e.metrics(yb, pb, ym, pm),
            }
        )
        print("preflight", key(combo), "passed", flush=True)
    receipt = {
        "status": "passed",
        "run_id": run_id,
        "identity": value,
        "utc": now(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "imported_prediction_files_verified": len(imported),
        "checks": checks,
    }
    atomic_json(existing, receipt)
    event("preflight_passed", run_id=run_id, controls=len(checks))


def run(value, run_id, x, ids, yb, ym):
    gate = json.loads((ROOT / "preflight.json").read_text())
    if gate["status"] != "passed" or gate["run_id"] != run_id:
        raise ValueError("Matching current preflight required")
    for check in gate["checks"]:
        if e.sha256(Path(check["prediction_path"])) != check["prediction_sha256"]:
            raise ValueError("Preflight prediction was modified")
    prior = imported_rows(ids, yb, ym, run_id)
    output = ROOT / "raw.csv"
    rows = read_rows(output) if output.exists() else prior
    completed = {r["lead_indices"] for r in rows}
    expected = {key(c) for c in all_subsets()}
    if len(completed) != len(rows) or not completed <= expected:
        raise ValueError("Duplicate/invalid resumed subsets")
    if not {r["lead_indices"] for r in prior} <= completed:
        raise ValueError("Resumed CSV lost imported rows; preserve and investigate")
    for row in rows:
        if row["stage1_run_id"] != run_id:
            raise ValueError("Resume identity mismatch")
        checked_prediction(row, ids, yb, ym)
    if not output.exists():
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=prior[0].keys())
            writer.writeheader()
            writer.writerows(prior)
            stream.flush()
            os.fsync(stream.fileno())
    progress = {
        "status": "running",
        "run_id": run_id,
        "expected_rows": 4095,
        "rows": len(rows),
        "reused_parent_rows": 794,
        "resumed_new_rows": len(rows) - 794,
        "new_rows_this_invocation": 0,
        "failures_this_invocation": 0,
        "utc_started": now(),
    }
    atomic_json(ROOT / "status.json", progress)
    event("run_started", **progress)
    model = e.model_for("P1_a07")
    pred_dir = Path("outputs/electrode_coverage_v1") / run_id / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with output.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prior[0].keys())
        for combo in all_subsets():
            if key(combo) in completed:
                continue
            clock = time.perf_counter()
            pb, pm = e.predict(model, x, combo, 32, 42)
            scores = e.metrics(yb, pb, ym, pm)
            path = pred_dir / f"{'-'.join(map(str, combo))}.npz"
            save_prediction(path, ids, yb, ym, pb, pm)
            row = {
                **prior[0],
                **scores,
                "n_leads": len(combo),
                "lead_indices": key(combo),
                "lead_names": ";".join(e.LEAD_NAMES[i] for i in combo),
                "runtime_sec": time.perf_counter() - clock,
                "prediction_path": path.as_posix(),
                "prediction_sha256": e.sha256(path),
                "origin": "new_inference",
                "measurement_contacts": len(contacts(combo)),
                "contact_names": ";".join(contacts(combo)),
                "contacts_with_extra_drl": len(contacts(combo)) + 1,
            }
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
            completed.add(key(combo))
            progress.update(
                rows=len(completed),
                utc_updated=now(),
                new_rows_this_invocation=progress["new_rows_this_invocation"] + 1,
                elapsed_sec=time.perf_counter() - start,
            )
            atomic_json(ROOT / "status.json", progress)
            print(
                len(completed),
                "/4095",
                row["contact_names"],
                row["lead_names"],
                row["macro_f1"],
                flush=True,
            )
    if completed != expected:
        raise ValueError("Incomplete expanded experiment")
    progress.update(
        status="inference_complete_pending_independent_audit",
        raw_sha256=e.sha256(output),
        utc_finished=now(),
    )
    atomic_json(ROOT / "status.json", progress)
    event("inference_complete", **progress)
    print(json.dumps(progress), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "run"], required=True)
    args = parser.parse_args()
    with exclusive_run():
        try:
            value, run_id = identity()
            x, ids, yb, ym = e.data_for("P1_a07")
            event("invocation", stage=args.stage, run_id=run_id, pid=os.getpid())
            (prepare if args.stage == "prepare" else run)(value, run_id, x, ids, yb, ym)
        except Exception:
            event(
                "failure",
                stage=args.stage,
                traceback=traceback.format_exc().replace(str(Path.cwd()), "."),
            )
            raise


if __name__ == "__main__":
    main()
