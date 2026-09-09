"""After inference finishes, replay the restart boundary without replacing source predictions."""

import json
import time
from pathlib import Path

import analyze_electrode_coverage as a
import evaluate_electrode_coverage as runner
import numpy as np


def main():
    with runner.exclusive_run():
        status = json.loads((runner.ROOT / "status.json").read_text())
        if status["status"] != "inference_complete_pending_independent_audit":
            raise ValueError(
                "Finish original inference before allocating a replay model"
            )
        _, run_id = runner.identity()
        raw = runner.read_rows(runner.ROOT / "raw.csv")
        a.validate_rows(raw, run_id)
        recovery = json.loads(
            Path("results/electrode_resume_audit_20260909_1748.json").read_text()
        )
        boundary = recovery["saved_rows"]
        selected = [
            raw[boundary - 1],
            raw[boundary],
            next(r for r in raw if int(r["n_leads"]) == 12),
        ]
        x, ids, yb, ym = runner.e.data_for("P1_a07")
        model = runner.e.model_for("P1_a07")
        target = (
            Path("outputs/electrode_coverage_v1")
            / run_id
            / "restart_parity"
            / str(time.time_ns())
        )
        target.mkdir(parents=True, exist_ok=False)
        checks = []
        for row in selected:
            old_pb, old_pm = a.check_prediction(row, ids, yb, ym)
            pb, pm = runner.e.predict(model, x, a.combo(row), 32, 42)
            path = target / ("-".join(map(str, a.combo(row))) + ".npz")
            runner.save_prediction(path, ids, yb, ym, pb, pm)
            np.testing.assert_allclose(pb, old_pb, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(pm, old_pm, atol=1e-6, rtol=1e-6)
            np.testing.assert_array_equal(pm.argmax(1), old_pm.argmax(1))
            checks.append(
                {
                    "leads": row["lead_indices"],
                    "status": "passed",
                    "maximum_probability_error": float(
                        max(np.max(abs(pb - old_pb)), np.max(abs(pm - old_pm)))
                    ),
                    "prediction_path": path.as_posix(),
                    "prediction_sha256": a.digest(path),
                    "source_prediction_sha256": row["prediction_sha256"],
                }
            )
        a.write_json(
            runner.ROOT / "restart_parity.json",
            {
                "status": "passed",
                "run_id": run_id,
                "utc": runner.now(),
                "raw_sha256": a.digest(runner.ROOT / "raw.csv"),
                "script_sha256": a.digest(__file__),
                "checks": checks,
                "scope": "Replay last pre-interruption, first resumed, and 12-lead predictions; not every-combination replay",
            },
        )
        print("Restart parity: three complete-cohort comparisons passed")


if __name__ == "__main__":
    main()
