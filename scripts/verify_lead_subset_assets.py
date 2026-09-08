"""Audit immutable evaluation assets before lead-subset inference; never trains."""

import argparse
import hashlib
import json
import platform
import traceback
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--out", default="results/lead_subset_asset_audit.json")
    args = parser.parse_args()
    report = {"python": platform.python_version(), "assets": {}, "errors": []}
    for name in (
        "checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt",
        "outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt",
        "outputs/lora_multisnr/lora_multisnr_best.pt",
    ):
        path = Path(name)
        if path.is_file() and path.stat().st_size:
            report["assets"][name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        else:
            report["errors"].append(f"Missing checkpoint: {name}")
    import numpy as np

    for dataset, count, label_names in (
        ("cpsc2018", 474, ("labels.npy",)),
        ("cpsc2018_mc", 936, ("labels.npy", "labels_bin.npy")),
    ):
        root = Path("data/processed") / dataset / "test"
        for name in ("signals.npy", *label_names, "record_ids.npy"):
            path = root / name
            if not path.is_file():
                report["errors"].append(f"Missing test asset: {path.as_posix()}")
                continue
            array = np.load(path, allow_pickle=name == "record_ids.npy")
            entry = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": sha256(path),
            }
            report["assets"][path.as_posix()] = entry
            if len(array) != count:
                report["errors"].append(f"{path}: expected {count}, got {len(array)}")
            if name == "signals.npy":
                if (
                    array.ndim != 3
                    or array.shape[1:] != (12, 5000)
                    or array.dtype != np.float32
                ):
                    report["errors"].append(f"Invalid signal format: {path}")
                entry["nonfinite"] = int((~np.isfinite(array)).sum())
                if entry["nonfinite"]:
                    report["errors"].append(f"Nonfinite signal: {path}")
            elif name == "record_ids.npy":
                entry["unique_records"] = len(set(array.tolist()))
                if entry["unique_records"] != len(array):
                    report["errors"].append(f"Record IDs not unique: {path}")
            else:
                values, counts = np.unique(array, return_counts=True)
                entry["distribution"] = dict(zip(map(str, values), map(int, counts)))
                if array.ndim != 1:
                    report["errors"].append(f"Not single-label vector: {path}")
    if args.forward:
        try:
            import torch
            from train_lora_multitask import (
                BinaryHead,
                MulticlassHead,
                inject_lora,
                load_ecgfm,
            )

            report["torch"] = torch.__version__
            report["cuda_runtime"] = torch.version.cuda
            report["cuda_available"] = torch.cuda.is_available()
            report["compiled_architectures"] = torch.cuda.get_arch_list()
            report["gpu"] = torch.cuda.get_device_name(0)
            report["compute_capability"] = list(torch.cuda.get_device_capability(0))
            torch.manual_seed(42)
            model = load_ecgfm(
                "checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt", "cuda"
            )
            ck = torch.load(
                "outputs/lora_multitask_snr_a07/lora_multitask_snr_best.pt",
                map_location="cpu",
                weights_only=False,
            )
            report["checkpoint_metadata"] = {
                k: ck.get(k)
                for k in ("epoch", "alpha", "lora_rank", "lora_alpha", "val_composite")
            }
            replaced = inject_lora(
                model, ck.get("lora_rank", 8), ck.get("lora_alpha", 16.0), 0.0
            )
            report["lora_modules"] = len(replaced)
            state = model.load_state_dict(ck["backbone_lora"], strict=False)
            report["missing_keys"] = state.missing_keys
            report["unexpected_keys"] = state.unexpected_keys
            if state.missing_keys or state.unexpected_keys:
                raise RuntimeError("Checkpoint/backbone state mismatch; see key lists")
            hb, hm = BinaryHead().cuda(), MulticlassHead().cuda()
            hb.load_state_dict(ck["head_bin_state"])
            hm.load_state_dict(ck["head_mc_state"])
            model.cuda().eval()
            hb.eval()
            hm.eval()
            with torch.inference_mode():
                # Hardware/model smoke only; synthetic input is never scored as ECG data.
                x = torch.randn(1, 12, 5000, device="cuda") * 0.1
                z = model(source=x, padding_mask=None, features_only=True)["x"].mean(1)
                binary, multiclass = hb(z), hm(z)
                torch.cuda.synchronize()
                if not all(
                    torch.isfinite(t).all().item() for t in (z, binary, multiclass)
                ):
                    raise RuntimeError("Nonfinite model outputs")
                report["forward"] = {
                    "status": "passed",
                    "embedding": list(z.shape),
                    "binary": list(binary.shape),
                    "multiclass": list(multiclass.shape),
                    "input": "synthetic hardware smoke, not performance evaluation",
                }
        except Exception:  # noqa: BLE001 -- retain full diagnostic traceback and fail closed
            report["forward"] = {
                "status": "failed",
                "traceback": traceback.format_exc(),
            }
            report["errors"].append("Model forward failed")
    report["status"] = "blocked" if report["errors"] else "assets_verified"
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
