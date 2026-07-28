"""
PTB-XL 서브셋 다운로더 (HSIC 리드그룹 데모 전용)
==================================================
기존 download_ptbxl.py는 전체 21,798레코드를 받고 --limit이 없다.
이 스크립트는 GS-SHAP HSIC 그룹핑 재현(§Appendix D)에 필요한 소량만 받는다.

★안전 설계: 원본 스크립트의 urllib.request.urlretrieve(url, dst) 직접쓰기는
비원자적이다 — 중간에 끊기면 dst가 "존재"하게 되고, 재시도 시 os.path.exists(dst)로
스킵되어 손상 파일이 완료로 오판된다. 여기서는 <dst>.part 로 받은 뒤 os.replace로
원자적 치환한다 — 끊겨도 dst는 절대 반쯤 쓰인 상태로 존재하지 않는다.

산출 경로는 preprocess_ptbxl.py가 기대하는 것과 동일(data/raw/ptbxl) — 이후
표준 전처리 스크립트를 그대로 쓸 수 있다.

용도 다름: 이 파일은 재현/HSIC 데모 전용 부산물이다. P1 GPU 파이프라인
requirements(PyTorch cu118 등)와 무관 — 별도 .venv-hsic-demo에서 실행.
"""
import argparse
import os
import random
import socket
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
ESSENTIAL_FILES = ["ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt", "SHA256SUMS.txt"]
# ★1차 실행에서 5분/15파일로 사실상 멈췄다 — urlretrieve 기본 타임아웃이 무제한이라
#   한 커넥션이 물고 늘어지면 뒤가 전부 밀린다. 명시 타임아웃 + 스레드풀로 교체.
TIMEOUT_SEC = 20


def atomic_download(url, dst, max_retries=4, base_delay=2.0):
    if os.path.exists(dst):
        return "skip"
    part = dst + ".part"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_SEC) as resp, open(part, "wb") as f:
                f.write(resp.read())
            os.replace(part, dst)  # 원자적 — 여기서 끊기면 dst는 존재하지 않는다
            return "ok"
        except Exception as e:
            if os.path.exists(part):
                try:
                    os.remove(part)
                except OSError:
                    pass
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                return f"fail: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw/ptbxl")
    ap.add_argument("--n_records", type=int, default=80,
                     help="받을 레코드 수 (기본 80 — HSIC N_HSIC=3000 시점 확보에 충분)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("[1/3] 메타 파일")
    for fname in ESSENTIAL_FILES:
        r = atomic_download(BASE_URL + fname, os.path.join(out_dir, fname))
        print(f"  {fname}: {r}")

    print("[2/3] RECORDS 인덱스")
    records_index_path = os.path.join(out_dir, "RECORDS")
    r = atomic_download(BASE_URL + "RECORDS", records_index_path)
    print(f"  RECORDS: {r}")

    with open(records_index_path) as f:
        all_records = [line.strip() for line in f if line.strip()]
    hr_records = [r for r in all_records if r.startswith("records500/")]
    print(f"  전체 500Hz 레코드: {len(hr_records)}건")

    # 순차 상위 N개가 아니라 균등 간격 표본 — 특정 환자군·조건에 쏠리지 않게
    random.seed(args.seed)
    step = max(1, len(hr_records) // args.n_records)
    selected = hr_records[::step][: args.n_records]
    print(f"  선택(균등간격, seed={args.seed}): {len(selected)}건")

    print(f"[3/3] 레코드 다운로드 ({len(selected)}건 x .hea/.dat, 병렬 {args.workers})")
    tasks = [(rec, ext) for rec in selected for ext in [".hea", ".dat"]]

    def _one(t):
        rec, ext = t
        dst = os.path.join(out_dir, rec + ext)
        return rec, ext, atomic_download(BASE_URL + rec + ext, dst)

    ok = fail = skip = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_one, t) for t in tasks]
        for fut in as_completed(futures):
            rec, ext, result = fut.result()
            done += 1
            if result == "ok":
                ok += 1
            elif result == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  [실패] {rec}{ext}: {result}")
            if done % 20 == 0:
                print(f"  진행: {done}/{len(tasks)}  ok={ok} skip={skip} fail={fail}", flush=True)

    print(f"\n완료: ok={ok} skip={skip} fail={fail}")
    with open(os.path.join(out_dir, "SUBSET_RECORDS.txt"), "w") as f:
        f.write("\n".join(selected) + "\n")
    print(f"선택된 레코드 목록: {os.path.join(out_dir, 'SUBSET_RECORDS.txt')}")


if __name__ == "__main__":
    main()
