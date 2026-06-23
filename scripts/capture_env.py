"""capture_env.py — Environment snapshot for reproducibility. Self-contained."""
from __future__ import annotations
import argparse, json, platform, sys
from datetime import datetime
from pathlib import Path

def safe_ver(m):
    try:
        return getattr(__import__(m), "__version__", "unknown")
    except ImportError:
        return "NOT_INSTALLED"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("./"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snap = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version.replace("\n", " "),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "libraries": {m: safe_ver(m) for m in
                      ["numpy","scipy","cv2","skimage","pandas",
                       "matplotlib","PIL","numba","phasepack"]},
    }
    try:
        import psutil
        snap["physical_cores"] = psutil.cpu_count(logical=False)
        snap["ram_gb"] = round(psutil.virtual_memory().total/(1024**3),2)
    except ImportError:
        pass
    (args.output/"env_info.json").write_text(json.dumps(snap, indent=2))
    print(json.dumps(snap, indent=2))
    print(f"\n[OK] {args.output/'env_info.json'}")

if __name__ == "__main__":
    main()
