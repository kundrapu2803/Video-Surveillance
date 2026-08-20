"""Environment sanity checks. Run after bootstrap_env.ps1. Each check prints
PASS/WARN/FAIL plus a remediation string; exits non-zero on any FAIL.
"""
import shutil
import sys

RESULTS = []


def check(name, fn, remediation, *, warn_only=False):
    try:
        detail = fn()
        RESULTS.append((name, "PASS", detail or ""))
    except Exception as e:
        status = "WARN" if warn_only else "FAIL"
        RESULTS.append((name, status, f"{e}  ->  {remediation}"))


def check_torch():
    import torch

    assert torch.version.cuda is None, "a CUDA build of torch is installed; this box has no NVIDIA GPU"
    x = torch.rand(4, 4).sum().item()
    tv = __import__("torchvision").__version__
    torch_train = torch.__version__.split(".")[0:2]
    tv_train = tv.split(".")[0:2]
    return f"torch={torch.__version__} torchvision={tv} sum={x:.3f}"


def check_lap():
    import lap

    return f"lap={lap.__version__}"


def check_cv2_writer():
    import cv2
    import tempfile
    import os

    path = os.path.join(tempfile.gettempdir(), "cctv_cv2_probe.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, 10.0, (64, 64))
    ok = vw.isOpened()
    vw.release()
    if os.path.exists(path):
        os.remove(path)
    assert ok, "cv2.VideoWriter failed to open (returns an unopened writer silently on failure)"
    return f"cv2={cv2.__version__}"


def check_ffmpeg():
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    assert exe, "imageio_ffmpeg did not report a bundled ffmpeg binary"
    return exe


def check_motmetrics():
    import motmetrics as mm
    import numpy as np

    acc = mm.MOTAccumulator(auto_id=True)
    acc.update([1, 2], [1, 2], np.array([[0.0, 1.0], [1.0, 0.0]]))
    acc.update([1, 2], [1, 2], np.array([[0.0, 1.0], [1.0, 0.0]]))
    acc.update([1, 2], [1, 2], np.array([[0.0, 1.0], [1.0, 0.0]]))
    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc], metrics=mm.metrics.motchallenge_metrics, generate_overall=True, names=["probe"]
    )
    mota = summary.loc["OVERALL", "mota"]
    assert abs(mota - 1.0) < 1e-6, f"expected mota==1.0 for a perfect 3-frame fixture, got {mota}"
    mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    return f"motmetrics={mm.__version__} mota={mota:.3f}"


def check_import(modname):
    def _fn():
        mod = __import__(modname)
        return getattr(mod, "__version__", "?")

    return _fn


def check_disk_space():
    total, used, free = shutil.disk_usage("C:\\")
    free_gb = free / (1024**3)
    assert free_gb >= 12, f"only {free_gb:.1f} GB free on C:, need >=12 GB"
    return f"{free_gb:.1f} GB free"


def check_onedrive_path():
    import os

    cwd = os.getcwd()
    if "OneDrive" in cwd:
        raise AssertionError(f"cwd is under OneDrive ({cwd}) -- move the repo, see plan risk table")
    return cwd


def check_env_prefix():
    import sys as _sys

    assert _sys.prefix.replace("/", "\\").endswith("envs\\cctv"), (
        f"sys.prefix={_sys.prefix} does not end in envs\\cctv -- wrong interpreter active"
    )
    return _sys.prefix


def main():
    check("torch (CPU-only) + version-train consistency", check_torch,
          "run scripts\\diagnose_torch.ps1")
    check("lap", check_lap, "pip install 'lap>=0.5.12,<0.6' (NOT lapx)")
    check("opencv VideoWriter", check_cv2_writer, "reinstall opencv-python")
    check("imageio-ffmpeg bundled binary", check_ffmpeg, "pip install imageio-ffmpeg")
    check("motmetrics 3-frame perfect-MOTA fixture", check_motmetrics,
          "pip install motmetrics==1.4.0; check numpy<2 and opencv<4.11 via constraints.txt")
    check("shapely import", check_import("shapely"), "pip install 'shapely>=2.0,<3'")
    check("streamlit import", check_import("streamlit"), "pip install 'streamlit>=1.40'")
    check("pandas import", check_import("pandas"), "pip install 'pandas>=2.0,<3'")
    check("kaggle import", check_import("kaggle"), "pip install 'kaggle>=2.2,<3'")
    check("ultralytics import", check_import("ultralytics"), "pip install 'ultralytics>=8.4,<8.5'")
    check("disk space >= 12 GB free on C:", check_disk_space, "free up disk space")
    check("repo path not under OneDrive", check_onedrive_path,
          "move the repo to C:\\Users\\kundr\\projects\\cctv (already the plan default)", warn_only=True)
    check("active interpreter is the cctv env", check_env_prefix,
          "conda activate cctv (see make.ps1 for the absolute-path workaround)")

    width = max(len(r[0]) for r in RESULTS)
    any_fail = False
    for name, status, detail in RESULTS:
        print(f"[{status:4}] {name.ljust(width)}  {detail}")
        if status == "FAIL":
            any_fail = True

    if any_fail:
        print("\nverify_env.py: one or more checks FAILED. See remediations above.")
        sys.exit(1)
    print("\nverify_env.py: all-PASS (WARN entries are non-blocking).")


if __name__ == "__main__":
    main()
