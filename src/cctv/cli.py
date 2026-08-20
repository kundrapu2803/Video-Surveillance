"""Full CLI surface. Every flag defaults to None so cctv.config decides
precedence (CLI > --config YAML > per-source profile > default.yaml > builtin).
"""
from __future__ import annotations

import argparse
import logging
import sys

EPILOG = """
Precedence (lowest to highest): builtin defaults < configs/default.yaml <
per-source profile (configs/sources/*.yaml) < --config YAML < CLI flags.
An explicit per-zone rules[...] key in zones.json always beats a CLI flag
unless the zone config's zone_class preset is itself overridden by --force-override.

Example:
  python run.py --video input.mp4 --zones zones.json --output results/
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Person detection, tracking, and zone-based event detection over CCTV footage.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g_in = p.add_argument_group("input")
    g_in.add_argument("--video", dest="video", default=None, help="video file or a directory of frames (also accepts --source)")
    g_in.add_argument("--source", dest="video", default=None, help=argparse.SUPPRESS)
    g_in.add_argument("--fps", type=float, default=None)
    g_in.add_argument("--stride", type=int, default=None)
    g_in.add_argument("--max-frames", type=int, default=None)
    g_in.add_argument("--allow-lowres", action="store_true", default=None)

    g_zone = p.add_argument_group("zones")
    g_zone.add_argument("--zones", default=None, help="zones.json path")
    g_zone.add_argument("--zone-autoscale", action="store_true", default=None)
    g_zone.add_argument("--allow-moving-camera-zones", action="store_true", default=None)

    g_model = p.add_argument_group("model")
    g_model.add_argument("--detector", choices=["yolo", "stub"], default=None)
    g_model.add_argument("--model", default=None)
    g_model.add_argument("--backend", choices=["pytorch"], default=None)
    g_model.add_argument("--device", default=None)
    g_model.add_argument("--imgsz", type=int, default=None)
    g_model.add_argument("--iou", dest="iou_nms", type=float, default=None)
    g_model.add_argument("--threads", type=int, default=None)
    g_model.add_argument("--half", action="store_true", default=None)

    g_track = p.add_argument_group("tracker")
    g_track.add_argument("--tracker", choices=["bytetrack", "botsort_reid"], default=None,
                          help="botsort_reid has with_reid: true -- tracker-internal appearance "
                               "matching for surviving short-to-medium occlusion")
    g_track.add_argument("--tracker-set", action="append", default=None,
                          help="K=V override for the tracker yaml, repeatable")

    g_rules = p.add_argument_group("rules")
    g_rules.add_argument("--loiter-seconds", type=float, default=None)
    g_rules.add_argument("--enter-seconds", type=float, default=None)
    g_rules.add_argument("--exit-seconds", type=float, default=None)
    g_rules.add_argument("--min-confidence", type=float, default=None)
    g_rules.add_argument("--cooldown-seconds", type=float, default=None)
    g_rules.add_argument("--merge-gap-seconds", type=float, default=None)
    g_rules.add_argument("--max-events-per-minute", type=int, default=None)

    g_out = p.add_argument_group("output")
    g_out.add_argument("--output", required=True)
    g_out.add_argument("--run-name", default=None)
    g_out.add_argument("--no-run-subdir", action="store_true", default=None)
    g_out.add_argument("--overwrite", action="store_true", default=None)
    g_out.add_argument("--no-video", action="store_true", default=None)
    g_out.add_argument("--stage-videos", action="store_true", default=None,
                        help="also write detection.mp4 (boxes only), tracking.mp4 (boxes+ids+trails, no "
                             "zones), intrusion.mp4 and loitering.mp4 (zone overlay filtered to that rule, "
                             "with an on-screen indicator when the event is active) alongside annotated.mp4 "
                             "(the full combined view)")
    g_out.add_argument("--dets-only", action="store_true", default=None)
    g_out.add_argument("--save-mot", action="store_true", default=None)
    g_out.add_argument("--output-format", choices=["mp4", "frames", "none"], default=None)

    g_meta = p.add_argument_group("meta")
    g_meta.add_argument("--config", default=None, help="--config YAML path")
    g_meta.add_argument("--seed", type=int, default=None)
    g_meta.add_argument("--dry-run", action="store_true", default=None)
    g_meta.add_argument("-v", "--verbose", action="store_true", default=None)
    g_meta.add_argument("-q", "--quiet", action="store_true", default=None)
    g_meta.add_argument("--version", action="version", version="cctv 0.1.0")

    return p


EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_SOURCE = 3
EXIT_MODEL = 4
EXIT_WRITER = 5
EXIT_INTERRUPT = 130


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else (logging.ERROR if args.quiet else logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.video:
        parser.error("--video (or --source) is required")

    from cctv.config import resolve_effective_config

    cli_dict = {k: v for k, v in vars(args).items() if k not in ("video", "zones", "output", "config")}
    if args.tracker_set:
        overrides = {}
        for kv in args.tracker_set:
            k, _, v = kv.partition("=")
            try:
                v = float(v) if "." in v else int(v)
            except ValueError:
                v = {"true": True, "false": False}.get(v.lower(), v)
            overrides[k] = v
        cli_dict["tracker_set"] = overrides

    try:
        eff = resolve_effective_config(cli_dict, config_path=args.config)
    except Exception as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG

    if args.dry_run:
        print("effective config:")
        for k, v in sorted(eff.as_dict().items()):
            print(f"  {k} = {v!r}")
        return EXIT_OK

    from cctv.io.frame_source import LowResolutionSourceError
    from cctv.io.zones import MovingCameraZoneError
    from cctv.pipeline import run_source

    try:
        run_dir = run_source(
            video_path=args.video, zones_path=args.zones, output_dir=args.output,
            eff=eff, run_name=args.run_name, argv=argv,
        )
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPT
    except (FileNotFoundError, LowResolutionSourceError, IOError) as e:
        print(f"source error: {e}", file=sys.stderr)
        return EXIT_SOURCE
    except MovingCameraZoneError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    except OSError as e:
        if "WinError 1114" in str(e) or "c10.dll" in str(e):
            print(
                f"model load error: {e}\n"
                "This looks like the known torch/c10.dll load failure -- "
                "run scripts\\diagnose_torch.ps1.",
                file=sys.stderr,
            )
            return EXIT_MODEL
        print(f"error: {e}", file=sys.stderr)
        return EXIT_WRITER

    print(f"\nrun complete: {run_dir.resolve()}")
    print(f"  events:     {(run_dir / 'cameras' / 'cam01' / 'events.csv').resolve()}")
    print(f"  video:      {(run_dir / 'cameras' / 'cam01' / 'annotated.mp4').resolve()}")
    print(f"  run.json:   {(run_dir / 'run.json').resolve()}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
