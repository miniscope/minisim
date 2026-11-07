"""The ``minisim`` command line: generate simulated recordings on disk.

Currently one subcommand::

    minisim generate --preset miniscope_v4 --region ca1 --duration 60 --seed 0 \
        --motion physical --motion-um 3 --out ./sim_v4_ca1
    minisim generate --spec spec.json --out ./sim_from_spec

``--layout daq`` (default) writes a Miniscope-DAQ output folder with the ground truth
in ``ground_truth/`` beside it (see :mod:`minisim.daq`); ``--layout zarr`` writes
the single-directory :meth:`~minisim.Recording.save` format (simulated in memory).
Only argparse and the standard library are used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from minisim import presets
from minisim.spec import BrainMotion, Spec

SCOPES = {"miniscope_v4": presets.miniscope_v4, "generic_1p": presets.generic_1p}
REGIONS = {"ca1": presets.ca1, "cortex_l23": presets.cortex_l23}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minisim", description="Physically-driven synthetic miniscope data."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser(
        "generate",
        help="simulate a recording and write it to disk",
        description="Simulate a recording (from presets or a Spec JSON) and write it.",
    )
    src = g.add_argument_group("what to simulate (or --spec to load a full Spec)")
    src.add_argument(
        "--spec", type=Path, help="Spec JSON file; overrides the options below"
    )
    src.add_argument(
        "--preset", choices=sorted(SCOPES), default="miniscope_v4", help="scope preset"
    )
    src.add_argument(
        "--region", choices=sorted(REGIONS), default="ca1", help="tissue region preset"
    )
    src.add_argument(
        "--duration", type=float, default=60.0, help="seconds (default 60)"
    )
    src.add_argument("--fps", type=float, default=20.0, help="frame rate (default 20)")
    src.add_argument("--seed", type=int, default=0, help="master RNG seed (default 0)")
    src.add_argument(
        "--motion",
        choices=["none", "physical", "walk"],
        default="none",
        help="brain-motion model (default none)",
    )
    src.add_argument(
        "--motion-um",
        type=float,
        default=None,
        help="physical model: extreme excursion in um (1 px ~ 1.6 um on a V4); "
        "walk model: max shift in um",
    )
    out = g.add_argument_group("output")
    out.add_argument("--out", type=Path, required=True, help="output directory")
    out.add_argument("--layout", choices=["daq", "zarr"], default="daq")
    out.add_argument(
        "--frames-per-file", type=int, default=1000, help="daq: frames per N.avi"
    )
    out.add_argument(
        "--device-name", default="Miniscope", help="daq: device folder name"
    )
    out.add_argument(
        "--jitter-ms", type=float, default=0.0, help="daq: timestamp jitter (sd, ms)"
    )
    out.add_argument(
        "--no-ground-truth", action="store_true", help="daq: skip ground_truth/"
    )
    out.add_argument("--overwrite", action="store_true", help="replace existing output")
    out.add_argument("--quiet", action="store_true", help="no progress output")
    g.set_defaults(func=_generate)
    return parser


def spec_from_args(args: argparse.Namespace) -> Spec:
    """The Spec a ``generate`` invocation describes."""
    if args.spec is not None:
        return Spec.model_validate_json(args.spec.read_text())
    extra = []
    if args.motion != "none":
        kw: dict = {"model": args.motion}
        if args.motion_um is not None:
            if args.motion == "physical":
                kw["motion_amplitude_um"] = args.motion_um
                kw["max_shift_um"] = max(15.0, 1.5 * args.motion_um)
            else:
                kw["max_shift_um"] = args.motion_um
        extra.append(BrainMotion(**kw))
    return presets.build_spec(
        SCOPES[args.preset](),
        REGIONS[args.region](),
        duration_s=args.duration,
        fps=args.fps,
        seed=args.seed,
        extra_steps=extra,
    )


def _generate(args: argparse.Namespace) -> int:
    from minisim import simulate, write_miniscope_daq

    spec = spec_from_args(args)
    if args.layout == "daq":
        folder = write_miniscope_daq(
            spec,
            args.out,
            frames_per_file=args.frames_per_file,
            device_name=args.device_name,
            jitter_ms=args.jitter_ms,
            ground_truth=not args.no_ground_truth,
            progress=not args.quiet,
            overwrite=args.overwrite,
        )
        print(
            f"wrote {folder.n_frames} frames in {len(folder.avi_files)} file(s) to {folder.device_dir}"
        )
    else:
        if args.out.exists() and not args.overwrite:
            print(f"error: {args.out} exists; pass --overwrite", file=sys.stderr)
            return 1
        simulate(spec).save(args.out)
        print(f"wrote recording to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``minisim``."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
