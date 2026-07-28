#!/usr/bin/env python
"""Single entry point for the four calibrated experiments.

Every setting lives in ``configs/protocol/<name>.yaml``. This script only sets up
the environment, resolves the YAML, assembles a command line, and runs it. It
holds no implicit defaults -- whatever the YAML does not say is not supplied here,
so "what was this number run with" always has exactly one source.

    python run_protocol.py --list
    python run_protocol.py llada2_quality --arm ours
    python run_protocol.py llada2_latency --arm both              # serial
    python run_protocol.py sdar_quality  --arm both --bl 8        # other checkpoint pair
    python run_protocol.py sdar_latency  --arm tp2cg              # paper's Vanilla+CG row
    python run_protocol.py llada2_quality --arm ours --dry_run
    python run_protocol.py llada2_quality --arm ours --set num_samples=20 datasets='[gsm8k]'

The four experiments:

    llada2_quality   LLaDA2, EOS on, report pass@1
    llada2_latency   LLaDA2, EOS off + fixed length, report tok/s
    sdar_quality     SDAR,   EOS on, report pass@1 (truncate+argmax, LLaDA2-aligned)
    sdar_latency     SDAR,   EOS off + fixed length, report tok/s (historical protocol)

Design notes:

* ``--arm`` values come from each YAML's ``arms:``; they are not a fixed
  ours/target pair. sdar_latency has four (ours / nat1 / tp2cg / tp2van), matching
  the rows of the paper's Table 1. ``--arm both`` runs ``default_arms``
  SERIALLY, because one 8-GPU node cannot hold two LLaDA2 arms at once.
* Args merge in three layers, later overriding earlier: shared ``args:`` ->
  ``arms.<arm>.args:`` -> command-line ``--set``.
* ``--bl`` always emits both ``block_length`` and ``denoising_steps`` (and
  rescales ``num_blocks`` when ``fixed_tokens`` is set) regardless of what the
  YAML lists. ``ds = bl`` is the controlled protocol -- one position revealed per
  step, so forward count equals ds and does not drift with the data. SDAR
  expresses bl through ``checkpoints:`` and may not list the keys at all; without
  this the runner would fall back to its own default of 4 and silently decode a
  b8 checkpoint at bl=4.
* Flags are validated against the runner's own argparse before the command is
  assembled. sdar_latency drives three different runners, so a shared ``args:``
  key can easily be one that a given runner does not accept -- which would
  otherwise fail only after the model finished loading.
* ``ENV_DEFAULTS`` inlines what used to be ``scripts/env.sh``, so this entry point
  does not depend on ``scripts/``. Each value is applied with ``setdefault``, so
  exporting it in the real environment wins -- no edit needed to move machines.
  ``HF_MODULES_CACHE`` must stay pinned: ``trust_remote_code`` modules are cached
  per repo+revision, and a different directory means a re-fetch, i.e. no longer
  provably the same modeling code.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
CONFIG_DIR = REPO / "configs/protocol"

ENV_DEFAULTS = {
    "SIMSD_ENV": "/mnt/home/haotian.ye/envs/simsd",
    "HF_HUB_CACHE": "/mnt/home/haotian.ye/hf_cache",
    "HF_MODULES_CACHE": "/mnt/home/haotian.ye/hf_modules",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def make_env(verbose: bool = True) -> tuple[dict, str]:
    """Return (env, python) for the child process."""
    env = dict(os.environ)
    for k, v in ENV_DEFAULTS.items():
        env.setdefault(k, v)
    env.pop("PYTHONPATH", None)

    simsd = Path(env["SIMSD_ENV"])
    py = simsd / "bin/python"
    if py.exists():
        env["PATH"] = f"{simsd / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        python = str(py)
    else:
        python = sys.executable
        if verbose:
            print(f"[protocol] warning: {py} not found, falling back to {python}. "
                  f"Point SIMSD_ENV at your prefix.", flush=True)
    if verbose:
        print(f"[protocol] python={python}  HF_HUB_CACHE={env['HF_HUB_CACHE']}", flush=True)
    return env, python


def load(name: str) -> dict:
    p = CONFIG_DIR / f"{name}.yaml"
    if not p.exists():
        avail = sorted(f.stem for f in CONFIG_DIR.glob("*.yaml"))
        raise SystemExit(
            f"[protocol] no such file: {p}\n"
            f"available: {', '.join(avail) or '(configs/protocol/ is empty)'}")
    cfg = yaml.safe_load(p.read_text()) or {}
    cfg["_path"] = str(p.relative_to(REPO))
    return cfg


def resolve_models(cfg: dict, bl: int | None) -> tuple[str, str | None, int]:
    """Return (target, draft, block_length)."""
    if "checkpoints" in cfg:
        ck = {int(k): v for k, v in cfg["checkpoints"].items()}
        if bl is None:
            bl = sorted(ck)[0]
        if bl not in ck:
            raise SystemExit(
                f"[protocol] {cfg['name']} only has checkpoints for bl={sorted(ck)} "
                f"(SDAR trains one pair per block length); got --bl {bl}")
        m = ck[bl]
        return m["target"], m.get("draft"), bl
    m = cfg["models"]
    if bl is None:
        bl = cfg["args"]["block_length"]
    elif (choices := cfg.get("bl_choices")) and bl not in choices:
        # Not an error: one model pair covers every bl, so the weights exist. It
        # just is not one of the swept points, so there is no matching baseline.
        print(f"[protocol] note: bl={bl} is outside {cfg['name']}'s swept set "
              f"{choices}; results will have no matching reference run.", flush=True)
    return m["target"], m.get("draft"), bl


def runner_flags(runner: str) -> set[str]:
    """Long option names the runner's argparse declares.

    Scans the source with a regex rather than importing: runners are __main__
    scripts and importing them has side effects.
    """
    src = (REPO / runner).read_text()
    return set(re.findall(r'add_argument\(\s*"--([a-z0-9_]+)"', src))


def _flatten(args: dict) -> list[str]:
    out: list[str] = []
    for k, v in args.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                out.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            out += [f"--{k}", *[str(x) for x in v]]
        else:
            out += [f"--{k}", str(v)]
    return out


def build(cfg: dict, arm: str, bl: int | None, overrides: dict,
          python: str) -> tuple[list[str], list[int], Path]:
    """Return (argv, gpus, out_dir)."""
    arms = cfg["arms"]
    if arm not in arms:
        raise SystemExit(f"[protocol] {cfg['name']} has no arm={arm}; has {list(arms)}")
    spec = dict(arms[arm])

    target, draft, bl = resolve_models(cfg, bl)

    args = dict(cfg.get("args") or {})
    args.update(spec.pop("args", None) or {})
    args["block_length"] = bl
    args["denoising_steps"] = bl
    if (tok := cfg.get("fixed_tokens")) and "num_blocks" in args:
        if tok % bl:
            raise SystemExit(f"[protocol] fixed_tokens={tok} not divisible by bl={bl}")
        args["num_blocks"] = tok // bl
    args.update(overrides)
    if args["denoising_steps"] != args["block_length"]:
        print(f"[protocol] note: ds={args['denoising_steps']} != bl="
              f"{args['block_length']}, departing from the ds=bl protocol", flush=True)

    runner = spec.pop("runner", cfg.get("runner"))
    if not runner:
        raise SystemExit(f"[protocol] arm={arm} has no runner (nor does the top level)")
    nproc = spec.pop("torchrun", None)

    argv = [python]
    if nproc:
        # Derive the port from the arm name so two arms can run concurrently.
        port = 29500 + (abs(hash(arm)) % 400)
        argv += ["-m", "torch.distributed.run",
                 f"--nproc_per_node={nproc}", f"--master_port={port}"]
    argv.append(runner)

    if m := spec.pop("method", None):
        argv += ["--method", m]
    argv += ["--target_model", target]
    if spec.pop("needs_draft", False):
        if not draft:
            raise SystemExit(f"[protocol] arm={arm} needs a draft model; YAML has none")
        argv += ["--draft_model", draft]

    gpus: list[int] = []
    if (tg := spec.pop("target_gpus", None)) is not None:
        argv += ["--target_gpus", ",".join(str(g) for g in tg)]
        gpus += list(tg)
        if mm := cfg.get("target_max_memory"):
            argv += ["--target_max_memory", str(mm)]
    if (dev := spec.pop("target_device", None)) is not None:
        argv += ["--target_device", dev]
        gpus.append(int(dev.rsplit(":", 1)[1]))
    if (dev := spec.pop("draft_device", None)) is not None:
        argv += ["--draft_device", dev]
        gpus.append(int(dev.rsplit(":", 1)[1]))
    if (n := spec.pop("gpus", None)) is not None:
        gpus += list(range(n)) if isinstance(n, int) else list(n)
    if spec:
        raise SystemExit(f"[protocol] arm={arm} has unknown keys: {sorted(spec)}")

    known = runner_flags(runner)
    if unknown := sorted(set(args) - known):
        raise SystemExit(
            f"[protocol] {cfg['name']} arm={arm}: {runner} does not accept "
            f"{['--' + u for u in unknown]}.\n"
            f"           Move them from the shared args: into the arm that does "
            f"accept them -- or confirm that runner's argparse default is already "
            f"the protocol you want.")

    argv += _flatten(args)
    out = REPO / cfg["output_dir"] / f"{arm}_bl{bl}"
    argv += ["--output_dir", str(out)]
    return argv, sorted(set(gpus)), out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?",
                    help="llada2_quality | llada2_latency | sdar_quality | sdar_latency")
    ap.add_argument("--arm", default="both",
                    help="one of the YAML's arms, or both (= default_arms, serial)")
    ap.add_argument("--bl", type=int, default=None,
                    help="override block_length (ds and num_blocks follow). "
                         "SDAR only accepts values that have a checkpoint pair; "
                         "LLaDA2 accepts any, and warns outside its swept set.")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="override any args key, e.g. --set num_samples=20 datasets='[gsm8k]'")
    ap.add_argument("--dry_run", action="store_true", help="print the command, do not run")
    ap.add_argument("--list", action="store_true", help="list available experiments")
    a = ap.parse_args()

    if a.list or not a.experiment:
        print("available experiments (configs/protocol/):")
        for f in sorted(CONFIG_DIR.glob("*.yaml")):
            c = yaml.safe_load(f.read_text()) or {}
            if "checkpoints" in c:
                bls = sorted(c["checkpoints"])          # hard limit: one pair per bl
            else:
                bls = c.get("bl_choices") or [c.get("args", {}).get("block_length")]
            print(f"  {f.stem:16s} {c.get('family','?'):7s} {c.get('mode','?'):8s} "
                  f"bl={bls}  arms={list(c.get('arms') or {})}")
            if d := c.get("note"):
                print(f"      {d.splitlines()[0]}")
        if not a.experiment:
            return 0

    overrides: dict = {}
    for item in a.set:
        if "=" not in item:
            raise SystemExit(f"[protocol] --set wants KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k] = yaml.safe_load(v)

    cfg = load(a.experiment)
    env, python = make_env()
    arms = cfg.get("default_arms", list(cfg["arms"])) if a.arm == "both" else [a.arm]
    print(f"[protocol] {cfg['name']}  ({cfg['_path']})  arms={arms}", flush=True)

    for arm in arms:
        argv, gpus, out = build(cfg, arm, a.bl, overrides, python)
        e = dict(env)
        # cuda:N inside the runner is a REMAPPED index, so expose the physical
        # ids from the YAML as one set.
        if gpus:
            e["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        print(f"\n[protocol] --- arm={arm}  "
              f"CUDA_VISIBLE_DEVICES={e.get('CUDA_VISIBLE_DEVICES', '(inherited)')}", flush=True)
        print("  " + " ".join(argv[1:]), flush=True)
        if a.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(argv, env=e, cwd=REPO)
        if r.returncode != 0:
            print(f"[protocol] arm={arm} exited {r.returncode}, stopping", flush=True)
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
