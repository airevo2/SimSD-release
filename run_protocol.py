#!/usr/bin/env python
"""四个标定实验的单一入口。

设定全部写在 ``configs/protocol/<name>.yaml`` 里，本脚本只做四件事：装好环境、
解析 YAML、拼命令行、执行。没有隐含默认值 —— YAML 里没写的东西这里不会替它补，
这样「跑的是什么」永远只有一个出处。

    python run_protocol.py --list
    python run_protocol.py llada2_quality --arm ours
    python run_protocol.py llada2_latency --arm both              # 串行跑两个 arm
    python run_protocol.py sdar_quality  --arm both --bl 8        # SDAR 换 checkpoint 对
    python run_protocol.py sdar_latency  --arm ours               # 全 kernel 优化栈
    python run_protocol.py sdar_latency  --arm tp2cg              # 论文的 Vanilla+CG 行
    python run_protocol.py llada2_quality --arm ours --dry_run
    python run_protocol.py llada2_quality --arm ours --set num_samples=20 datasets='[gsm8k]'

四个实验：

    llada2_quality   LLaDA2 开 EOS，报 pass@1
    llada2_latency   LLaDA2 关 EOS 定长，报 tok/s
    sdar_quality     SDAR   开 EOS，报 pass@1（truncate+argmax，与 LLaDA2 对齐）
    sdar_latency     SDAR   关 EOS 定长，报 tok/s（历史口径 + 全 kernel 优化栈）

``--arm`` 的取值由每个 YAML 的 ``arms:`` 决定，不是固定的 ours/target：
sdar_latency 有 ours / nat1 / tp2cg / tp2van 四个（论文 Table 1 的那几行）。
``--arm both`` = 该 YAML 的 ``default_arms``，**串行**执行（一个 8 卡节点放不下
两个 LLaDA2 arm）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
CONFIG_DIR = REPO / "configs/protocol"

# ── 环境（原 scripts/env.sh，内联进来让入口自足）────────────────────────────
# 独立 prefix，绝不与 ~/miniconda3/envs/* 混淆。每一项都可以被真实环境变量覆盖，
# 所以换机器时 export 一下就行，不需要改这个文件。
ENV_DEFAULTS = {
    "SIMSD_ENV": "/mnt/home/haotian.ye/envs/simsd",
    "HF_HUB_CACHE": "/mnt/home/haotian.ye/hf_cache",        # 权重
    "HF_MODULES_CACHE": "/mnt/home/haotian.ye/hf_modules",  # trust_remote_code 的动态模块
    "PYTHONNOUSERSITE": "1",                                # 不吃 ~/.local
    "TOKENIZERS_PARALLELISM": "false",
}


def make_env(verbose: bool = True) -> tuple[dict, str]:
    """(env, python)。返回子进程环境和要用的解释器路径。

    ``HF_MODULES_CACHE`` 必须固定：``trust_remote_code`` 的动态模块按 repo+revision
    落盘，换目录会重新拉一份，那就不是同一份 modeling 代码了。
    """
    env = dict(os.environ)
    for k, v in ENV_DEFAULTS.items():
        env.setdefault(k, v)
    env.pop("PYTHONPATH", None)     # 原 env.sh 的 `unset PYTHONPATH`

    simsd = Path(env["SIMSD_ENV"])
    py = simsd / "bin/python"
    if py.exists():
        env["PATH"] = f"{simsd / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        python = str(py)
    else:
        python = sys.executable
        if verbose:
            print(f"[protocol] 警告：{py} 不存在，回退到当前解释器 {python}。"
                  f"换机器时用 SIMSD_ENV=<prefix> 指过去。", flush=True)
    if verbose:
        print(f"[protocol] python={python}  HF_HUB_CACHE={env['HF_HUB_CACHE']}", flush=True)
    return env, python


def load(name: str) -> dict:
    p = CONFIG_DIR / f"{name}.yaml"
    if not p.exists():
        avail = sorted(f.stem for f in CONFIG_DIR.glob("*.yaml"))
        raise SystemExit(
            f"[protocol] 没有 {p}\n可用实验：{', '.join(avail) or '（configs/protocol/ 是空的）'}")
    cfg = yaml.safe_load(p.read_text()) or {}
    cfg["_path"] = str(p.relative_to(REPO))
    return cfg


def resolve_models(cfg: dict, bl: int | None) -> tuple[str, str | None, int]:
    """(target, draft, block_length)。

    SDAR 每个 block length 有独立的 checkpoint 对（按 bl 分别训练），所以用
    ``checkpoints:`` 按 bl 索引；LLaDA2 是同一对模型换参数，用 ``models:``。
    """
    if "checkpoints" in cfg:
        ck = {int(k): v for k, v in cfg["checkpoints"].items()}
        if bl is None:
            bl = sorted(ck)[0]
        if bl not in ck:
            raise SystemExit(
                f"[protocol] {cfg['name']} 只有 bl={sorted(ck)} 的 checkpoint"
                f"（SDAR 按 block length 分别训练），给了 --bl {bl}")
        m = ck[bl]
        return m["target"], m.get("draft"), bl
    m = cfg["models"]
    return m["target"], m.get("draft"), bl if bl is not None else cfg["args"]["block_length"]


def runner_flags(runner: str) -> set[str]:
    """Runner 的 argparse 里声明过的长选项名（不含 ``--``）。

    用来在**拼命令之前**就发现「这个 arm 的 runner 不接受这个 flag」。
    sdar_latency 有四个 arm 走三个不同 runner，公共 ``args:`` 很容易塞进某个
    runner 不认识的键 —— 那会在模型加载完之后才 argparse 报错，白等几分钟。
    正则扫源码而不是 import：runner 是 __main__ 脚本，import 会有副作用。
    """
    import re
    src = (REPO / runner).read_text()
    return set(re.findall(r'add_argument\(\s*"--([a-z0-9_]+)"', src))


def _flatten(args: dict) -> list[str]:
    out: list[str] = []
    for k, v in args.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:                                   # store_true：只在 True 时给
                out.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            out += [f"--{k}", *[str(x) for x in v]]
        else:
            out += [f"--{k}", str(v)]
    return out


def build(cfg: dict, arm: str, bl: int | None, overrides: dict,
          python: str) -> tuple[list[str], list[int], Path]:
    """(argv, gpus, out_dir)。"""
    arms = cfg["arms"]
    if arm not in arms:
        raise SystemExit(f"[protocol] {cfg['name']} 没有 arm={arm}，有 {list(arms)}")
    spec = dict(arms[arm])

    target, draft, bl = resolve_models(cfg, bl)

    # 参数三层合并：公共 args -> 该 arm 的 args -> 命令行 --set
    args = dict(cfg.get("args") or {})
    args.update(spec.pop("args", None) or {})
    # block_length / denoising_steps 总是显式下发（不看 YAML 里有没有写）：
    # ds = bl 是「每步揭 1 个位置」的受控口径（前向次数 = ds，不随数据变），
    # 只改一个会静默破坏这个不变量。而 SDAR 用 checkpoints: 表达 bl，args 里
    # 可能根本没有这两个键 —— 那时如果不补，runner 会用它自己的默认值 4，
    # 于是「加载 b8 checkpoint 却按 bl=4 解码」，静默错配。
    args["block_length"] = bl
    args["denoising_steps"] = bl
    # num_blocks 型的 runner（run_ours_dual_gpu / *_tp2_cache）用块数而不是
    # gen_length 表达定长预算，所以 bl 变了它要跟着重算。
    if (tok := cfg.get("fixed_tokens")) and "num_blocks" in args:
        if tok % bl:
            raise SystemExit(f"[protocol] fixed_tokens={tok} 不能被 bl={bl} 整除")
        args["num_blocks"] = tok // bl
    args.update(overrides)

    runner = spec.pop("runner", cfg.get("runner"))
    if not runner:
        raise SystemExit(f"[protocol] arm={arm} 没有 runner（YAML 顶层也没有）")
    nproc = spec.pop("torchrun", None)

    argv = [python]
    if nproc:
        # torchrun 起 TP：master_port 由 arm 名派生，避免并发跑两个 arm 时撞端口
        port = 29500 + (abs(hash(arm)) % 400)
        argv += ["-m", "torch.distributed.run",
                 f"--nproc_per_node={nproc}", f"--master_port={port}"]
    argv.append(runner)

    if m := spec.pop("method", None):
        argv += ["--method", m]
    argv += ["--target_model", target]
    if spec.pop("needs_draft", False):
        if not draft:
            raise SystemExit(f"[protocol] arm={arm} 需要 draft 模型，YAML 里没有")
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
        raise SystemExit(f"[protocol] arm={arm} 里有不认识的键：{sorted(spec)}")

    # 在拼命令前校验，而不是等 runner 加载完模型才 argparse 报错。
    known = runner_flags(runner)
    if unknown := sorted(set(args) - known):
        raise SystemExit(
            f"[protocol] {cfg['name']} arm={arm}: {runner} 不接受 "
            f"{['--' + u for u in unknown]}。\n"
            f"           把它们从公共 args: 挪到接受它们的那个 arm 的 args: 下面 —— "
            f"或者确认该 runner 的 argparse 默认值就是你想要的口径。")

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
                    help="YAML 的 arms 之一，或 both（= default_arms，串行）")
    ap.add_argument("--bl", type=int, default=None,
                    help="覆盖 block_length（ds 与 num_blocks 跟着变）。"
                         "SDAR 只能取有 checkpoint 的值")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="覆盖 args 里的任意键，如 --set num_samples=20 datasets='[gsm8k]'")
    ap.add_argument("--dry_run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--list", action="store_true", help="列出可用实验")
    a = ap.parse_args()

    if a.list or not a.experiment:
        print("可用实验（configs/protocol/）：")
        for f in sorted(CONFIG_DIR.glob("*.yaml")):
            c = yaml.safe_load(f.read_text()) or {}
            bls = sorted(c["checkpoints"]) if "checkpoints" in c else \
                [c.get("args", {}).get("block_length")]
            print(f"  {f.stem:16s} {c.get('family','?'):7s} {c.get('mode','?'):8s} "
                  f"bl={bls}  arms={list(c.get('arms') or {})}")
            if d := c.get("note"):
                print(f"      {d.splitlines()[0]}")
        if not a.experiment:
            return 0

    overrides: dict = {}
    for item in a.set:
        if "=" not in item:
            raise SystemExit(f"[protocol] --set 要 KEY=VALUE，得到 {item!r}")
        k, v = item.split("=", 1)
        overrides[k] = yaml.safe_load(v)          # "20" -> 20、"true" -> True

    cfg = load(a.experiment)
    env, python = make_env()
    arms = cfg.get("default_arms", list(cfg["arms"])) if a.arm == "both" else [a.arm]
    print(f"[protocol] {cfg['name']}  ({cfg['_path']})  arms={arms}", flush=True)

    for arm in arms:
        argv, gpus, out = build(cfg, arm, a.bl, overrides, python)
        e = dict(env)
        # cuda:N 在 runner 里是**重映射后**的索引，所以按 YAML 里的物理编号整体暴露
        if gpus:
            e["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        print(f"\n[protocol] --- arm={arm}  "
              f"CUDA_VISIBLE_DEVICES={e.get('CUDA_VISIBLE_DEVICES', '(继承)')}", flush=True)
        print("  " + " ".join(argv[1:]), flush=True)
        if a.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(argv, env=e, cwd=REPO)
        if r.returncode != 0:
            print(f"[protocol] arm={arm} 退出码 {r.returncode}，中止", flush=True)
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
