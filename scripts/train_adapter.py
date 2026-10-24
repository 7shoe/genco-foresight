"""Entrypoint for cached, resumable A training on CPU or one Aurora XPU."""
import argparse
from pathlib import Path

from common import load_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=["cpu","xpu"],default="xpu")
    p.add_argument("--cache-only",action="store_true")
    p.add_argument("--run-root",type=Path)
    p.add_argument("--stop-after-steps",type=int)
    p.add_argument("--crash-at-steps",type=int)
    p.add_argument("--resume-test",action="store_true")
    a=p.parse_args();cfg=load_config()
    if a.cache_only:
        from training.inputs import build_cache
        print(build_cache(cfg,a.device));return
    from training.loop import run
    run(cfg,a.device,a.run_root,a.stop_after_steps,a.crash_at_steps,a.resume_test)


if __name__=="__main__":main()
