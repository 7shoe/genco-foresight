"""Keep the combined status current after either independent controller finishes."""
import argparse
import fcntl
import os
from pathlib import Path
import time

from common import sha256, write_json
from scripts.pilots.run import read
from scripts.recovery.run import status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    with (root / 'status_watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            controllers = [read(root / family / 'controller.json') for family in ('evaluation', 'pilots')]
            # The evaluation controller owns the combined page while it is active.
            if controllers[0].get('status') != 'running':status(root)
            done = all(c.get('status') in ('complete_or_blocked', 'allocation_cap_reached', 'launcher_failed') and
                       all(j.get('finished') for j in c.get('jobs', [])) for c in controllers)
            write_json(root / 'status_watcher.json', dict(pid=os.getpid(), updated_epoch=time.time(),
                       status='finished' if done else 'watching', source_sha256=sha256(__file__)))
            if done:return
            time.sleep(30)


if __name__ == '__main__':main()
