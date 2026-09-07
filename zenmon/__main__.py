"""zenmon CLI 启动入口与参数解析器."""
import argparse
import curses
import json
import os
import sys
from .app import ZenMonApp, PSU_WATTS_DEFAULT, REFRESH_DEFAULT
from . import __version__

def load_config() -> dict:
    cfg_path = os.path.expanduser("~/.config/zenmon/config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        prog="zenmon",
        description="ZenMon: A serene, zero-dependency telemetry TUI for modern workstations & AI rigs."
    )
    parser.add_argument("-p", "--psu", type=float, default=cfg.get("psu_watts", PSU_WATTS_DEFAULT),
                        help=f"Total PSU power budget in Watts (default: {PSU_WATTS_DEFAULT})")
    parser.add_argument("-r", "--refresh", type=float, default=cfg.get("refresh", REFRESH_DEFAULT),
                        help=f"Telemetry refresh interval in seconds (default: {REFRESH_DEFAULT})")
    parser.add_argument("-w", "--wide", action="store_true", default=cfg.get("wide_mode", False),
                        help="Start in fullscreen wide mode instead of centered card")
    parser.add_argument("-c", "--classic", action="store_true", default=False,
                        help="Start in 1:1 pixel-perfect classic mode")
    parser.add_argument("-v", "--view", type=int, choices=[0, 1, 2, 3, 4, 5], default=cfg.get("default_view", 0),
                        help="Initial view index (0:Main, 1:CPU, 2:GPU, 3:IO, 4:Proc, 5:Wave)")
    parser.add_argument("--snapshot", action="store_true", default=False,
                        help="Capture a headless telemetry snapshot to ~/ and exit immediately")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    app = ZenMonApp(
        psu_watts=args.psu,
        refresh=args.refresh,
        start_classic=args.classic,
        initial_view=args.view,
        compact_mode=not args.wide
    )

    if args.snapshot:
        snap = app.update_telemetry()
        res = app.export_snapshot(snap)
        print(f"✦ {res}")
        sys.exit(0)

    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
