#!/usr/bin/env python3
"""Simple runner for 2D stellar system demos.

Usage:
  python run.py            # runs the two-body demo (Sun–Earth)
  python run.py two        # same as above
  python run.py three      # runs the three-body demo (Sun–Earth–Moon)
"""

import argparse

from test import two, three


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple 2D N-body demos (animated)")
    parser.add_argument(
        "scenario",
        nargs="?",
        default="two",
        choices=["two", "three"],
        help="Which demo to run (default: two)",
    )
    args = parser.parse_args()

    if args.scenario == "two":
        two()
    else:
        three()


if __name__ == "__main__":
    main()

