"""Minimal CLI: the Observatory refuses unsupported operations."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relate",
        description="RELATE -- relation-aware embedding runtime",
    )
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="run a tiny synthetic observatory demo")
    sub.add_parser("spaces", help="print space-identity help")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from importlib.metadata import version

        try:
            print(version("relate-search"))
        except Exception:
            print("0.1.0")
        return 0
    if args.command == "demo":
        import numpy as np

        from relate import Observatory

        runtime = Observatory()
        space = runtime.register_space(model="demo", dimensions=8)
        print(f"space: {space.short}")
        vectors = runtime.attach(np.random.default_rng(0).normal(size=(16, 8)), space=space)
        print(runtime.inspect(vectors))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
