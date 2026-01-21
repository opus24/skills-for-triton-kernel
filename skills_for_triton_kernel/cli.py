"""CLI for skills-for-triton-kernel: install torch-to-triton-kernel skill to ~/.claude/skills/."""

import argparse
import shutil
import sys
from pathlib import Path

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files  # type: ignore

SKILL_NAME = "torch-to-triton-kernel"


def get_skills_source() -> Path:
    """Path to the torch-to-triton-kernel skill directory in the package."""
    try:
        return files("skills_for_triton_kernel") / "skills" / SKILL_NAME  # type: ignore
    except Exception:
        return Path(__file__).resolve().parent / "skills" / SKILL_NAME


def get_default_target() -> Path:
    return Path.home() / ".claude" / "skills"


def _copy_directory(src: Path, dst: Path, force: bool, dry_run: bool) -> int:
    """Recursively copy src to dst. Skips reference.md. Returns files copied count."""
    count = 0
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dst_item = dst / item.name
        if item.name == "reference.md":
            continue
        if item.is_dir():
            count += _copy_directory(item, dst_item, force, dry_run)
        else:
            if dst_item.exists() and not force:
                continue
            if dry_run:
                print(f"[DRY RUN] Would copy: {item} -> {dst_item}")
            else:
                shutil.copy2(item, dst_item)
                print(f"Copied: {dst_item}")
            count += 1
    return count


def install(target: Path | None = None, force: bool = False, dry_run: bool = False) -> int:
    """Copy torch-to-triton-kernel (SKILL.md, examples.md, references/) to target/torch-to-triton-kernel/."""
    target = target or get_default_target()
    src = get_skills_source()
    if not src.is_dir():
        print(f"Source not found: {src}", file=sys.stderr)
        return 1
    dst = target / SKILL_NAME
    n = _copy_directory(src, dst, force=force, dry_run=dry_run)
    if dry_run:
        print(f"[DRY RUN] Would install {n} file(s) to {dst}")
    else:
        print(f"Installed {n} file(s) to {dst}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="skills-for-triton-kernel",
        description="Install torch-to-triton-kernel skill to ~/.claude/skills/",
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    p_install = sub.add_parser("install", help="Install skill to ~/.claude/skills/")
    p_install.add_argument("--target", type=Path, help="Target dir (default: ~/.claude/skills)")
    p_install.add_argument("--force", action="store_true", help="Overwrite existing files")
    p_install.add_argument("--dry-run", action="store_true", help="Show what would be copied")

    sub.add_parser("list", help="List available skills")

    args = parser.parse_args()

    if args.command == "install":
        sys.exit(install(args.target, args.force, args.dry_run))
    if args.command == "list":
        print("torch-to-triton-kernel")
        sys.exit(0)
    # No command: default to install (plan: 인자 없을 때 ~/.claude/skills에 설치)
    sys.exit(install(None, False, False))


if __name__ == "__main__":
    main()
