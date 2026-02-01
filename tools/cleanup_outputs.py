#!/usr/bin/env python3
"""
Output Cleanup Tool
===================

Cleans up old test output files to free disk space.
Keeps reference files and recent outputs.

Usage:
    python tools/cleanup_outputs.py              # Dry run (show what would be deleted)
    python tools/cleanup_outputs.py --execute    # Actually delete files
    python tools/cleanup_outputs.py --days 3     # Keep files newer than 3 days

Default retention:
    - Reference files: Keep forever (reference_*.wav)
    - Other files: Delete if older than 7 days
"""

import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta


def get_file_age_days(file_path: Path) -> float:
    """Get file age in days."""
    stat = file_path.stat()
    age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
    return age.total_seconds() / (24 * 3600)


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


def cleanup_directory(dir_path: Path, dry_run: bool = True, max_age_days: int = 7) -> tuple:
    """
    Clean up a directory of old output files.
    
    Args:
        dir_path: Directory to clean
        dry_run: If True, only show what would be deleted
        max_age_days: Delete files older than this
        
    Returns:
        (files_deleted, space_saved_mb)
    """
    if not dir_path.exists():
        return 0, 0.0
    
    files_deleted = 0
    space_saved_mb = 0.0
    
    for file_path in dir_path.glob("*.wav"):
        # Skip reference files
        if file_path.name.startswith("reference_"):
            continue
        
        age_days = get_file_age_days(file_path)
        size_mb = get_file_size_mb(file_path)
        
        if age_days > max_age_days:
            if dry_run:
                print(f"  [DRY RUN] Would delete: {file_path.name} ({size_mb:.1f} MB, {age_days:.1f} days old)")
            else:
                print(f"  Deleting: {file_path.name} ({size_mb:.1f} MB, {age_days:.1f} days old)")
                try:
                    file_path.unlink()
                    files_deleted += 1
                    space_saved_mb += size_mb
                except Exception as e:
                    print(f"    Error: {e}")
    
    return files_deleted, space_saved_mb


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old test output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show what would be deleted (dry run)
    python tools/cleanup_outputs.py
    
    # Actually delete old files
    python tools/cleanup_outputs.py --execute
    
    # Keep files newer than 3 days
    python tools/cleanup_outputs.py --days 3 --execute
        """
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default is dry run)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Keep files newer than N days (default: 7)"
    )
    
    args = parser.parse_args()
    
    # Project directories
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    benchmarks_dir = project_root / "benchmarks"
    
    mode = "DRY RUN" if not args.execute else "EXECUTING"
    print(f"\n{'='*60}")
    print(f"OUTPUT CLEANUP TOOL - {mode}")
    print(f"{'='*60}")
    print(f"Max age: {args.days} days")
    print(f"Preserving: reference_*.wav files")
    print(f"{'='*60}\n")
    
    total_files = 0
    total_space = 0.0
    
    # Clean output directory
    print(f"Checking: {output_dir}")
    files, space = cleanup_directory(output_dir, dry_run=not args.execute, max_age_days=args.days)
    total_files += files
    total_space += space
    
    # Clean benchmarks directory
    print(f"\nChecking: {benchmarks_dir}")
    files, space = cleanup_directory(benchmarks_dir, dry_run=not args.execute, max_age_days=args.days)
    total_files += files
    total_space += space
    
    # Summary
    print(f"\n{'='*60}")
    if args.execute:
        print(f"CLEANUP COMPLETE")
        print(f"  Files deleted: {total_files}")
        print(f"  Space saved: {total_space:.1f} MB ({total_space/1024:.2f} GB)")
    else:
        print(f"DRY RUN COMPLETE")
        print(f"  Files that would be deleted: {total_files}")
        print(f"  Space that would be saved: {total_space:.1f} MB ({total_space/1024:.2f} GB)")
        print(f"\nRun with --execute to actually delete files")
    print(f"{'='*60}\n")
    
    return 0 if not args.execute or total_files > 0 else 0


if __name__ == "__main__":
    exit(main())
