#!/usr/bin/env python3
"""
Batch process YouTube playlist with audio restoration
Resumable - skips already processed videos
"""

import subprocess
import sys
import time
from pathlib import Path

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLh5w5JaGwAyJpvAWw1vIjScKN-nES0AbM"
OUTPUT_DIR = "output/playlist_fabric_ai_series"

def get_video_list():
    """Get list of video IDs from playlist"""
    print("📋 Fetching playlist videos...")
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s", PLAYLIST_URL],
        capture_output=True, text=True
    )
    return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]

def get_video_title(video_id):
    """Get video title"""
    result = subprocess.run(
        ["yt-dlp", "--print", "%(title)s", f"https://youtu.be/{video_id}"],
        capture_output=True, text=True
    )
    return result.stdout.strip()[:60]

def is_processed(video_id):
    """Check if video already processed"""
    output_file = Path(OUTPUT_DIR) / f"audio_{video_id}_enhanced.wav"
    return output_file.exists()

def process_video(video_id, index, total):
    """Process a single video"""
    url = f"https://youtu.be/{video_id}"
    title = get_video_title(video_id)
    
    print(f"\n[{index}/{total}] 🎬 {title}")
    print(f"       ID: {video_id}")
    
    start = time.time()
    try:
        result = subprocess.run(
            ["python", "run.py", url, 
             "--audio-only", 
             "--enhancer", "deepfilter",
             "--output-dir", OUTPUT_DIR,
             "--quiet"],
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout per video
        )
        
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"       ✅ Success ({elapsed:.1f}s)")
            return True, elapsed
        else:
            print(f"       ❌ Failed")
            return False, elapsed
            
    except subprocess.TimeoutExpired:
        print(f"       ⏱️  Timeout (>10min)")
        return False, 600
    except Exception as e:
        print(f"       ❌ Error: {e}")
        return False, 0

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    video_ids = get_video_list()
    total = len(video_ids)
    
    print("=" * 70)
    print("🎙️  BATCH AUDIO RESTORATION - Filosofía Moderna 2")
    print("=" * 70)
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"🧠 Enhancer: DeepFilterNet")
    print(f"📊 Total videos: {total}")
    
    # Check already processed
    already_done = sum(1 for vid in video_ids if is_processed(vid))
    print(f"✅ Already processed: {already_done}")
    print(f"⏳ Remaining: {total - already_done}")
    print("=" * 70)
    
    if already_done == total:
        print("\n🎉 All videos already processed!")
        return
    
    results = []
    start_time = time.time()
    
    for i, vid in enumerate(video_ids, 1):
        if is_processed(vid):
            results.append((vid, "SKIPPED", 0))
            continue
            
        success, elapsed = process_video(vid, i, total)
        status = "SUCCESS" if success else "FAILED"
        results.append((vid, status, elapsed))
        
        # Progress update every 5 videos
        if i % 5 == 0:
            elapsed_total = time.time() - start_time
            progress = sum(1 for _, s, _ in results if s in ("SUCCESS", "SKIPPED"))
            print(f"\n📈 Progress: {progress}/{total} ({progress/total*100:.1f}%)")
            print(f"⏱️  Time elapsed: {elapsed_total/60:.1f} min")
    
    # Final summary
    total_time = time.time() - start_time
    success_count = sum(1 for _, s, _ in results if s == "SUCCESS")
    skipped_count = sum(1 for _, s, _ in results if s == "SKIPPED")
    
    print("\n" + "=" * 70)
    print("📊 BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✅ Successful: {success_count}")
    print(f"⏭️  Skipped (already done): {skipped_count}")
    print(f"❌ Failed: {total - success_count - skipped_count}")
    print(f"⏱️  This session: {total_time/60:.1f} minutes")
    print(f"📁 Output: {OUTPUT_DIR}/")
    print(f"\nEnhanced files:")
    
    output_files = sorted(Path(OUTPUT_DIR).glob("audio_*_enhanced.wav"))
    for f in output_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  • {f.name} ({size_mb:.1f} MB)")
    
    # Save summary
    with open(f"{OUTPUT_DIR}/processing_summary.txt", "w") as f:
        f.write("Filosofía Moderna 2 - Processing Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total videos: {total}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Skipped: {skipped_count}\n")
        f.write(f"Failed: {total - success_count - skipped_count}\n")
        f.write(f"Session time: {total_time/60:.1f} minutes\n\n")
        f.write("Files:\n")
        for wav in output_files:
            f.write(f"  {wav.name}\n")
    
    print(f"\n📝 Summary saved to: {OUTPUT_DIR}/processing_summary.txt")

if __name__ == "__main__":
    main()
