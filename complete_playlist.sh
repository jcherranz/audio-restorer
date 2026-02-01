#!/bin/bash
# Complete processing the remaining playlist videos
# Run this in terminal: bash complete_playlist.sh

cd /home/jcherranz/audio-restorer
source venv/bin/activate

OUTPUT_DIR="output/playlist_fabric_ai_series"
mkdir -p "$OUTPUT_DIR"

echo "=================================="
echo "Completing Playlist Processing"
echo "=================================="
echo "Output: $OUTPUT_DIR"
echo ""

# Remaining videos (32)
VIDEOS=(
  "8HQ1z_03T48" "tdJCt65DzzA" "F6NkmvOzRRQ" "FVHg47HNZms"
  "XgwS61OrMho" "bQFB3rys5Ek" "c8BstYOSqbI" "avVW33ryGck"
  "2lGUjFUVnVg" "JhBdMfHiXJQ" "8LU1vCeBGXg" "AsL1ETc5Xrg"
  "f-Dw2SxrRw8" "CfRAGvXi0r0" "PLe6xdvMRlc" "XCobQujNN_4"
  "yzhE-PWV_sw" "fubju6N5K0I" "oEf2g1z60JA" "5WcF3XiqYLI"
  "0qJIPe53Cg4" "gsl5SM-8iWM" "5iaPCh73qzM" "_ppzfqUe3FI"
  "Lra0Os9pmPo" "y0_4z_uJipg" "E6ZcrQM0u08" "4TJr4L9kPYo"
  "3nwPbS7-Okk" "RwoKZ0wsNa0" "a3W3eYb9jjM" "6QV8FPXBrRw"
)

TOTAL=${#VIDEOS[@]}
COMPLETED=0
FAILED=0

echo "Processing $TOTAL remaining videos..."
echo "This will take approximately $((TOTAL * 2))-$((TOTAL * 3)) minutes"
echo ""

for i in "${!VIDEOS[@]}"; do
  VID="${VIDEOS[$i]}"
  NUM=$((i + 1))
  
  # Skip if already processed
  if [ -f "$OUTPUT_DIR/audio_${VID}_enhanced.wav" ]; then
    echo "[$NUM/$TOTAL] ⏭️  $VID (already done)"
    ((COMPLETED++))
    continue
  fi
  
  echo "[$NUM/$TOTAL] 🎬 Processing $VID..."
  
  if python run.py "https://youtu.be/$VID" --audio-only --enhancer deepfilter --output-dir "$OUTPUT_DIR" --quiet; then
    echo "     ✅ Success"
    ((COMPLETED++))
  else
    echo "     ❌ Failed"
    ((FAILED++))
  fi
  
  # Progress update every 5 videos
  if [ $((NUM % 5)) -eq 0 ]; then
    echo ""
    echo "📈 Progress: $NUM/$TOTAL completed"
    echo ""
  fi
done

echo ""
echo "=================================="
echo "Processing Complete!"
echo "=================================="
echo "✅ Completed: $COMPLETED"
echo "❌ Failed: $FAILED"
echo "📁 Output: $OUTPUT_DIR"
echo ""
echo "Total files:"
ls -1 "$OUTPUT_DIR"/audio_*_enhanced.wav 2>/dev/null | wc -l
echo ""
