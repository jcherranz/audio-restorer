# Phase 2 Implementation Plan: ML-Powered Enhancement

> **Next immediate steps to improve audio quality**
> 
> **DO NOT START UNTIL:** Testing framework is verified working
> **SUCCESS CRITERIA:** Quality score >80, SNR >20dB, Processing time <5min

## Overview

We will integrate DeepFilterNet and other ML models for significantly better noise suppression than our current spectral gating approach.

### Why DeepFilterNet?
- State-of-the-art neural noise suppression
- Specifically designed for speech
- Open source and well-maintained
- Can run on CPU (no GPU required)
- Real-time capable

---

## Implementation Steps

### Step 1: Install Dependencies

**Task:** Install PyTorch and DeepFilterNet

**Commands:**
```bash
cd audio-restorer
source venv/bin/activate

# Install PyTorch (CPU version for compatibility)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install DeepFilterNet
pip install deepfilternet

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from df import enhance; print('DeepFilterNet: OK')"
```

**Expected Time:** 5-10 minutes  
**Success Criteria:** Imports work without errors

**Document in:** ITERATION_LOG.md

---

### Step 2: Create DeepFilter Enhancer

**Task:** Create new enhancer class in `audio_enhancer.py`

**Code Structure:**
```python
class DeepFilterEnhancer:
    """
    Neural noise suppression using DeepFilterNet.
    
    Usage:
        enhancer = DeepFilterEnhancer()
        enhancer.enhance(input_path, output_path)
    """
    
    def __init__(self, model_type='deepfilternet2'):
        self.model = None
        self.model_type = model_type
        self._load_model()
    
    def _load_model(self):
        # Load DeepFilterNet model
        pass
    
    def enhance(self, input_path: Path, output_path: Path) -> Path:
        # Process audio through DeepFilterNet
        pass
```

**Integration Points:**
- Add to `audio_enhancer.py`
- Make it optional in config (fallback to SimpleEnhancer)
- Handle errors gracefully (fall back if model fails)

**Expected Time:** 1-2 hours  
**Success Criteria:** Successfully enhances reference video

**Document in:** ITERATION_LOG.md with:
- Code added
- Test results
- Any issues encountered

---

### Step 3: Integrate with Pipeline

**Task:** Modify pipeline to use DeepFilter when available

**Changes to `pipeline.py`:**
```python
# In AudioRestorationPipeline.__init__:
try:
    from .audio_enhancer import DeepFilterEnhancer
    self.enhancer = DeepFilterEnhancer()
    print("Using DeepFilterNet for enhancement")
except ImportError:
    self.enhancer = SimpleEnhancer()
    print("DeepFilterNet not available, using ffmpeg filters")
```

**Configuration in `config.py`:**
```python
ENHANCEMENT = {
    "use_deepfilternet": True,  # Try to use ML model
    "deepfilternet_model": "deepfilternet2",  # Model variant
    "fallback_to_simple": True,  # Use simple if ML fails
}
```

**Expected Time:** 30 minutes  
**Success Criteria:** Pipeline automatically uses best available enhancer

**Document in:** ITERATION_LOG.md

---

### Step 4: Test and Measure Quality

**Task:** Run full test with quality metrics

**Commands:**
```bash
# Run pipeline with DeepFilter
python run.py "https://youtu.be/cglDoG0GzyA" --audio-only --keep-temp

# Measure quality
python tests/measure_quality.py output/audio_cglDoG0GzyA_enhanced.wav > benchmarks/phase2_result.txt

# Run benchmark suite
python tests/test_benchmark.py --full
```

**Expected Results:**
- Quality score: >80 (was ~65 with simple enhancer)
- SNR: >20 dB (was ~15 dB)
- Processing time: <5 minutes (acceptable trade-off for quality)

**Document in:** ITERATION_LOG.md with full metrics table

---

### Step 5: Compare with Previous

**Task:** Direct before/after comparison

**Commands:**
```bash
# Process with old method
python run.py "https://youtu.be/cglDoG0GzyA" --audio-only --quick -o "old_method"

# Process with new method (DeepFilter)
python run.py "https://youtu.be/cglDoG0GzyA" --audio-only -o "new_method"

# Compare
python tests/measure_quality.py --compare \
    output/old_method_enhanced.wav \
    output/new_method_enhanced.wav
```

**Expected Output:**
```
Comparison showing:
- SNR improvement: +5 to +10 dB
- Quality score improvement: +15 to +20 points
- Processing time increase: +2-3 minutes (acceptable)
```

**Document in:** ITERATION_LOG.md and ROADMAP.md (mark Phase 2 complete)

---

## Testing Protocol

### Every Code Change Must:

1. **Test imports work:**
   ```bash
   python -c "from src.audio_enhancer import DeepFilterEnhancer"
   ```

2. **Test with reference video:**
   ```bash
   python run.py "https://youtu.be/cglDoG0GzyA" --audio-only
   ```

3. **Measure quality:**
   ```bash
   python tests/measure_quality.py output/audio_cglDoG0GzyA_enhanced.wav
   ```

4. **Verify no regressions:**
   - SimpleEnhancer still works as fallback
   - CLI interface unchanged
   - Config works as expected

### Quality Gates

| Gate | Metric | Minimum | Target |
|------|--------|---------|--------|
| 1 | Import Test | Must pass | Pass |
| 2 | Processing Test | Must complete | <5min |
| 3 | SNR | >18 dB | >20 dB |
| 4 | Quality Score | >75 | >80 |
| 5 | No Artifacts | Subjective | Clean |

---

## Fallback Strategy

If DeepFilterNet fails or is too slow:

1. **Keep SimpleEnhancer as fallback** (already done)
2. **Try lighter models:**
   - DeepFilterNet Lite (if available)
   - RNNoise (lighter alternative)
   - WebRTC NS (very light, lower quality)
3. **Optimize current spectral gating** with better parameters

---

## Potential Issues & Solutions

### Issue 1: PyTorch Installation Too Large
**Problem:** PyTorch is 500MB+, might be slow to install  
**Solution:** Use CPU-only version, document for user

### Issue 2: DeepFilterNet Model Download Fails
**Problem:** Models download on first run, might fail offline  
**Solution:** Pre-download models, include in repo or document download

### Issue 3: Processing Too Slow
**Problem:** ML processing takes >10 minutes  
**Solution:** 
- Add progress indicator
- Make it optional (default to quick mode)
- Add "quality vs speed" option

### Issue 4: Quality Not Improved
**Problem:** ML doesn't improve over simple method  
**Solution:**
- Check model is loading correctly
- Try different model variant
- Fall back to optimized simple method

---

## Success Criteria Checklist

Phase 2 is complete when:

- [ ] PyTorch installed and working
- [ ] DeepFilterNet installed and working
- [ ] DeepFilterEnhancer class created
- [ ] Pipeline integrates ML enhancement
- [ ] Reference video processed successfully
- [ ] Quality score >80 (was ~65)
- [ ] SNR >20 dB (was ~15 dB)
- [ ] Processing time <5 minutes
- [ ] Fallback to SimpleEnhancer works
- [ ] All tests pass
- [ ] ITERATION_LOG.md updated
- [ ] ROADMAP.md updated (Phase 2 marked complete)

---

## Time Estimate

| Step | Estimated Time |
|------|---------------|
| 1. Install dependencies | 10 min |
| 2. Create DeepFilterEnhancer | 2 hours |
| 3. Integrate with pipeline | 30 min |
| 4. Test and measure | 30 min |
| 5. Compare and document | 30 min |
| **Total** | **~4 hours** |

---

## Next Steps After Phase 2

Once Phase 2 is complete (ML enhancement working):

1. **Phase 3: Speaker Enhancement**
   - Add speaker diarization
   - Isolate main speaker
   - Reduce crowd noise

2. **Phase 4: Echo Removal**
   - Dereverberation
   - Room mode correction

3. **Phase 5: Polish**
   - Optimization
   - Batch processing
   - Presets for different scenarios

---

**DO NOT PROCEED TO PHASE 3 UNTIL:**
- Phase 2 quality score >80
- All success criteria met
- Documentation updated

---

**Ready to start Phase 2?**

Begin with Step 1: Install dependencies, then update ITERATION_LOG.md.

**Last Updated:** 2026-01-31  
**Status:** Ready to begin  
**Priority:** HIGH
