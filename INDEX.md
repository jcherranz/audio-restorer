# Project Index - START HERE

> **Welcome! This is your navigation hub for the Audio Restoration Project.**

## 🎯 Quick Navigation

### For Users (You)
| Document | Purpose |
|----------|---------|
| **QUICKSTART.md** | How to use the tool right now |
| **README.md** | Full user documentation |
| **run.py** | The actual tool - run this! |

### For LLM Agents (Development)
| Document | Purpose |
|----------|---------|
| **AGENTS.md** | ⚠️ READ FIRST - Rules and context |
| **ITERATION_LOG.md** | History of all changes made |
| **ROADMAP.md** | Current priorities and phases |
| **PHASE2_PLAN.md** | Next implementation steps |
| **QUALITY_METRICS.md** | How we measure success |

### For Testing
| Document | Purpose |
|----------|---------|
| **tests/measure_quality.py** | Measure audio quality |
| **tests/test_pipeline.py** | Run integration tests |
| **tests/test_benchmark.py** | Benchmark different settings |

---

## 📊 Current Project Status

### What's Working Now ✅
- YouTube audio downloading
- Basic noise reduction (spectral gating)
- Audio normalization
- Quality measurement tools
- Full test suite

### Current Quality (Iteration 0)
**Reference Video:** https://youtu.be/cglDoG0GzyA

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Quality Score | 61.7 | >80 | ⚠️ Needs improvement |
| SNR | 19.9 dB | >25 dB | ⚠️ Needs improvement |
| Processing | 69s | <5min | ✅ Good |

### Current Phase
**Phase 2: ML-Powered Enhancement** (Ready to start)

---

## 🚀 Immediate Next Steps

### Option 1: Use Current Tool (Right Now)
```bash
cd audio-restorer
source venv/bin/activate
python run.py "https://youtu.be/YOUR_VIDEO" --audio-only --quick
```

### Option 2: Improve Audio Quality (Next Iteration)
1. Read **PHASE2_PLAN.md**
2. Install PyTorch + DeepFilterNet
3. Integrate ML enhancement
4. Test with reference video
5. Update **ITERATION_LOG.md**

### Option 3: Run Tests
```bash
# Measure quality of any audio file
python tests/measure_quality.py output/audio.wav

# Run full test suite
python tests/test_pipeline.py

# Run benchmarks
python tests/test_benchmark.py
```

---

## 📁 Project Structure

```
audio-restorer/
│
├── 📄 START HERE
│   ├── INDEX.md           ← You are here!
│   ├── AGENTS.md          ← Rules for LLMs
│   └── QUICKSTART.md      ← User quick start
│
├── 📋 PROJECT MANAGEMENT
│   ├── ITERATION_LOG.md   ← All changes logged
│   ├── ROADMAP.md         ← Current priorities
│   ├── PHASE2_PLAN.md     ← Next steps
│   └── QUALITY_METRICS.md ← How we measure quality
│
├── 🔧 CODE
│   ├── run.py             ← Main tool
│   ├── config.py          ← Settings
│   └── src/
│       ├── pipeline.py    ← Main orchestration
│       ├── audio_enhancer.py ← Audio processing
│       └── downloader.py  ← YouTube download
│
├── 🧪 TESTING
│   └── tests/
│       ├── measure_quality.py  ← Quality analysis
│       ├── test_pipeline.py    ← Integration tests
│       └── test_benchmark.py   ← Benchmarks
│
└── 📁 OUTPUT
    ├── output/            ← Enhanced audio files
    ├── benchmarks/        ← Test results
    └── temp/              ← Temporary files
```

---

## 🎓 How to Work on This Project

### As a User
1. Read **QUICKSTART.md**
2. Run `python run.py "URL" --audio-only --quick`
3. Listen to results in `output/` folder

### As an LLM Agent
1. **ALWAYS** read **AGENTS.md** first
2. Check **ITERATION_LOG.md** for current state
3. Check **ROADMAP.md** for priorities
4. Implement changes following AGENTS.md rules
5. **ALWAYS** update **ITERATION_LOG.md** after changes
6. **ALWAYS** test with reference video

---

## ✅ Success Checklist

### Phase 1 (Complete) ✅
- [x] Basic pipeline works
- [x] Quality metrics defined
- [x] Testing framework created
- [x] Documentation complete

### Phase 2 (In Progress)
- [ ] PyTorch installed
- [ ] DeepFilterNet integrated
- [ ] Quality score >80
- [ ] SNR >20 dB
- [ ] All tests pass

### Phase 3 (Future)
- [ ] Speaker diarization
- [ ] Speaker isolation
- [ ] Crowd noise reduction

---

## 📞 When to Read Which Document

| Situation | Read This |
|-----------|-----------|
| First time using the tool | QUICKSTART.md |
| Want to understand the project | README.md |
| About to make code changes | AGENTS.md |
| Want to know what to do next | ROADMAP.md |
| Ready to implement Phase 2 | PHASE2_PLAN.md |
| Made changes, need to document | ITERATION_LOG.md (template) |
| Want to measure audio quality | QUALITY_METRICS.md |
| Need to run tests | tests/measure_quality.py --help |

---

## 🎯 Project Goal (Reminders)

**Primary Goal:** Best possible audio quality for conference recordings  
**Secondary Goal:** Fast, easy-to-use tool  
**Non-Goal:** Video processing (secondary priority)

**Quality Target:** Professional podcast quality from poor conference audio

---

## 🔗 Quick Links

- **Test Video:** https://youtu.be/cglDoG0GzyA
- **Current Quality Score:** 61.7/100
- **Next Phase:** Phase 2 - ML Enhancement
- **Estimated Time to Phase 2:** ~4 hours

---

**Last Updated:** 2026-01-31  
**Current Status:** Phase 1 Complete, Phase 2 Ready  
**Documentation Version:** 1.0

---

**Ready to proceed?**

- To use the tool: See **QUICKSTART.md**
- To improve quality: See **PHASE2_PLAN.md**
- To understand everything: Read **AGENTS.md** → **ROADMAP.md** → **QUALITY_METRICS.md**
