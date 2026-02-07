# CLAUDE.md — Auto-loaded every session

## GitHub Auth (do this first)
```bash
source ~/.config/github/audio-restorer.env && export GH_TOKEN="$GITHUB_TOKEN"
```
- SSH works for git push/pull: `git@github.com:jcherranz/audio-restorer.git`
- Token needed for `gh` CLI (PRs, issues, repo info)
- Full reference: `docs/GITHUB_SETUP.md`

## Environment
```bash
source venv/bin/activate
```
- Use `python` (not `python3`) inside venv
- ffmpeg binary is local at project root — use `FFMPEG_PATH` from config.py

## Read Order (for iterations)
1. `AGENTS.md` — Project context, rules, architecture, current status
2. `ITERATION_LOG.md` — What's been done
3. `tasks/todo.md` — Current priorities

## Key Constraints
- **Audio-only focus** — video processing is secondary
- **One iteration per session** — complete, test, document
- **No regressions** — quality score >= 75, SNR >= 25dB
- **39 unit tests must pass** before committing: `python -m pytest tests/test_modules.py -v`
- **48kHz sample rate** throughout (matches DeepFilterNet)

## Quick Commands
```bash
# Run tests
python -m pytest tests/test_modules.py -v

# Process reference video
python run.py "https://youtu.be/cglDoG0GzyA" --audio-only --keep-temp

# Measure quality
python tests/sota_benchmark.py output/audio_cglDoG0GzyA_enhanced.wav

# Git push
git push origin main
```

## Commit Format
`type: Brief description` — types: feat, fix, docs, test, refactor, perf, chore
