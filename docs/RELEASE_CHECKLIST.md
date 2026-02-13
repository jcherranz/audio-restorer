# Release Checklist

Use this checklist before publishing a release.

## Preparation
- [ ] Ensure `ITERATION_LOG.md` is up to date
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Confirm `tasks/todo.md` has no in-progress items

## Verification
- [ ] `python -m pytest tests/test_modules.py -v`
- [ ] `python tests/ci_quality_smoke.py`
- [ ] `python run.py "https://youtu.be/cglDoG0GzyA" --audio-only --keep-temp`
- [ ] `python tests/measure_quality.py output/audio_cglDoG0GzyA_enhanced.wav`
- [ ] Optional: `python tests/sota_benchmark.py output/audio_cglDoG0GzyA_enhanced.wav`
- [ ] Optional: `python tests/multi_video_benchmark.py --videos 5 --sota`

## Release
- [ ] Tag the release (e.g., `vX.Y.Z`)
- [ ] Create GitHub release notes from `CHANGELOG.md`
- [ ] Announce key quality deltas (DNSMOS + quality score)

