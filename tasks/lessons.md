# Lessons Learned

> After ANY correction from the user, document the pattern here to prevent repeating mistakes.
> See `docs/WORKFLOW_ORCHESTRATION.md` for the self-improvement loop.

## Patterns to Remember

### Documentation
- Always update ITERATION_LOG.md immediately after changes
- Keep ROADMAP.md phase status current
- Update CLI --help descriptions when adding flags

### Testing
- Test with reference video before claiming quality improvements
- Run quality gate before committing: `python tests/quality_gate.py`
- Use multi-video benchmark for statistical validation

### Dependencies
- Check requirements.txt when importing new modules
- scipy, numpy, soundfile are core dependencies
- DeepFilterNet requires 48kHz sample rate

### Code Quality
- New enhancers should inherit from BaseEnhancer or follow the interface
- Add fallback behavior when optional features fail
- Keep processing steps modular and optional via CLI flags

---

*Add new lessons below this line as they occur:*
