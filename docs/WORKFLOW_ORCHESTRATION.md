# Workflow Orchestration

> **Purpose:** Process and mechanics for how to work.
> For mindset and behavioral principles, see `SENIOR_ENGINEER_PROMPT.md`.

## 1. Plan Mode

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

## 2. Subagent Strategy

Keep the main context window clean:
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

## 3. Self-Improvement Loop

After ANY correction from the user:
1. Update `tasks/lessons.md` with the pattern
2. Write rules for yourself that prevent the same mistake
3. Review lessons at session start

## 4. Verification

Never mark a task complete without proving it works:
- Run tests, check logs, demonstrate correctness
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"

## 5. Autonomous Bug Fixing

When given a bug report:
- Just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests -> then resolve them
- Go fix failing CI tests without being told how

## Task Management

Use `tasks/todo.md` for tracking:

1. **Plan First**: Write plan with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Document Results**: Add summary when done
5. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Notes

- If `tasks/` does not exist, create it with `todo.md` and `lessons.md`
- This document covers *process*. For *principles* (simplicity, scope discipline, etc.), see `SENIOR_ENGINEER_PROMPT.md`
