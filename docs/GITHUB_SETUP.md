# GitHub Setup Documentation

> **For AI Agents:** This document contains all information needed to interact with GitHub for this project.

## 🎯 When to Use GitHub - Decision Guide for AI Agents

### Auto-Commit Triggers (Commit & Push When)

| Trigger | Action Required | Example Commit Message |
|---------|----------------|------------------------|
| **Feature Complete** | Commit + Push | `Add DNSMOS metric calculation to quality gate` |
| **Bug Fix** | Commit + Push | `Fix: Handle NaN values in SNR calculation` |
| **Documentation Update** | Commit + Push | `Update AGENTS.md with new enhancer options` |
| **Test Added/Fixed** | Commit + Push | `Add unit tests for DeepFilter enhancer` |
| **Config Change** | Commit + Push | `Update default sample rate to 44.1kHz` |
| **Iteration Complete** | Commit + Push | `Complete iteration 7: Add Resemble Enhance` |

### Do NOT Commit (Yet)

| Situation | Reason | What to Do Instead |
|-----------|--------|-------------------|
| Work in progress | Unfinished code | Continue working, commit when feature works |
| Broken tests | Would break CI | Fix tests first, then commit |
| Temporary files | Not meant for repo | Add to .gitignore, don't commit |
| Secrets/API keys | Security risk | Use env files, add to .gitignore |
| Binary outputs | Large/temporary | Add output/* to .gitignore |

### Commit Frequency Guidelines

**Kaizen Workflow (This Project):**
- ✅ **End of each iteration** - Always commit
- ✅ **After fixing a bug** - Commit immediately
- ✅ **After adding tests** - Commit with the feature
- ✅ **Documentation updates** - Commit as you go
- ✅ **Before asking user** - Commit current progress
- ❌ **Every 5 minutes** - Too frequent
- ❌ **Only at end of day** - Risk losing work

### Pre-Commit Checklist

Before every commit, verify:

```bash
# 1. Check what changed
git status
git diff --stat

# 2. Are tests passing?
python -m pytest tests/ -v --tb=short

# 3. Is quality gate satisfied?
python tests/quality_gate.py output/test_enhanced.wav

# 4. Review your changes
git diff

# 5. Commit with descriptive message
git commit -m "type: description"
```

### Commit Message Format

```
type: Brief description (50 chars or less)

Longer explanation if needed (wrap at 72 chars).
Explain WHAT changed and WHY, not HOW.

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding/updating tests
- `refactor:` - Code change that neither fixes bug nor adds feature
- `perf:` - Performance improvement
- `chore:` - Maintenance, config, build changes

**Examples:**
```
feat: Add DeepFilterNet neural denoising

Implements DeepFilterNet2 for superior noise reduction.
Quality score improved from 81.0 to 115.9.

fix: Handle zero-division in SNR calculation

Prevents crash when processing silent audio segments.

docs: Update ROADMAP with Phase 3 timeline
```

## Repository Information

| Setting | Value |
|---------|-------|
| **Owner** | jcherranz |
| **Repository** | audio-restorer |
| **Full Name** | jcherranz/audio-restorer |
| **URL** | https://github.com/jcherranz/audio-restorer |
| **Clone URL** | https://github.com/jcherranz/audio-restorer.git |
| **SSH URL** | git@github.com:jcherranz/audio-restorer.git |
| **Visibility** | Public |
| **Default Branch** | main |

## Quick Commands

### Clone the Repository (Fresh Start)
```bash
git clone https://github.com/jcherranz/audio-restorer.git
cd audio-restorer
```

### Check Status
```bash
git status
git log --oneline -5
git remote -v
```

### Push Changes
```bash
# Add all changes
git add .

# Commit with message
git commit -m "Your commit message"

# Push to GitHub
git push origin main
```

### Pull Latest Changes
```bash
git pull origin main
```

## Authentication Methods

### Method 1: Personal Access Token (Recommended for Automation)

**Token Location:** `~/.config/github/audio-restorer.env`

**Usage:**
```bash
# Load credentials
source ~/.config/github/audio-restorer.env

# Configure git to use token
git remote set-url origin https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${GITHUB_REPO}.git

# Push
git push origin main

# Clean up (remove token from URL)
git remote set-url origin https://github.com/${GITHUB_USERNAME}/${GITHUB_REPO}.git
```

### Method 2: SSH Key (Recommended for Interactive Use)

**Your SSH Key:** `~/.ssh/id_ed25519.pub`
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICYrRXYtxc68283QFsK7YCLcPzZZGr4/5H4MLG2oSIoq jcherranz@DESKTOP-QRB28CC
```

**Setup:**
1. Add the SSH key to GitHub: https://github.com/settings/keys
2. Use SSH URL: `git remote set-url origin git@github.com:jcherranz/audio-restorer.git`

### Method 3: GitHub CLI

**Install Location:** `~/bin/gh`

**Usage:**
```bash
export PATH="$HOME/bin:$PATH"

# Authenticate with token
echo "$GITHUB_TOKEN" | gh auth login --with-token

# Or login interactively
gh auth login

# Create repo, issues, PRs, etc.
gh repo view
gh issue list
```

## Repository Structure on GitHub

```
audio-restorer/
├── .claude/
├── .gitignore          ← Generated for Python projects
├── docs/
│   ├── GITHUB_SETUP.md ← This file
│   └── primer.md
├── examples/
├── models/
├── src/                ← Main source code
├── tests/              ← Test files
├── benchmarks/
├── output/
├── temp/
├── venv/               ← Not tracked (in .gitignore)
├── AGENTS.md
├── README.md
├── ROADMAP.md
├── ITERATION_LOG.md
└── [other project files]
```

## Large Files Notice

The following files exceed GitHub's recommended 50MB limit but are committed:
- `ffmpeg` (76.13 MB)
- `ffprobe` (75.98 MB)

These are binary executables required for audio processing. They are tracked in Git but could be moved to Git LFS in the future if needed.

## Common Git Operations

### View Commit History
```bash
git log --oneline --graph -15
```

### Create a New Branch
```bash
git checkout -b feature-name
git push -u origin feature-name
```

### View Changes Before Committing
```bash
git diff                    # Show unstaged changes
git diff --staged          # Show staged changes
git diff HEAD~1            # Show changes from last commit
```

### Undo Changes
```bash
git checkout -- filename    # Undo changes to a file
git reset HEAD~1           # Undo last commit (keep changes)
git reset --hard HEAD~1    # Undo last commit (discard changes)
```

## GitHub API Usage

### Create Repository (Already Done)
```bash
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d '{"name":"audio-restorer","description":"Audio restoration tool","private":false}'
```

### Get Repository Info
```bash
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/jcherranz/audio-restorer
```

### List Issues
```bash
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/jcherranz/audio-restorer/issues
```

## Security Notes

1. **Token Permissions:** The current token has `repo` scope (full repository access)
2. **Token Storage:** Token is stored in `~/.config/github/audio-restorer.env` with 600 permissions (owner read/write only)
3. **Remote URL:** Always clean the token from the remote URL after use to prevent accidental exposure
4. **SSH Keys:** Prefer SSH for interactive use, tokens for automation

## Complete Agent Workflow

### Starting a Session

```bash
# 1. Navigate to project
cd ~/audio-restorer

# 2. Activate environment
source venv/bin/activate

# 3. Load GitHub credentials
source ~/.config/github/audio-restorer.env

# 4. Configure git for push
git remote set-url origin https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${GITHUB_REPO}.git

# 5. Pull latest changes
git pull origin main

# 6. Check status
git status
```

### During Work (Every Significant Change)

```bash
# Check what you've changed
git diff

# Check summary
git diff --stat

# If tests pass and you're ready to save progress:
git add .
git commit -m "type: Description"
```

### Ending a Session (Mandatory)

```bash
# 1. Final status check
git status

# 2. Run tests
python -m pytest tests/ -v

# 3. If tests pass, commit any remaining changes
if [ -n "$(git status --porcelain)" ]; then
    git add .
    git commit -m "chore: Session checkpoint - [brief summary of work done]"
fi

# 4. Push to GitHub
git push origin main

# 5. Cleanup (REMOVE token from URL!)
git remote set-url origin https://github.com/${GITHUB_USERNAME}/${GITHUB_REPO}.git

# 6. Verify cleanup
git remote -v
```

### Handling Common Scenarios

**Scenario 1: Remote has changes you don't have**
```bash
# Error: "failed to push some refs"
# Solution:
git pull origin main
# Resolve any conflicts if prompted
git push origin main
```

**Scenario 2: Made a mistake in last commit**
```bash
# Fix the mistake
git add .
git commit --amend -m "Corrected message"
git push --force-with-lease origin main  # Only if already pushed
```

**Scenario 3: Want to undo last commit (keep changes)**
```bash
git reset HEAD~1
# Files are unstaged but changes are preserved
git status
```

**Scenario 4: Accidentally committed secrets**
```bash
# DON'T PUSH!
git reset HEAD~1
git checkout -- the-secret-file
echo "the-secret-file" >> .gitignore
git add .gitignore
git commit -m "fix: Add secret file to .gitignore"
```

## Troubleshooting

### "Repository not found"
- Repository may not exist or token lacks permissions
- Check: `curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user/repos`

### "Permission denied"
- Token may be expired or lack required scopes
- SSH key may not be added to GitHub account

### "Failed to push some refs"
- Remote has changes you don't have locally
- Solution: `git pull origin main` then push again

### Large file warnings
- These are expected for ffmpeg/ffprobe binaries
- Warnings are harmless, but consider Git LFS for future large files

## First Push Complete

- **Date:** 2026-02-01
- **Commit:** 68c8aad (Initial commit: Audio restoration tool)
- **Files:** 38 files, 8614 insertions

---

**For AI Agents:** When working on this project:
1. Always check `git status` first
2. Commit changes with descriptive messages
3. Push to GitHub after significant milestones
4. Update this document if authentication methods change
