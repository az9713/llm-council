# Complete Git & GitHub CLI Guide

This guide covers everything you need to know about using Git and GitHub CLI (`gh`) for the LLM Council project. Written for beginners with no prior Git experience.

---

## Table of Contents

1. [Understanding Git Concepts](#understanding-git-concepts)
2. [One-Time Setup](#one-time-setup)
3. [Initial Repository Creation & First Push](#initial-repository-creation--first-push)
4. [Daily Workflow: Making Changes](#daily-workflow-making-changes)
5. [Feature Development with Branches](#feature-development-with-branches)
6. [Pull Requests (PRs)](#pull-requests-prs)
7. [Syncing with Remote](#syncing-with-remote)
8. [Common Commands Reference](#common-commands-reference)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Understanding Git Concepts

### What is Git?

Git is a **version control system** that tracks changes to your files. Think of it as:
- An "undo" system that remembers every version of your code
- A way to collaborate with others without overwriting each other's work
- A backup system that stores your code history

### Key Terminology

| Term | Definition |
|------|------------|
| **Repository (repo)** | A folder tracked by Git, containing your project and its history |
| **Commit** | A snapshot of your files at a specific point in time |
| **Branch** | A separate line of development (like a parallel universe for your code) |
| **Main/Master** | The primary branch, usually containing stable code |
| **Remote** | A copy of your repository stored online (e.g., on GitHub) |
| **Origin** | The default name for your remote repository |
| **Clone** | Download a copy of a remote repository |
| **Push** | Upload your local commits to the remote |
| **Pull** | Download commits from the remote to your local repo |
| **Stage** | Mark files to be included in the next commit |
| **Merge** | Combine changes from one branch into another |
| **Pull Request (PR)** | A request to merge your branch into another branch (with review) |

### Visual: How Git Works

```
Your Computer (Local)                    GitHub (Remote)
┌─────────────────────┐                 ┌─────────────────────┐
│                     │                 │                     │
│  Working Directory  │                 │   origin/main       │
│  (your actual files)│                 │   (GitHub copy)     │
│         │           │                 │                     │
│         ▼           │                 └─────────────────────┘
│   Staging Area      │                          ▲
│   (files ready to   │                          │
│    be committed)    │                          │
│         │           │      git push            │
│         ▼           │ ─────────────────────────┘
│   Local Repository  │
│   (.git folder)     │ ◄────────────────────────┐
│                     │      git pull            │
└─────────────────────┘                          │
                                                 │
                                    (downloads from GitHub)
```

---

## One-Time Setup

These steps only need to be done once on your computer.

### Step 1: Verify Git is Installed

```bash
git --version
```

**Expected output:** `git version 2.x.x`

If not installed, download from: https://git-scm.com/downloads

### Step 2: Verify GitHub CLI is Installed

```bash
gh --version
```

**Expected output:** `gh version 2.x.x`

If not installed, download from: https://cli.github.com/

### Step 3: Configure Your Identity

Git needs to know who you are for commit history:

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"
```

**Verify configuration:**

```bash
git config --global --list
```

### Step 4: Authenticate GitHub CLI

```bash
gh auth login
```

Follow the prompts:
1. Select `GitHub.com`
2. Select `HTTPS`
3. Select `Yes` to authenticate with GitHub credentials
4. Select `Login with a web browser`
5. Copy the one-time code shown
6. Press Enter to open browser
7. Paste the code in the browser
8. Authorize the application

**Verify authentication:**

```bash
gh auth status
```

---

## Initial Repository Creation & First Push

Use this flow when starting a new project or pushing an existing project to GitHub for the first time.

### Complete Flow (Copy-Paste Ready)

```bash
# Navigate to your project folder
cd "C:\Users\simon\Downloads\projects\TBD__llm-council_karpathy\llm-council-master"

# Step 1: Initialize Git repository
# This creates a hidden .git folder that tracks your project
git init

# Step 2: Stage all files for commit
# The "." means "all files" (respects .gitignore)
git add .

# Step 3: Create your first commit
# -m flag adds a message describing what this commit contains
git commit -m "Initial commit: LLM Council with 21 features"

# Step 4: Create GitHub repository AND push in one command
# --public    : Make repository visible to everyone (use --private for private)
# --source=.  : Use current directory as source
# --remote=origin : Name the remote "origin" (standard convention)
# --push      : Push immediately after creating
gh repo create llm-council --public --source=. --remote=origin --push
```

### What Each Command Does

| Command | What It Does |
|---------|--------------|
| `git init` | Creates a new Git repository in current folder |
| `git add .` | Stages ALL files (except those in .gitignore) for commit |
| `git commit -m "message"` | Creates a snapshot with a description |
| `gh repo create` | Creates repository on GitHub and connects it |

### After Success

Your repository will be available at:
```
https://github.com/YOUR_USERNAME/llm-council
```

---

## Daily Workflow: Making Changes

Use this flow when you make changes and want to save them.

### The Basic Cycle

```bash
# 1. Check what files have changed
git status

# 2. See the actual changes (optional, but recommended)
git diff

# 3. Stage your changes
git add .                    # Stage ALL changed files
# OR
git add filename.py          # Stage specific file
# OR
git add frontend/            # Stage specific folder

# 4. Commit with a descriptive message
git commit -m "Add feature X"

# 5. Push to GitHub
git push
```

### Understanding `git status` Output

```bash
$ git status

On branch main
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
        modified:   backend/council.py      # Changed but not staged

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        new_file.py                         # New file, not tracked

Changes to be committed:
        modified:   frontend/App.jsx        # Staged, ready to commit
```

### Commit Message Best Practices

```bash
# Good commit messages (describe WHAT and WHY)
git commit -m "Add caching feature to reduce API costs"
git commit -m "Fix bug where rankings weren't parsing correctly"
git commit -m "Update documentation for new users"

# Bad commit messages (too vague)
git commit -m "Update"
git commit -m "Fix bug"
git commit -m "Changes"
```

---

## Feature Development with Branches

When adding a new feature, use branches to keep your work separate from the main code.

### Why Use Branches?

- Keep `main` branch stable and working
- Experiment without breaking anything
- Easy to abandon failed experiments
- Required for Pull Requests

### Visual: How Branches Work

```
main:     A ─── B ─── C ─────────────── G ─── H
                       \               /
feature:                D ─── E ─── F

A = Initial commit
B, C = More commits on main
D, E, F = Your feature work (on separate branch)
G = Merge commit (feature merged into main)
H = Continues on main
```

### Complete Feature Branch Flow

```bash
# 1. Make sure you're on main and up-to-date
git checkout main
git pull origin main

# 2. Create and switch to a new branch
# Name should describe the feature (use hyphens, no spaces)
git checkout -b feature/add-export-pdf

# 3. Make your changes...
# (edit files, add new files, etc.)

# 4. Stage and commit your changes (can do multiple times)
git add .
git commit -m "Add PDF export button to UI"

git add .
git commit -m "Implement PDF generation logic"

# 5. Push your branch to GitHub
# -u flag sets up tracking (only needed first time for this branch)
git push -u origin feature/add-export-pdf

# 6. Create a Pull Request (see next section)
gh pr create

# 7. After PR is approved and merged, clean up
git checkout main           # Switch back to main
git pull origin main        # Get the merged changes
git branch -d feature/add-export-pdf  # Delete local branch
```

### Common Branch Commands

```bash
# List all branches (* marks current branch)
git branch

# List all branches including remote
git branch -a

# Switch to existing branch
git checkout branch-name

# Create new branch and switch to it
git checkout -b new-branch-name

# Delete a branch (must not be on that branch)
git branch -d branch-name

# Force delete (if branch has unmerged changes)
git branch -D branch-name
```

---

## Pull Requests (PRs)

Pull Requests are how you propose changes to be merged into the main branch.

### What is a Pull Request?

- A request to merge your branch into another branch
- Allows code review before merging
- Shows all changes in one place
- Can be discussed and refined

### Creating a Pull Request

#### Method 1: Interactive (Recommended for Beginners)

```bash
# Make sure your branch is pushed
git push -u origin your-branch-name

# Create PR interactively
gh pr create
```

You'll be prompted for:
1. **Title**: Short description (e.g., "Add PDF export feature")
2. **Body**: Detailed description (opens your text editor)
3. **Base branch**: Usually `main`

#### Method 2: One-Line Command

```bash
gh pr create --title "Add PDF export feature" --body "This PR adds the ability to export conversations as PDF files.

## Changes
- Added PDF button to export menu
- Implemented PDF generation using jsPDF library
- Added tests for PDF export

## Testing
- Tested with various conversation lengths
- Verified formatting is correct"
```

#### Method 3: Create PR with Web Browser

```bash
# Opens your browser to create the PR
gh pr create --web
```

### Viewing Pull Requests

```bash
# List all open PRs
gh pr list

# View specific PR details
gh pr view 123

# View PR in browser
gh pr view 123 --web
```

### Reviewing and Merging PRs

```bash
# Check out a PR locally to test it
gh pr checkout 123

# Approve a PR
gh pr review 123 --approve

# Request changes
gh pr review 123 --request-changes --body "Please fix the typo on line 42"

# Merge a PR
gh pr merge 123

# Merge options
gh pr merge 123 --merge    # Create merge commit (default)
gh pr merge 123 --squash   # Squash all commits into one
gh pr merge 123 --rebase   # Rebase commits onto main
```

### Complete PR Workflow Example

```bash
# === Starting new feature ===

# 1. Update main and create feature branch
git checkout main
git pull origin main
git checkout -b feature/dark-mode

# 2. Make changes and commit
# ... edit files ...
git add .
git commit -m "Add dark mode toggle to settings"

# ... more edits ...
git add .
git commit -m "Implement dark mode CSS variables"

# ... more edits ...
git add .
git commit -m "Add dark mode persistence to localStorage"

# 3. Push branch to GitHub
git push -u origin feature/dark-mode

# 4. Create Pull Request
gh pr create --title "Add dark mode support" --body "## Summary
Adds dark mode toggle to the application.

## Changes
- Toggle switch in settings panel
- CSS variables for theme colors
- Persists preference to localStorage

## Screenshots
[Add screenshots here]

## Testing
- Toggle works correctly
- Preference persists across sessions
- All components render correctly in both modes"

# 5. Address review feedback (if any)
# ... make requested changes ...
git add .
git commit -m "Address review feedback: fix contrast ratio"
git push

# 6. After approval, merge the PR
gh pr merge --squash

# 7. Clean up
git checkout main
git pull origin main
git branch -d feature/dark-mode
```

---

## Syncing with Remote

Keep your local repository in sync with GitHub.

### Getting Latest Changes

```bash
# Fetch changes (download but don't apply)
git fetch origin

# Pull changes (fetch + merge)
git pull origin main
```

### When to Pull

- Before starting new work
- Before creating a new branch
- When collaborators have pushed changes

### Handling Conflicts

If you and someone else edited the same file:

```bash
# Try to pull
git pull origin main

# If conflict occurs, Git will tell you which files
# Open the conflicted files - they'll look like:
<<<<<<< HEAD
Your changes
=======
Their changes
>>>>>>> origin/main

# Edit the file to resolve (keep what you want)
# Then:
git add .
git commit -m "Resolve merge conflicts"
git push
```

---

## Common Commands Reference

### Quick Reference Card

```bash
# ══════════════════════════════════════════════════════════════
# SETUP & CONFIG
# ══════════════════════════════════════════════════════════════
git init                          # Initialize new repo
git config --global user.name "Name"
git config --global user.email "email"
gh auth login                     # Authenticate GitHub CLI

# ══════════════════════════════════════════════════════════════
# DAILY WORKFLOW
# ══════════════════════════════════════════════════════════════
git status                        # Check what's changed
git diff                          # See actual changes
git add .                         # Stage all changes
git add file.py                   # Stage specific file
git commit -m "message"           # Commit staged changes
git push                          # Push to GitHub

# ══════════════════════════════════════════════════════════════
# BRANCHES
# ══════════════════════════════════════════════════════════════
git branch                        # List branches
git checkout main                 # Switch to main
git checkout -b new-branch        # Create & switch to branch
git branch -d branch-name         # Delete branch

# ══════════════════════════════════════════════════════════════
# SYNCING
# ══════════════════════════════════════════════════════════════
git pull origin main              # Get latest from GitHub
git fetch origin                  # Fetch without merging

# ══════════════════════════════════════════════════════════════
# PULL REQUESTS (using gh)
# ══════════════════════════════════════════════════════════════
gh pr create                      # Create PR interactively
gh pr list                        # List open PRs
gh pr view 123                    # View PR details
gh pr checkout 123                # Test a PR locally
gh pr merge 123                   # Merge a PR

# ══════════════════════════════════════════════════════════════
# REPOSITORY (using gh)
# ══════════════════════════════════════════════════════════════
gh repo create name --public --source=. --push  # Create & push
gh repo view                      # View repo info
gh repo clone user/repo           # Clone a repo

# ══════════════════════════════════════════════════════════════
# HISTORY & INFO
# ══════════════════════════════════════════════════════════════
git log                           # View commit history
git log --oneline                 # Compact history
git show abc123                   # Show specific commit
git blame file.py                 # Who changed each line

# ══════════════════════════════════════════════════════════════
# UNDO & FIX
# ══════════════════════════════════════════════════════════════
git checkout -- file.py           # Discard changes to file
git reset HEAD file.py            # Unstage a file
git reset --soft HEAD~1           # Undo last commit (keep changes)
git reset --hard HEAD~1           # Undo last commit (discard changes) ⚠️
git stash                         # Temporarily save changes
git stash pop                     # Restore stashed changes
```

---

## Troubleshooting

### Problem: "fatal: not a git repository"

**Cause:** You're not in a folder with Git initialized.

**Solution:**
```bash
git init
```

### Problem: "Please tell me who you are"

**Cause:** Git doesn't know your identity.

**Solution:**
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### Problem: "failed to push some refs"

**Cause:** Remote has changes you don't have locally.

**Solution:**
```bash
git pull origin main
# Resolve any conflicts if needed
git push
```

### Problem: "Permission denied (publickey)"

**Cause:** SSH key issues.

**Solution:** Use HTTPS instead, or set up SSH keys:
```bash
gh auth login
# Select HTTPS when prompted
```

### Problem: "Your branch is behind"

**Cause:** Others pushed changes you don't have.

**Solution:**
```bash
git pull origin main
```

### Problem: Accidentally committed sensitive file (.env)

**Solution:**
```bash
# Remove from Git but keep local file
git rm --cached .env
git commit -m "Remove .env from tracking"
git push

# Make sure .env is in .gitignore
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Add .env to gitignore"
git push
```

### Problem: Want to undo last commit

**Solution:**
```bash
# Undo commit but keep changes
git reset --soft HEAD~1

# Undo commit and discard changes (⚠️ DANGEROUS)
git reset --hard HEAD~1
```

### Problem: Merge conflicts

**Solution:**
1. Open the conflicted file(s)
2. Look for conflict markers: `<<<<<<<`, `=======`, `>>>>>>>`
3. Edit to keep the code you want
4. Remove the conflict markers
5. Save the file
6. `git add .` and `git commit -m "Resolve conflicts"`

---

## Best Practices

### Do's

1. **Commit often** - Small, focused commits are easier to understand and revert
2. **Write clear commit messages** - Future you will thank present you
3. **Pull before you push** - Avoid conflicts by staying up-to-date
4. **Use branches for features** - Keep main branch stable
5. **Review your changes before committing** - Use `git diff` and `git status`
6. **Keep .gitignore updated** - Don't commit secrets or generated files

### Don'ts

1. **Don't commit secrets** - No API keys, passwords, or .env files
2. **Don't force push to main** - Can destroy others' work
3. **Don't commit large binary files** - Git isn't designed for them
4. **Don't ignore merge conflicts** - Resolve them properly
5. **Don't commit broken code to main** - Use branches instead

### Commit Message Format

```
<type>: <short description>

<optional longer description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Formatting (no code change)
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance tasks
```

**Examples:**
```bash
git commit -m "feat: Add PDF export functionality"
git commit -m "fix: Resolve ranking parse error for edge cases"
git commit -m "docs: Update README with new setup instructions"
git commit -m "refactor: Simplify council.py response handling"
```

---

## Quick Start Checklist

For your LLM Council project, here's exactly what to do:

```bash
# One-time: Navigate to project
cd "C:\Users\simon\Downloads\projects\TBD__llm-council_karpathy\llm-council-master"

# One-time: Initialize and push
git init
git add .
git commit -m "Initial commit: LLM Council with 21 features"
gh repo create llm-council --public --source=. --remote=origin --push

# Future changes: Daily workflow
git add .
git commit -m "Description of what you changed"
git push

# Future features: Branch workflow
git checkout -b feature/your-feature-name
# ... make changes ...
git add .
git commit -m "Description"
git push -u origin feature/your-feature-name
gh pr create
# After merge:
git checkout main
git pull
git branch -d feature/your-feature-name
```

---

## Additional Resources

- [Git Official Documentation](https://git-scm.com/doc)
- [GitHub CLI Manual](https://cli.github.com/manual/)
- [GitHub Guides](https://guides.github.com/)
- [Interactive Git Tutorial](https://learngitbranching.js.org/)

---

*This guide was created for the LLM Council project. Last updated: 2025-11-29*
