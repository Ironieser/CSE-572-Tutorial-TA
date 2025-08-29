# Git Basic Commands

Essential Git commands for beginners.

## 🚀 Setup

```bash
# Set your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## 📁 Get Started

```bash
# Download a project
git clone https://github.com/username/project.git

# Start a new project
git init
```

## 📝 Basic Workflow

```bash
# Check what files changed
git status

# Add files to save
git add filename.py
git add .                    # Add all files

# Save your changes
git commit -m "Add new feature"

# Upload to GitHub
git push origin main

# Download updates
git pull origin main
```

## 🌿 Branches

```bash
# Create new branch
git checkout -b new-feature

# Switch branches
git checkout main

# Merge branch
git merge new-feature
```

## 📋 View History

```bash
# See recent changes
git log --oneline

# See what changed
git diff
```

## 🚨 Common Problems

**Undo changes:**
```bash
git checkout -- filename.py
```

**Change last commit:**
```bash
git commit --amend -m "New message"
```

## 📚 Learn More
- [Git Tutorial](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
