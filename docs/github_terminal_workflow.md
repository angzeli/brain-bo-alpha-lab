# GitHub Terminal Workflow

This is the practical Git/GitHub workflow for the `brain-bo-alpha-lab` project.

It is written for teammates who may not have used Git or the terminal before.

---

## 0. What is the terminal?

The **terminal** is a text-based way to control your computer and your project folder.

Instead of clicking buttons, you type commands such as:

```
git status
```

For this project, you will mainly use the terminal to:

- download the newest version of the project from GitHub,
- check which files you have changed,
- save a checkpoint of your changes,
- and upload your changes back to GitHub.

You do **not** need to memorise every command. Most of the time, you will repeat the same small workflow.

### Make sure you are in the project folder

Before running Git commands, your terminal should be inside the project folder.

For this project, that folder is likely called:

```
brain-bo-alpha-lab
```

You can check your current folder with:

```
pwd
```

You can list the files in the current folder with:

```
ls
```

If you see files such as:

```
alpha_bo.py
alpha.ipynb
README.md
.gitignore
```

then you are probably in the right place.

### The basic idea of Git

Think of Git as a checkpoint system:

```
git pull      = get the latest version from GitHub
git status    = check what changed
git add       = choose changes for the next checkpoint
git commit    = create the checkpoint locally
git push      = upload the checkpoint to GitHub
```

The safest routine is:

```
git pull --rebase origin main
git status
git add alpha_bo.py alpha.ipynb README.md
git commit -m "Describe your change"
git push
```

---

## 1. Download the repo for the first time

If this is your first time working on the project, you need to download the GitHub repository to your computer.

Choose a location where you want to keep the project, then run:

```
git clone https://github.com/angzeli/brain-bo-alpha-lab.git
```

This creates a new folder called:

```
brain-bo-alpha-lab
```

Move into that folder with:

```
cd brain-bo-alpha-lab
```

Now check that you are in the right place:

```
ls
```

You should see files such as:

```
alpha_bo.py
alpha.ipynb
README.md
.gitignore
```

You only need to run `git clone` once. After the project has been downloaded, use `git pull` to get future updates.

---

## 2. Check current status

Before doing anything, always check:

```
git status
```

This tells you whether files are modified, staged, committed, or already synced with GitHub.

A clean output looks like:

```
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

This means everything is synced.

---

## 3. Pull latest changes from GitHub

Before you start editing, run:

```
git pull --rebase origin main
```

This updates your local project with the newest GitHub version.

Use this especially if:

- you edited files on GitHub directly,
- your teammate pushed changes,
- you are working on another computer,
- or Git says your push was rejected because you need to “fetch first”.

---

## 4. See what changed

After editing files, run:

```
git status
```

You may see something like:

```
Changes not staged for commit:
    modified:   alpha_bo.py
    modified:   alpha.ipynb
```

This means Git sees your changes, but they are not included in the next commit yet.

---

## 5. Stage changed files

To include specific modified files in your next commit:

```
git add alpha_bo.py alpha.ipynb README.md
```

Important: `git add` does **not** only mean “add new files”. It means:

> add these changes to the next commit
> 

So you use `git add` for both new files and modified files.

To stage everything:

```
git add .
```

Be careful with `git add .` — only use it if your `.gitignore` is correct.

---

## 6. Commit changes

A commit is like a saved checkpoint.

Basic commit:

```
git commit -m "Add user-specific BO logging"
```

Better commit with more detail:

```
git commit -m "Add user-specific BO logging" -m "Create separate resume-safe CSV logs by region, universe, and user so teammates can run independent BO campaigns."
```

The first `-m` is the title.

The second `-m` is the longer explanation.

---

## 7. Push to GitHub

After committing locally, upload the commit to GitHub:

```
git push
```

If this works, GitHub is updated.

Then check:

```
git status
```

You want:

```
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
```

---

## 8. Normal daily workflow

Most of the time, your workflow is:

```
git pull --rebase origin main
git status
git add alpha_bo.py alpha.ipynb README.md
git commit -m "Describe your change"
git push
git status
```

Example:

```
git pull --rebase origin main
git status
git add alpha_bo.py alpha.ipynb
git commit -m "Fix resume-safe user logging"
git push
git status
```

---

## 9. If push is rejected

If you see:

```
! [rejected] main -> main (fetch first)
error: failed to push some refs
```

It means GitHub has newer changes that you do not have locally.

Fix with:

```
git pull --rebase origin main
git push
```

This means:

> download GitHub changes first, then replay my local commit on top, then push again
> 

This is what happened when `README.md` was changed on GitHub.

---

## 10. If there is a rebase conflict

Sometimes after:

```
git pull --rebase origin main
```

Git may say there is a conflict.

Check:

```
git status
```

Open the conflicted file and look for conflict markers:

```
<<<<<<< HEAD
GitHub version
=======
your local version
>>>>>>> your commit
```

Edit the file manually so only the correct final version remains.

Then run:

```
git add conflicted_file.py
git rebase --continue
```

After the rebase finishes:

```
git push
```

If you panic and want to cancel the rebase:

```
git rebase --abort
```

---

## 11. Check commit history

To see recent commits:

```
git log --oneline
```

Example output:

```
c28edb6 Add user-specific BO logging
ea16f2a Update README
11042b2 Initial project setup
```

To quit the log view, press:

```
q
```

---

## 12. See exactly what changed before staging

Before `git add`, use:

```
git diff
```

This shows unstaged changes.

After `git add`, use:

```
git diff --staged
```

This shows what will be included in the next commit.

---

## 13. Undo unstaged local changes

If you modified a file but want to discard the local changes:

```
git restore alpha_bo.py
```

For multiple files:

```
git restore alpha_bo.py alpha.ipynb
```

Be careful: this removes your uncommitted changes.

---

## 14. Unstage a file

If you accidentally staged a file with `git add`, unstage it with:

```
git restore --staged alpha_bo.py
```

This does not delete the changes. It just removes them from the next commit.

---

## 15. Amend the last commit message

If you just committed and want to change the message before pushing:

```
git commit --amend -m "Better commit message"
```

Then push:

```
git push
```

If you already pushed the commit, avoid amending for now unless you know what you are doing.

---

## 16. Files you should usually commit

For this project, commit:

```
alpha_bo.py
alpha.ipynb
README.md
.gitignore
LICENSE
```

---

## 17. Files you should not commit

Do not commit experiment logs or private outputs:

```
brain_bo_*.csv
brain_bo_log.csv
__pycache__/
.ipynb_checkpoints/
.venv/
.idea/
.DS_Store
```

Your `.gitignore` should handle these.

---

## 18. Check remote repo URL

To see where your local repo pushes:

```
git remote -v
```

Expected output:

```
origin  <https://github.com/angzeli/brain-bo-alpha-lab.git> (fetch)
origin  <https://github.com/angzeli/brain-bo-alpha-lab.git> (push)
```

---

## 19. If authentication fails

If GitHub asks for password, use a **Personal Access Token**, not your GitHub password.

Username:

```
angzeli
```

Password:

```
your GitHub token
```

If macOS cached a bad token, clear it:

```
git credential-osxkeychain erase
```

Then type:

```
protocol=https
host=github.com
```

Press Enter twice, then try:

```
git push
```
