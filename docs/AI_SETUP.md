# colab-train: AI setup guide (for a teammate and their Claude Code)

Goal: a teammate with their own repo (PyTorch, Isaac, mjlab, anything with a train command) and their own AI
assistant can run `colab-train launch` and `status` with only a YAML and Colab Secrets.

## 0. Your own setup, in order (new person, new repo)

Use your own accounts and your own repo. Nothing here depends on anyone else's setup.

1. **Google account** with Colab. The free tier allows one session at a time, a daily GPU quota and about 10 h per
   runtime. Sign in once in the dedicated browser if you use `--browser`; the tool stops at any sign-in or captcha and
   waits for you.
2. **W&B: yes or no.** With W&B, create an account at wandb.ai, note your user or team name (the entity), and create an
   API key. Without W&B, skip all of this: use `wandb.mode: disabled` (the default when no entity is set) and
   `progress.regex`. Checkpoints then go to your Google Drive (`resume.source: drive`) or nowhere (`none`).
3. **Your repo** on GitHub, GitLab or another git host, with a train command that runs from the repo root. Public repos
   need no token. A private repo needs a token with read access to that repo only (fine-grained token on GitHub).
4. **This tool**: install (section 1), then in your repo `colab-train init` (asks, or `--yes` with flags), then
   `colab-train doctor`. Fix every `[FAIL]` line; `[warn]` lines are optional features.
5. **Secrets**: `colab-train secrets set GITHUB_TOKEN` and/or `colab-train secrets set WANDB_API_KEY`, typed in your
   terminal with a hidden prompt. Or add them yourself in Colab Secrets (key icon) with Notebook access ON. Never paste a
   value into a chat, a job, a notebook or a commit.
6. **Run**: `colab-train render jobs/<name>.yaml`, upload the notebook in Colab, set the GPU, Run all.

Examples to copy: `jobs/cleanrl-ppo-cartpole.yaml` (public repo, CPU, no W&B), `jobs/example-pytorch.yaml` (copy and
change `repo.url` and `train.command`) and `jobs/mjlab-rsl-rl.yaml` (mjlab + rsl_rl with W&B; placeholder repo and task id). `colab-train doctor` will tell you when
a value still has the example placeholder.

## 1. Install the CLI

Needs `uv` and git. The CLI is public, so no credential is needed to install it:

```bash
# wandb: needed when a job logs to W&B (status, logs, worker); browser: needed for --browser
uv tool install --force 'colab-train[wandb,browser] @ git+https://github.com/Barathaner/colab-gpu-trainer.git'
colab-train --version
colab-train list
```

Upgrade later with the same command (it has `--force`) (or `uvx --refresh --from git+... colab-train list`).

To run from a checkout instead: `uv run --with pyyaml --with wandb python -m colab_train list` in the repo.

## 2. Credentials (names only in the repo)

| where | what | how |
|---|---|---|
| Colab Secrets (key icon) | `WANDB_API_KEY` | your W&B API key, Notebook access ON (only with W&B) |
| Colab Secrets | your git token name, e.g. `GITHUB_TOKEN` (GitHub) or `GIT_TOKEN` (for a second git host) | a token with read access to that repo, Notebook access ON (only for private repos) |
| local store (`colab-train secrets set NAME`) | the same names, kept on your machine | hidden prompt; `colab-train secrets list` shows names only |
| your machine, `~/.netrc` | W&B key | `wandb login` writes it. Check presence only: `test -f ~/.netrc` |
| your machine, git | repo read access | git credential helper or a `~/.netrc` entry for the git host |
| GitHub (optional) | secret gist for one-click URLs | `gh auth login` once. Without it, `launch` writes a local notebook |

Never put any of these values in a YAML, a notebook, a command, or a commit. The CLI and the notebook use names only.

## 3. Write your job

```bash
cd /path/to/your-repo
colab-train init my-job --preset example-pytorch    # writes jobs/my-job.yaml
```

Edit `jobs/my-job.yaml`:

- `repo.url`: your repo (https). `repo.token_secret`: the secret NAME for private repos, or `null`.
- `install`: the commands that set up your environment, for example `pip install -r requirements.txt`, or
  `uv sync --frozen` with `python: "3.13"`.
- `train.command`: your train command. Use the placeholders in `docs/AI_USAGE.md` (`{iters}`, `{run_id}`,
  `{resume_args}` and so on). Pass `{resume_args}` (or `{resume_path}`) if the training can resume.
- `wandb.entity`/`project`, `run.name`, `train.max_iters`, `train.save_every` (checkpoint frequency, this is your
  resume point).
- `resume.source`: `wandb` if your training uploads checkpoints as W&B run files; `drive` if you want Google Drive
  (the notebook then asks for Drive consent once); `none` to start fresh each time.

Then:

```bash
colab-train validate jobs/my-job.yaml     # ok: ... or the errors to fix
colab-train render jobs/my-job.yaml --out build/my-job.ipynb   # optional look at the notebook
```

Start from a preset if it is close: `colab-train init my-copy --preset mjlab-rsl-rl`.

## 4. First run

```bash
colab-train launch jobs/my-job.yaml
```

It prints the notebook path (and a one-click gist URL if `gh` is logged in). Then, in Colab with the account you use
for W&B:

1. Open the notebook (URL, or **File > Upload notebook** on colab.research.google.com).
2. Check **Runtime > Change runtime type**: T4 GPU.
3. Add the secrets listed by `launch` (key icon), with Notebook access ON.
4. **Runtime > Run all**. Keep the tab open. A disconnect is normal: reconnect and Run all again.

Follow it from the terminal: `colab-train status my-job`, `colab-train logs my-job`.

## 4q. Shared queue (one Colab session per account)

The account runs one Colab session at a time; a second runtime gets "Too many sessions". Queue jobs instead:

```bash
colab-train submit my-job --run-name my-run-1   # adds the job to ~/.cache/colab-train/queue.json
colab-train worker                              # foreground, in its own terminal; starts the next job when the session is free
colab-train queue                               # status of every item
colab-train cancel ID                           # removes a queued job, or stops a running one (asks first)
```

Details and the stop rules: `README.md` (Queue and worker) and `docs/AI_USAGE.md`.

## 4a. Secrets (values stay on your machine)

```bash
colab-train secrets set GITHUB_TOKEN     # hidden input; the value is stored in ~/.config/colab-train (mode 600)
colab-train secrets import               # WANDB_API_KEY from ~/.netrc or the WANDB_API_KEY variable
colab-train secrets list                 # names and set=yes only
```

The AI assistant never asks you to paste a token into the chat. If it needs a value, it asks you to run `secrets set`.

## 4b. Optional: drive your own Chrome (`--browser`)

Install the extra (`uv tool install --force 'colab-train[wandb,browser] @ git+https://github.com/Barathaner/colab-gpu-trainer.git'`), then, in a terminal on your desktop, start the
dedicated browser and leave it open:

```bash
colab-train browser start
colab-train browser setup my-job     # opens Colab: sign in to Google yourself, create the Secrets once
colab-train launch --browser my-job  # uploads the notebook, sets T4, presses Run all
colab-train stop my-job --browser    # disconnect and delete the runtime (asks first)
```

The browser profile is separate from your everyday Chrome (`~/.cache/colab-train/browser-profile`). Passwords, 2FA
codes and Secret values are only ever typed by you. Set `"auto_jobs"` in `.colab-train/config.json` to choose the
default job for `launch --browser --auto`. The safety stops (captcha, robot check, usage limit, sign-in) are listed in
`docs/AI_USAGE.md`.

## 5. Optional: a Claude Code skill

Copy the skill so your Claude Code knows the workflow:

```bash
mkdir -p ~/.claude/skills/colab-train
cp /path/to/colab-gpu-trainer/skills/colab-train/SKILL.md ~/.claude/skills/colab-train/SKILL.md
```

Then ask Claude Code, for example: "launch my-job on Colab and follow it". It will use the CLI and never print
secrets.

## 6. Automating the Colab side: what is possible

Researched 2026-10-21. Summary: the reliable path is a rendered notebook plus one human click. Options:

- **Secret gist + Colab URL** (what `launch` does with `gh`). The URL form `colab.research.google.com/gist/<user>/<id>/<file>.ipynb`
  is the documented-by-pattern route; I could not confirm it against Google docs, so `launch` always prints the steps
  as well.
- **Official Colab MCP server** (github.com/googlecolab/colab-mcp, announced March 2026). It lets a local MCP client
  (Claude Code, Gemini CLI) drive a Colab notebook that is open in your own browser session: add cells, run cells,
  install packages. Reported issue: in some clients, notably Claude Code, only the connection tool appears, because
  the notebook tools rely on a `tools/list_changed` notification. Not integrated here: it needs your browser session
  and still needs you to open the notebook. A possible next step if you want cell-level control.
- **Colab VS Code extension** (Google, Nov 2025): attach a local notebook to a Colab runtime. Same human-in-the-loop
  requirement; not integrated.
- **Colab Enterprise / paid runtimes**: the supported route for unattended runs. Not used here.

Free-tier limits: Google's FAQ says a notebook runs for at most 12 hours and idle runtimes time out. Limits are not
published and fluctuate, and GPU availability is not guaranteed. The runner stops training after `limits.max_hours`
(default 10) and checkpoints are the resume point. Keep to one runtime at a time, and do not automate around Colab's
usage rules.

What is automated with `--browser`: the upload, the T4 choice, the Run all click, the two approved dialogs, and the
disconnect (after a confirmation), all in your own Chrome. What stays manual, and why: the Google sign-in and the Colab
Secrets (you do them once, by hand), and any captcha or usage-limit stop. Without `--browser`, the upload, GPU, Secrets
and Run all stay manual, and `stop` prints the steps.
