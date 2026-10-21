---
name: colab-train
description: Launch, monitor and resume training jobs on a free Google Colab GPU with the colab-train CLI. Use when the user asks to train on Colab, launch or check a Colab run, see a W&B run's progress or videos, resume a Colab training, or set up a new repo for Colab training.
---

# colab-train

Drives a training job on a free Colab T4 from the terminal. Rendered notebook + W&B status. The human does one click
(Run all) and the Secrets setup. Full guide: `docs/AI_USAGE.md` in the colab-train repo.

## Rules

- Never print, echo, cat, log or commit secret values (W&B key in ~/.netrc, git host tokens). Secrets are Colab
  Secret NAMES only. Never put a token into a YAML, a notebook, a command line, a URL or a commit.
- Never ask the user to paste a token into the chat. If a value is missing, tell them to run
  `colab-train secrets set NAME` in a terminal, or run `colab-train secrets import` for the local sources.
- Do not train on the local GPU. No nohup/setsid/background processes. No keep-alive or Colab UI scripting.
- One runtime per run id at a time. Check `status` before telling the human to start.

## Commands

```bash
uv tool install --force 'colab-train[wandb,browser] @ git+https://github.com/Barathaner/colab-gpu-trainer.git'   # install; wandb extra needed for W&B jobs
colab-train list
colab-train init --yes --repo URL --train CMD   # new job for a repo (asks in a terminal; flags otherwise)
colab-train doctor [JOB]            # local setup, job spec, secret NAMES, repo URL (HEAD only); exit 1 on FAIL
colab-train validate <job>          # JOB: preset (example-pytorch, cleanrl-ppo-cartpole, sb3-ppo, mjlab-rsl-rl), jobs/<name>.yaml or a path
colab-train launch <job>            # writes .colab-train/notebooks/*.ipynb, prints Colab URL + steps
colab-train status <job>            # W&B state, iteration, s/iter, ETA, checkpoints, videos (W&B off: progress from the --browser tab)
colab-train logs <job> --lines 60
colab-train resume <job>            # running? finished? or the notebook to Run all again
colab-train stop <job>              # prints the manual browser step

# browser (optional, needs colab-train[browser]); the human starts Chrome, signs in and makes the Secrets once
colab-train browser status          # is the dedicated browser running?
colab-train browser setup <job>     # human signs in once; fills the job's Secrets from the local store
colab-train launch --browser <job>  # upload, T4, Run all in the human's Chrome; exit 3 = refused or safety stop
colab-train launch --browser --auto # no job: priority list from .colab-train/config.json, new run name
colab-train stop <job> --browser    # Runtime > Disconnect and delete runtime (asks first; --yes skips)

# shared queue (one Colab session per account)
colab-train submit <job> --run-name <name> --priority <n>  # adds to the queue; prints "queued ID: ..."
colab-train queue [--json]          # id, status, progress; queued/running/stalled/finished/failed/cancelled/needs-human
colab-train cancel <id> [--yes]     # queued: cancelled; running: browser stop first
colab-train worker --once           # one reconcile + start pass (foreground; no loop). `worker` alone loops
```

## Workflow

1. `validate` the job; fix errors.
2. `launch`; give the human the URL (or notebook path + upload steps) and the secret names to enable. Check T4.
   With several jobs, use `submit` and let the worker run them one at a time instead.
3. After the human says Run all was pressed, poll `status` every 5 to 10 minutes and report iteration, speed, ETA.
4. On crash or disconnect: `logs`, diagnose from the log, then `resume` and ask for Run all again.
   Queued jobs: the worker re-queues a disconnect itself (2 times at most); check with `queue`.
5. Finished when state is `finished` and the log ends with `[colab-train] Done.`.
6. Stop and report to the human on `needs-human` (captcha, sign-in, robot check). Never start a second runtime while
   one runs. The queue pauses 2 h after a usage limit or no-GPU stop.

With `--browser`: never sign in, type passwords or 2FA, or read Secret values. On a safety stop (captcha, robot check,
usage limit, sign-in page; exit 3 with a screenshot path), stop and show the human the screenshot. Do not retry in a
loop. Details: `docs/AI_USAGE.md`.

## Setting up a new repo

`colab-train init --yes --repo https://github.com/<user>/<repo>.git --train "python train.py --steps {iters}"` writes
`jobs/<name>.yaml` (add `--wandb yes --entity <user>` only if the human wants W&B; `--token-secret GITHUB_TOKEN` for a
private repo; `--gpu none` for CPU). Then `colab-train doctor jobs/<name>.yaml`. Use `progress.regex` for progress
without W&B. Never ask the human to paste a token into the chat: tell them `colab-train secrets set NAME` (hidden
prompt) or Colab Secrets with Notebook access ON.

## Known limits

Tell the human plainly: one Colab session at a time, a daily GPU quota, about 10 h max runtime. The tool stops and
asks a human on captcha or sign-in; there is no keep-alive. A job without W&B shows progress only while its Colab tab
is open. A fine-grained GitHub token reads only the repos listed in its Repository access. Open problems and ideas:
`docs/OPEN_PROBLEMS.md`.
