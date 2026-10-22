# colab-gpu-trainer: architecture (arc42)

State: 2026-10-22. Short and factual; the README is the user guide, `docs/OPEN_PROBLEMS.md` the running list of
gaps.

## 1. Introduction and goals

`colab-train` runs training jobs on Google Colab GPUs without a person at the keyboard. A job is a YAML file (repo,
install, train command, W&B, resume, optional bench). The tool renders it into a self-contained notebook, opens that
notebook in a dedicated Chrome, starts it, watches progress through W&B and the Colab tab, resumes after disconnects,
and deletes the runtime when the job ends. A shared queue and a foreground worker let several people or agents feed
the same Colab account.

Quality goals, in order:

1. **No lost work.** Checkpoints and result files survive disconnects, failed uploads and deleted runtimes.
2. **Reproducible runs.** A job can be pinned to a commit; every W&B run records the commit it trained.
3. **Safe with secrets.** Job files hold secret names only; values never reach argv, logs or the notebook file.
4. **Unattended but honest.** The worker acts on what it can verify and reports "unknown" otherwise; it never clicks
   through sign-in, captcha or consent pages.

Stakeholders: people training RL or ML models on Colab Pro, and coding agents that submit jobs for them.

## 2. Constraints

- Colab has no public API for runtimes, secrets, quota or compute units. Everything goes through the web UI.
- Colab Pro gives at most 3 sessions at once, of which at most 2 can be L4. Usage costs about 1.1 compute units per
  hour on a T4, 1.5 on an L4 and 5.3 on an A100.
- Google sign-in, captcha and Drive consent need a human. The tool stops and reports instead of automating them.
- The runner runs inside Colab's Python, so it uses the standard library only (plus `wandb` when the job logs to
  W&B).
- Python 3.10+ locally; Playwright only for the browser mode.

## 3. Context and scope

```
 person / agent ──> colab-train CLI ──> queue.json ──> worker (foreground)
                         │                                 │
                         │ render                          │ launch / probe / stop
                         v                                 v
                    notebook (.ipynb) ──────────> dedicated Chrome (CDP) ──> Colab runtime
                                                                                │ runner.py
                                     GitHub (job repo, pinned commit) <─────────┤
                                     W&B (metrics, checkpoints, config) <───────┘
                                     local results/ <── browser download of the outputs folder
```

External systems: Colab (web UI), GitHub (job code, token from Colab Secrets), W&B (metrics, checkpoints, run
config), optionally Google Drive (checkpoints, needs a human consent once per runtime).

Out of scope: the training code itself, benchmark definitions, and other compute providers (RunPod pods are run by
separate scripts).

## 4. Solution strategy

- **Spec first.** One validated job dict (`spec.py`) feeds the CLI, the notebook renderer and the runner, so all three
  agree on paths, run ids and resume rules.
- **Self-contained notebook.** The runner source and the job JSON are embedded in the notebook; Colab needs nothing
  from this machine at run time.
- **Real browser, dedicated profile.** Playwright drives an ordinary Chrome over CDP with its own profile and English
  UI. No human-like movement, no bot-detection evasion.
- **W&B as source of truth.** Progress, state and checkpoints come from W&B; the Colab tab is the fallback for jobs
  without W&B.
- **Pure decision functions.** The worker's `reconcile` and `decide_launch` take values and return changes, so the
  scheduling rules are unit-tested without Colab.

## 5. Building block view

| Module | Responsibility |
|---|---|
| `cli.py` | Verbs: `init`, `doctor`, `validate`, `render`, `launch`, `status`, `logs`, `resume`, `stop`, `submit`, `queue`, `cancel`, `worker`, `secrets`, `browser`. |
| `spec.py` | YAML to validated job; defaults, unknown-key errors, derived run id (`<prefix>-<name>`) and paths. |
| `render.py` | Job to notebook: GPU check, secrets, Drive (only if used), runner, run cell with results download. |
| `runner.py` | On Colab: clone (optionally pinned commit), install, pick GL backend, resume from W&B or Drive, train with time budget, bench, collect outputs. |
| `browser.py`, `colab_web.py` | Start the dedicated Chrome; open, connect, run, read and stop notebook tabs; detect session limits, safety pages and missing GPUs. |
| `secrets.py`, `secrets_panel.py` | Local secret store (0600 file or keyring) and filling Colab's Secrets panel. |
| `queue.py` | Shared queue file with locked read-modify-write; priorities. |
| `worker.py` | Reconcile running items with W&B and the tab, stop ended runtimes, keep result downloads, launch the next item within GPU limits. |
| `wandb_status.py` | Read-only W&B queries: run by id, progress, checkpoints, logs. |
| `pick.py`, `publish.py` | Default job choice; where the rendered notebook is published (gist or local file). |

## 6. Runtime view

**Queued job to finished run.**

1. `colab-train submit job.yaml --priority N` appends an item to `~/.cache/colab-train/queue.json`. Only the path
   is stored; the file is read again at launch.
2. The worker (every 120 s) picks the highest-priority item whose GPU type is under its limit and runs
   `colab-train launch job.yaml --browser` as a child process. That renders the notebook, opens it, sets the
   accelerator, fills the secrets and runs all cells.
3. On Colab, the runner clones the repo, checks out `repo.commit` if set, logs `repo at commit <sha>`, installs,
   resumes from the newest checkpoint and trains. Every W&B run gets `colab_train_git_commit` in its config.
4. The worker probes each running item: the W&B run state and iteration, else the tab's runner lines. A stall, a
   crashed run or an error line leads to a resume (up to `max_resumes`) or a failed item. A probe that cannot ask W&B
   at all counts as unknown and never fails a job.
5. When the runner exits, the run cell packs `outputs/` (full log, bench JSONs, the job's `outputs:` files) and
   downloads it through the browser. The worker waits up to 3 min for that file, moves it to
   `~/.cache/colab-train/results/<run id>/`, then disconnects and deletes the runtime and closes the tab.

**Disconnect.** Re-running the training cell resumes from the newest checkpoint (W&B or Drive); the worker does this
by relaunching the job.

## 7. Deployment view

- Local machine: the CLI and worker in a Python environment; the dedicated Chrome with its profile in
  `~/.cache/colab-train/browser-profile`, CDP on a local port; queue, events and results under
  `~/.cache/colab-train/`; secret values in `~/.config/colab-train/secrets.json` (0600) or the OS keyring.
- Colab runtime: `/content/colab-train/` holds `runner.py`, `job.json`, the cloned repo and `outputs/`.
- The worker runs in the foreground of a terminal or agent session; nothing is detached.

## 8. Crosscutting concepts

- **Secrets.** Job files and notebooks contain secret names. Git gets its token from an environment variable through
  a credential helper, offered only for the repo's host. Logged argv never contains a value.
- **W&B standard.** Run id and name `<run.prefix>-<run.name>`, entity and project from the job, resume `allow`.
  Runs are looked up by id first. The config holds the training config plus `colab_train_git_commit`. The
  resume shim lets the trainer change its config on a resumed run only (opt-in through `resume.wandb_config_shim`).
- **Reproducibility.** `repo.commit` pins the code; the runner logs the actual HEAD in every case.
- **Result keeping.** The runner copies results into `outputs/` after every command and every minute while training;
  the notebook downloads the folder when the runner exits; the worker keeps the download before deleting the runtime.
  A job script must write its result files to disk before uploading them.
- **Commands.** A command with shell syntax (`&&`, pipes, `$(...)`, `~/`, a leading `VAR=value`) runs through
  `bash -c`; a plain one is split without a shell.
- **Errors.** Messages name the exact key or step and the fix. Browser steps stop with a screenshot on unexpected
  pages.

## 9. Architecture decisions

1. **Browser automation instead of an unofficial API.** Colab has no supported API; the UI is what a person uses
   and what the terms allow.
2. **W&B before Drive for checkpoints.** Drive needs a human consent on every new runtime; W&B works unattended.
3. **Results downloaded through the browser, not Drive.** Same reason; the dedicated Chrome is already connected.
4. **One queue file with a lock** instead of a server: several local sessions submit; no extra service to run.
5. **Commit pinning in the job file.** Later pushes to a branch must not change what a queued or restarted job
   trains.
6. **The queue stores job paths, not copies.** Edits before launch take effect; a running worker must be restarted
   before job files use new spec keys.

## 10. Quality requirements

| Scenario | Expected |
|---|---|
| A job's own W&B upload crashes after it wrote its JSON | The JSON is in `results/<run id>/` locally. |
| The branch moves after a pinned job was queued | The job trains the pinned commit; W&B config shows it. |
| Two L4 sessions run (`gpu_limits: {L4: 2}` in the config) | The worker skips L4 items and launches a T4 item instead. |
| The runtime disconnects mid-run | The relaunch resumes from the newest checkpoint. |
| A job file stops loading while the job runs | The worker reports a probe error and leaves the job running. |
| A sign-in or captcha page appears | The tool stops and reports that a human is needed. |

## 11. Risks and technical debt

- Colab UI changes break browser steps without notice (labels, shadow DOM).
- The GPU quota and compute units are visible only in the UI; a refused GPU shows up only at launch.
- If the VM dies mid-run, only what W&B (or Drive) already holds survives; the results download happens at runner
  exit.
- A very chatty trainer can freeze the tab and the automation.
- Resuming a W&B run whose history is ahead of the checkpoint loses log steps until it passes the old maximum.
- `limits.max_hours` is enforced for training only, not for bench-only commands.
- The worker blocks for up to 3 min per ended job while it waits for the results download.

See `docs/OPEN_PROBLEMS.md` for the full list.

## 12. Glossary

| Term | Meaning |
|---|---|
| Job | A YAML file describing one training run. |
| Run id | `<prefix>-<name>`; W&B run id and name. |
| Item | One queue entry: job path, run name, priority, status. |
| Worker | The foreground scheduler that launches, probes and stops Colab runs. |
| Runner | `runner.py`, executed on the Colab runtime. |
| Outputs | The per-run folder of result files that the notebook downloads at the end. |
| Pin | `repo.commit`, the exact commit a job trains. |
| CDP | Chrome DevTools Protocol, used by Playwright to drive the dedicated Chrome. |
