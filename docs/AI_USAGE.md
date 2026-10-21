# colab-train: usage guide for an AI agent

You drive Colab training through the `colab-train` CLI from a terminal (Claude Code or similar). Two ways to run:

- **Manual** (default): `launch` prints a Colab URL and steps; the human presses **Run all**.
- **Browser** (`launch --browser`, `stop --browser`): the CLI drives the human's dedicated Chrome for the Colab clicks
  listed below. The human signs in once. Everything else you do with the commands below.

## Hard rules

- Never print, echo, cat, log, or commit a secret value. Secrets are Colab Secret **names** (`GITHUB_TOKEN`,
  `WANDB_API_KEY`, `GIT_TOKEN`). Never write a token into a job YAML, a notebook, a command line, or a URL. Never
  read `~/.netrc` contents or a token file. Check presence only (for example `test -f ~/.netrc`).
- Do not run training on the local GPU. Do not run detached processes (`nohup`, `setsid`, `&`).
- The browser clicks are only the ones `--browser` makes: File > Upload notebook, Runtime > Change runtime type (T4),
  Runtime > Run all, the "Run anyway" and "Grant access" dialogs, and Runtime > Disconnect and delete runtime after a
  confirmation. Do not add other clicks, scripts, keep-alive, or auto-reconnect, and do not start a second runtime
  while one is running (check `status` first).
- Never sign in to Google, never type a password or 2FA code, and never read a secret value. The human signs in once,
  in the browser window.
- Secret values come from the local store (`colab-train secrets`), never from chat. **Never ask the user to paste a
  token, key or password into the chat.** If a value is missing, tell the user to run `colab-train secrets set NAME`
  in a terminal (the input is hidden), or run `colab-train secrets import` yourself when the source is local (W&B key
  from `~/.netrc`, a git host, or a `--from-file` the user names). `secrets list` shows names and a set flag only.
- On a safety stop (below), do not retry in a loop and do not click past it. Report the screenshot to the human.
- Do not push or commit job changes unless the user asks. `jobs/*.yaml` and `.colab-train/` are local.
- One Colab session per account. Queue jobs with `submit` and let `worker` start them. Never click "Manage sessions"
  or "Connect" in the browser; the tool dismisses "Too many sessions" by Cancel only.

## Commands

Run from the repo that holds the job (or with the job path). `JOB` is a preset name (`example-pytorch`,
`cleanrl-ppo-cartpole`, `sb3-ppo`, `mjlab-rsl-rl`), a file under `jobs/`, or a YAML path.

| step | command | expected output (abbreviated) |
|---|---|---|
| list | `colab-train list` | `presets (built in):` then names |
| new job | `colab-train init NAME --repo URL --train CMD [--install CMD] [--wandb yes\|no] [--entity E] [--gpu T4\|none] [--token-secret NAME] [--progress-regex R] [--checkpoints GLOB] --yes` | `wrote jobs/NAME.yaml` and the `next:` steps. Without `--yes` in a terminal it asks instead. Refuses to overwrite without `--force`. Exit 2 with `error:` when the train command or repo is missing |
| new job from preset | `colab-train init NAME --preset example-pytorch` | `wrote jobs/NAME.yaml (from preset ...)` |
| doctor | `colab-train doctor [JOB ...] [--no-network] [--run-name N]` | one `[ok]`/`[warn]`/`[FAIL]` line per check with `fix:` hints. Checks Python, Chrome, the profile dir (not opened), secret NAMES in the local store (never values), each job spec, and `git ls-remote <url> HEAD`. Exit 1 when a check fails |
| check | `colab-train validate JOB` | `ok: <job>  run <run-id>  W&B <entity>/<project>` plus `install:` and `train (fresh start):` lines. Exit 2 with `error: ...` lines when invalid |
| notebook only | `colab-train render JOB --out build/JOB.ipynb` | `wrote /abs/path.ipynb` |
| launch | `colab-train launch JOB [--run-name colab-2] [--publish auto\|gist\|local]` | `job ... run ...`, `notebook: /abs/...`, `one-click URL: https://colab.research.google.com/gist/...` (only if published), then the numbered Colab steps |
| status | `colab-train status JOB [--json]` | `run ... state=running\|finished\|crashed`, `iter N / MAX`, `speed X s/iter  ETA ~H h (running)`, `ckpts`, `videos`, `wall` |
| logs | `colab-train logs JOB --lines 60` | last lines of the W&B `output.log` (ANSI stripped) |
| resume | `colab-train resume JOB` | `still running` (do not restart), `finished` (nothing to do), or the notebook URL to Run all again |
| stop | `colab-train stop JOB` | prints the browser step (Runtime > Disconnect and delete runtime). The CLI cannot stop Colab |
| browser start | `colab-train browser start` | foreground: `browser ready: ... on 127.0.0.1:PORT`. Human runs it in a desktop terminal and leaves it open |
| browser status | `colab-train browser status` | `port N pid P profile ...: running` (exit 1 when not started) |
| browser setup | `colab-train browser setup [JOB] [--update]` | signs in (human), then with JOB fills the job's Secrets in its notebook tab from the local store. Ends with `ok: secrets set in Colab` |
| secrets list | `colab-train secrets list` | names, source, backend, time, `set=yes`. Never values |
| secrets import | `colab-train secrets import [--from-file PATH --name N \| --git-host H --name N]` | W&B key from `~/.netrc` or env when no flags. Prints the source only |
| secrets set | `colab-train secrets set NAME` | human, in a terminal: hidden input. You do not run this with a value |
| secrets rm | `colab-train secrets rm NAME` | removes the local value only |
| launch --browser | `colab-train launch --browser [JOB] [--auto] [--force]` | `chose JOB (...)`, `runtime type: ... <runtime.gpu_type> GPU`, first `[colab-train] ...` lines, `next: colab-train status JOB`. Exit 3 = refused or safety stop |
| stop --browser | `colab-train stop JOB --browser [--yes]` | `runtime of JOB disconnected and deleted`. Asks first unless `--yes` |
| submit | `colab-train submit JOB [--run-name N] [--priority P] [--notify-pane PANE]` | `queued ID: JOB run RUN ...`. Adds the job to the shared queue; it starts when the one session is free |
| queue | `colab-train queue [--json]` | table: id, status (`queued`, `running`, `stalled`, `finished`, `failed`, `cancelled`, `needs-human`), priority, job, run, iter, s/it, ETA, W&B. Empty: `queue is empty` |
| cancel | `colab-train cancel ID [--yes]` | queued: `cancelled`. running: browser stop of that runtime, then `cancelled`. Asks first unless `--yes` |
| worker | `colab-train worker [--once] [--interval S]` | foreground loop: reconciles running items, starts the next queued job. `--once` does one pass and exits |

`status` returns exit 0 also when no run exists yet: it prints `no W&B run ending in '<run id>' ... yet`. That means the
human has not pressed Run all, or the notebook has not reached training.

## Standard flow

1. `colab-train validate JOB`. Fix errors before anything else.
2. `colab-train launch JOB`. Give the human the printed URL (or the notebook path and the upload steps) and the
   list of secret names to enable. Tell them the GPU is set to T4 in the notebook, and to check it.
3. Wait for the human to say they pressed Run all. Then poll: `colab-train status JOB`. Poll every 5 to 10 minutes,
   not faster. Report: iteration, s/iter, ETA, state.
4. If `state=crashed`, or the human says the tab disconnected: `colab-train logs JOB --lines 80`. Diagnose from the
   log (see failure modes). Then `colab-train resume JOB` and give the human the URL to Run all again. The runner
   resumes from the newest checkpoint on W&B (or Drive).
5. Done when `state=finished` and the last log line is `[colab-train] Done.`. Report the final iteration and the W&B
   URL.

## Browser flow (launch --browser)

One time, by the human:

1. In a desktop terminal: `colab-train browser start`. Leave it open. (The CLI starts the installed Chrome with a
   dedicated profile and a debugging port on 127.0.0.1. Playwright must be installed: `colab-train[browser]`.)
2. Secrets: the values must be in the local store (`colab-train secrets list`). For each missing name, the human runs
   `colab-train secrets set NAME` in a terminal. Or you run `colab-train secrets import` for the W&B key.
3. `colab-train browser setup JOB`. The human signs in to Google in that window (only the first time). Then the tool
   fills the job's Secrets in the notebook tab from the store and turns notebook access on. Nothing is shown, and no
   screenshot is taken while the Secrets panel is open. It ends with `ok: secrets set in Colab`. Use `--update` only
   when the human asks to replace an existing value.

Per run, by you:

1. `colab-train validate JOB`. Fix errors first.
2. `colab-train browser status`. If it is not running, ask the human to start it (step 1 above). Check the secret
   names with `colab-train secrets list`; `launch --browser` refuses and names any value that is missing locally.
3. `colab-train launch --browser JOB` (or without JOB: `--auto` picks from the priority list and a new run name, and
   says what it chose). Show the human the printed lines.
4. Poll `colab-train status JOB` every 5 to 10 minutes, as in the manual flow.
5. To finish: `colab-train stop JOB --browser` (the human confirms, or pass `--yes` when the human asked you to stop).

### What to do on a safety stop

`launch --browser` prints `STOP: <reason>` and a screenshot path, then exits 3. Nothing was clicked past.

| reason | what it means | what you tell the human |
|---|---|---|
| `captcha` | Google asks to prove the user is human | Solve it yourself in the browser window, then run `launch --browser` again. Do not automate it. |
| `robot check` (`are you a robot`, `are you still there`) | Colab's idle or bot check | Click through it yourself if it is still there, then `status`. A repeated check means the runtime is idle: stop it. |
| `usage limit` (`cannot connect to a GPU backend`) | the free GPU quota is used up for now | Wait for the limit to reset (hours). Do not retry in a loop. |
| `google sign-in page` / `not signed in` | the browser is not signed in to Google | Ask the human to sign in in that window, then `browser setup --check`. |
| `refused: ... already has a browser runtime` / `W&B run ... is already running` | a runtime exists | `status` first. Stop the old runtime with `stop --browser`, or use `--force` only when the human says so. |

## Queue and worker (shared Colab jobs)

The account runs one Colab session at a time. Use the queue instead of starting runtimes yourself:

1. `colab-train submit JOB --run-name NAME --priority P` adds the job. Nothing starts yet.
2. `colab-train worker` runs in the foreground (its own terminal or tab). Each pass it reconciles the running
   item (W&B progress, the browser tab) and starts the next queued job when the session is free. `--once` does one
   pass and exits; use it to check the state without waiting.
3. `colab-train queue` shows the state. `colab-train cancel ID` removes a queued job, or stops a running one in the
   browser (asks first; pass `--yes` only when the human asked you to stop it).

Worker rules (defaults in `.colab-train/config.json`): a stall (no new iteration for 15 min) is reported, not
restarted; a disconnect re-queues the same run name at most 2 times; a usage limit or no-GPU stop pauses the queue for
2 h; "Too many sessions" re-queues the job and waits 10 min; captcha, sign-in and robot checks mark the item
`needs-human`. The worker never clicks past a safety stop. Notifications go to the events log and, when set, to
`notify_cmd`. Keep `max_parallel` at 1.

Never start a second runtime by hand while the worker or a running item holds the session.

## Reading the progress

- `iter N / MAX`: last logged iteration (the spec's `wandb.iteration_key`, default `_step`).
- `speed`: seconds per iteration over the last 20 logged points. Computed from W&B history, so it is only available
  after a few points.
- `ETA`: remaining iterations times speed. Shown only while `running`. A free runtime lasts at most ~12 h, and the
  runner stops training after `limits.max_hours` (default 10), so a long run needs several resumed sessions.
- `ckpts`: checkpoint files on W&B. If it stays `0` after training has run for a while, the training code is not
  uploading checkpoints, so resume will start fresh. Check the spec's `resume.checkpoint_glob`.
- `videos`: mp4 files on the run. Zero is normal when `video.every` is 0 or MuJoCo has no GL backend.

## Failure modes and fixes

| symptom (in cell output or `logs`) | cause | fix |
|---|---|---|
| `Colab secret X is missing or Notebook access is off` | secret not in Colab, or notebook access off | the human runs `colab-train secrets set X` (if the local value is missing), then `colab-train browser setup JOB`, then Run all |
| `environment variable X is empty` | same, the secret exists but is empty | Human: put the value in the secret |
| `git ... Authentication failed` / `repository not found` | token has no read access, or wrong `repo.token_user` | Human: create a token with read access to the repo; check `token_user` in the job |
| `train.command: required` / `unknown key` / `did you mean` | the job YAML is wrong | fix the key named in the message; `colab-train validate JOB` again |
| `resume.checkpoint_glob: required` | resume is on but no checkpoint pattern | set `checkpoint_glob` and `args`, or `resume.source: none` |
| `wandb.entity: ...` | `wandb.mode: online` without an entity | set `wandb.entity`, or `mode: disabled` (no W&B account needed) |
| `repo.token_secret: the NAME of a Colab secret` | a token value was put in the job | never put tokens in a job. Human: `colab-train secrets set NAME`, then keep the NAME in the job |
| `repo not reachable` from doctor | wrong URL, private repo without a token, or a token without read access | check the URL in a browser; add `repo.token_secret` for a private repo; human: `colab-train secrets set NAME` |
| `no GPU attached ... assigns a L4` | Colab gave a CPU runtime or another type | Human: Runtime > Disconnect and delete runtime, then Run all; or set `runtime.gpu_type` / `accelerator: none` |
| `wandb is not installed` | `status`/`logs`/worker need the extra for a W&B job | `uv tool install --force 'colab-train[wandb,browser] @ git+https://github.com/Barathaner/colab-gpu-trainer.git'` (jobs with W&B off do not need it) |
| `uv: command not found` in install | `uv` not installed in Colab | the runner installs uv through pip when an install line starts with `uv`; if it still fails, check the cell output above it |
| `MUJOCO_GL=disable (videos turned off)` | no EGL or OSMesa | informational; training runs without videos |
| `CUDA out of memory` | `num_envs` too high for the T4's 15 GB | `colab-train init` a copy with a lower `num_envs`, or pass `--run-name`; keep checkpoints of the old run |
| `training exited with N` | the training command failed | read the traceback in `logs`; fix in the repo, not in the notebook |
| `W&B resume lookup failed` | W&B unreachable | the run started fresh; if it is a real run, stop and retry after the network is back |
| disconnected / `Time budget ... reached` | normal session end | Run all again (or `resume`) |
| `already reaches max_iters` | nothing left to train | raise `train.max_iters` in a new job copy and launch with the same `--run-name` |
| `no W&B run ... yet` | Run all not pressed, or the run is not started | ask the human to check the notebook |

## What you must never do

- Print, log, or commit secret values, or write a token into a YAML, a notebook, a command, or a commit message.
- Use the local GPU for training.
- Keep the runtime alive, reconnect automatically, or click the Colab UI beyond the `--browser` steps.
- Sign in to Google, type a password or 2FA code, read or type a Colab Secret value, or continue past a captcha,
  robot check, usage-limit dialog or sign-in page.
- Start a second runtime for the same run id while the first is running.
- Publish a gist with a notebook for a private job without the user's consent. The default is `--publish local`;
  pass `--publish auto` or `--publish gist` only when the user has agreed to a secret GitHub gist.
