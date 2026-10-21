# colab-gpu-trainer

Train any repo on free Google Colab GPUs from your terminal (queue, resume, W&B optional).

`colab-train` turns a small YAML job spec into a Colab notebook that clones your repo, installs its dependencies,
runs your train command on a GPU, and resumes from the newest checkpoint after a disconnect. It works with any
training command: PyTorch, Stable-Baselines3, CleanRL, mjlab/rsl_rl, or a script of your own. W&B is optional.

Guides: [docs/AI_USAGE.md](docs/AI_USAGE.md) (for an AI agent driving the CLI),
[docs/AI_SETUP.md](docs/AI_SETUP.md) (one-time setup for a new person), and the Claude Code skill in
[skills/colab-train/SKILL.md](skills/colab-train/SKILL.md).

## What it is

- A CLI, `colab-train`, that validates a job, renders the notebook, and tells you the steps (or drives your own
  Chrome for the upload, GPU choice and Run all, see [Browser mode](#browser-mode-optional)).
- A runner that the notebook embeds. It is standard library only and does the work on Colab: clone, install,
  train with resume, optional benchmark.
- A local job queue and worker, so several jobs can wait for the one Colab session you are allowed.
- No credentials are needed to install or to validate and render a job.

## Install

You need Python 3.10+, git, and a Google account for Colab. Install with [uv](https://docs.astral.sh/uv/):

```bash
# with W&B status/logs and the browser mode:
uv tool install --force 'colab-train[wandb,browser] @ git+https://github.com/Barathaner/colab-gpu-trainer.git'
# minimal (no W&B, no browser):
uv tool install --force 'colab-train @ git+https://github.com/Barathaner/colab-gpu-trainer.git'
```

The `wandb` extra is needed only for jobs that log to W&B (`status`, `logs`, the worker). The `browser` extra is needed
only for `--browser`. `colab-train doctor` tells you when an extra is missing. No token or login is needed to install.

## Quick start

```bash
cd your-repo
colab-train init                            # asks for the repo URL, install steps, train command, W&B; writes jobs/<name>.yaml
colab-train doctor jobs/<name>.yaml         # local setup, job spec, secret names, repo URL (HEAD only)
colab-train launch jobs/<name>.yaml         # renders the notebook, prints the Colab URL and the steps
colab-train status <name>                   # run state from W&B, or from the Colab tab (--browser) when W&B is off
```

`init` also accepts flags, so it can run without questions:

```bash
colab-train init --yes --repo https://github.com/you/your-repo.git --train "python train.py --steps {iters}"
```

Other useful commands: `list` (presets and jobs), `validate JOB` (checks the spec and prints the resolved train
command), `render JOB --out build/job.ipynb` (notebook only), `logs JOB`, `resume JOB`, `stop JOB`.

`launch` writes `.colab-train/notebooks/<job>-<run>.ipynb`. With `--publish auto` or `--publish gist` it also publishes
the notebook as a secret GitHub gist when `gh` is logged in; the default `--publish local` keeps it on disk. Then
upload it to Colab (File > Upload notebook), set the GPU, add the Secrets, and press **Run all**. Keep the tab open.
After a disconnect, press Run all again: the run resumes from its newest checkpoint.

### Choosing a starting point

- `colab-train init` writes a job for your own repo.
- `colab-train init my-job --preset example-pytorch` copies a built-in preset to edit.
- Built-in presets (`colab-train list`): `example-pytorch` (generic PyTorch repo, no W&B),
  `cleanrl-ppo-cartpole` (public repo, pip, CPU, no W&B), `sb3-ppo` (Stable-Baselines3 with Drive checkpoints), and
  `mjlab-rsl-rl` (an mjlab + rsl_rl uv project with W&B; placeholder repo, task id and entity to replace).

## Supported setups

- **Any train command.** `train.command` is a template with placeholders such as `{iters}`, `{run_id}` and
  `{resume_args}`. The runner fills them in. See the placeholder list under [Job spec](#job-spec).
- **Progress.** With W&B, progress comes from the run. Without W&B, set `progress.regex` to a line your trainer
  prints, for example `step (?P<iter>\d+)/(?P<total>\d+)`. The runner logs the progress it parses.
- **Resume.** Checkpoints can be kept on Google Drive (`resume.source: drive`, consent asked once) or as W&B run
  files (`resume.source: wandb`, needs W&B online). `resume.checkpoint_glob` says which files are checkpoints and
  `resume.args` says how to pass the newest one to the trainer. Without checkpoints, a disconnect starts the run over.
- **Iteration counting.** `resume.iters_mode: total` (default) passes `train.max_iters` unchanged on resume.
  `remaining` passes `max_iters` minus the loaded iteration, for loops that restart at the loaded iteration and run
  their iteration count again. Set it explicitly for your framework.
- **Framework-specific options** (all off by default, set explicitly in the example preset `mjlab-rsl-rl`):
  `resume.experiment_config_key` (W&B config key whose folder names `{experiment}` in `resume.restore_dir`) and
  `resume.wandb_config_shim` (lets a resumed W&B run accept the config change that some training loops make).
- **Accelerator.** `runtime.accelerator: GPU` with `runtime.gpu_type` T4 (the usual free-tier assignment), L4, A100 or
  H100; or `accelerator: none` for CPU-only jobs. A GPU job with no GPU attached stops with a message.
- **Optional benchmark.** `bench.command` runs after training on the newest checkpoint.
- **Private repos.** `repo.token_secret` names a Colab Secret that holds a git token. An optional second host
  (`repo.fallback_url`) is tried only when the primary clone or fetch fails.

## Browser mode (optional)

`launch --browser` and `stop --browser` drive **your own Chrome** through the Chrome DevTools Protocol, with Playwright
attached over CDP (Google sign-in refuses Playwright's bundled browser). It is a convenience for the clicks, not a
way around Colab's rules.

1. **Start the browser** in a desktop terminal and leave it open: `colab-train browser start`. It runs Chrome with a
   dedicated profile under `~/.cache/colab-train/` (never your everyday profile) and a debugging port on `127.0.0.1`.
2. **Sign in once, yourself**: `colab-train browser setup JOB`. It opens Colab in that window and lists the Secret
   names the job needs. You type your password and 2FA code only in the browser, and create the Secrets yourself.
   The tool never asks for, types, stores or reads a password, a 2FA code or a secret value.
3. **Launch**: `colab-train launch --browser JOB`. It uploads the notebook, sets the T4 GPU, presses **Run all**, and
   confirms only the two approved dialogs (a notebook not authored by Google, and a Secret access grant). Then it
   waits for the first `[colab-train]` lines.
4. **Stop**: `colab-train stop JOB --browser` clicks Runtime > Disconnect and delete runtime after you confirm.

**Safety stops.** On a captcha, a robot check, a usage-limit dialog or a Google sign-in page, the tool stops, saves a
screenshot to `.colab-train/screens/` and exits with code 3. It never clicks past these. Keep-alive, auto-reconnect,
captcha solving and a second runtime are not implemented.

**Choosing the job without a name.** With `--auto` (or when stdin is not a terminal), `launch --browser` picks the
first available job from the `auto_jobs` list in `.colab-train/config.json`, or the first valid job when that list is
absent, and says what it chose. A refused launch exits with code 3 when another job or this one has a browser runtime
recorded, or when W&B shows a running run. `--force` overrides both checks.

Config keys in `.colab-train/config.json`: `browser` (a path or a list of paths; `COLAB_TRAIN_BROWSER` wins),
`auto_jobs`, and the queue keys below.

## Job queue and worker

A free Colab account runs one session at a time. Extra sessions are refused. So jobs go into a local queue and one
worker starts them in turn:

```bash
colab-train submit jobs/<name>.yaml --run-name trial-2 --priority 10 --notify-pane 2
colab-train queue                  # queued, running, finished; --json for scripts
colab-train cancel 3 --yes         # queued: marked cancelled; running: browser stop, then cancelled
colab-train worker                 # foreground; run it in its own terminal (--once: one pass, then exit)
```

- The queue is one file, `~/.cache/colab-train/queue.json` (mode 600, no secrets; override with `COLAB_TRAIN_QUEUE`).
  A higher `--priority` runs first.
- The worker checks running items every `interval_s` (default 120) against W&B and the browser tab. A run that finishes
  is marked finished. No new iteration for `stall_min` (15 minutes) marks it stalled. A crash, a runner error or no
  W&B run 45 minutes after launch marks it failed. A disconnect re-queues the job as a fresh launch with the same run
  name, up to `max_resumes` (2 by default), so the newest checkpoint is restored.
- A usage-limit or no-GPU stop pauses the queue for `backoff_h` (2 hours). A "too many sessions" refusal re-queues the
  job and pauses for 10 minutes. A captcha, sign-in or robot check marks the item `needs-human`. The worker clicks
  nothing.
- Each change is appended to `~/.cache/colab-train/events.log` (override `COLAB_TRAIN_EVENTS`). If
  `.colab-train/config.json` has `notify_cmd`, it runs with `{pane}` replaced by the item's `--notify-pane` (or
  `error_pane` for failures) and the event line as the last argument. It is run without a shell.
- Config keys in `.colab-train/config.json`: `max_parallel` (default 1; keep it at 1 on the free account),
  `stall_min`, `max_resumes`, `backoff_h`, `interval_s`, `notify_cmd`, `error_pane`. Example `notify_cmd`:
  `"herdr agent prompt {pane}"`, or any command of your own.
- Browser runs need the dedicated Chrome from `browser start`, so start it first.

## Secrets

- A job file holds Colab Secret **names** only (`repo.token_secret: GITHUB_TOKEN`, `wandb.api_key_secret:
  WANDB_API_KEY`, and `secret_env: {ENV_VAR: SECRET_NAME}`). Secret values are never accepted in a job, a URL or a
  notebook. A value-looking `token_secret` is refused with a message that names the fix.
- Store the values locally, once, with a hidden prompt: `colab-train secrets set GITHUB_TOKEN`. `secrets import`
  can take the W&B key from `~/.netrc` or the environment, and `secrets import --from-file PATH --name NAME` takes a
  value from a file. Values are kept in `~/.config/colab-train/secrets.json` (mode 600, directory 700), or in the OS
  keyring when one is available. `secrets list` shows names and whether each is set, never values. `secrets rm NAME`
  removes a local value.
- In Colab, add the same names under the key icon with **Notebook access** ON. `browser setup JOB` can fill the
  Colab Secrets panel from the local store. It does not read the page back, prints no value, and leaves existing
  secrets alone unless `--update` is given. `launch --browser` refuses before it opens the browser when a needed
  value is missing locally.
- The runner passes a git token to git through a credential helper and an environment variable. The token is not
  printed and is not put in a URL.

## For AI agents

[docs/AI_USAGE.md](docs/AI_USAGE.md) lists the commands, their exit codes and the errors an agent can fix by
following the message. [docs/AI_SETUP.md](docs/AI_SETUP.md) is the one-time setup for a person or an agent.
[skills/colab-train/SKILL.md](skills/colab-train/SKILL.md) is a Claude Code skill built on the same rules. Agents
must not handle passwords, 2FA codes or secret values, and must not add keep-alive or scripted clicking.

## Job spec

A job is a YAML file (see `colab_train/presets/`). The main keys:

| key | meaning |
|---|---|
| `name`, `description` | job name (lowercase, dashes); used for the run id and the state files |
| `repo.url`, `repo.branch`, `repo.dir`, `repo.submodules` | git source (https only, no credentials in the URL) |
| `repo.token_secret`, `repo.token_user` | Colab Secret name for a private repo, and the user name git expects with it |
| `repo.fallback_url`, `repo.fallback_token_secret`, `repo.fallback_token_user` | optional second host, used only when the primary fails |
| `python` | optional uv-managed Python version, e.g. `"3.13"`; null uses Colab's Python |
| `install` | shell commands run in the repo, e.g. `uv sync --frozen` or `pip install -r requirements.txt` |
| `mujoco_gl` | probe EGL, then OSMesa, for MuJoCo rendering; videos are turned off if neither works |
| `env`, `secret_env` | plain environment variables; variables filled from Colab Secrets |
| `wandb` | `mode` (auto / online / offline / disabled; auto = online when `entity` is set), `entity`, `project`, `api_key_secret`, `iteration_key` |
| `progress.regex` | regex with one group (or `(?P<iter>...)`) for the iteration, optional `(?P<total>...)`; works without W&B |
| `run` | `prefix` and `name` form the run id `<prefix>-<name>`; `--run-name` overrides `name` |
| `train` | `command` template, `task`, `num_envs`, `max_iters`, `save_every` |
| `video` | `every` (0 = off), `steps_per_iter`, `length`, `flags` (added to the train command) |
| `resume` | `source` (auto / wandb / drive / none), `checkpoint_glob`, `checkpoint_regex`, `restore_dir`, `args`, `iters_mode` (total / remaining), `experiment_config_key`, `wandb_config_shim`, `sync_extra` |
| `bench.command` | optional command run after training on the newest checkpoint |
| `limits.max_hours` | training gets a clean stop (SIGINT) after this many hours; default 11, 0 to 24 allowed |
| `runtime` | `accelerator` (GPU or none) and `gpu_type` (T4, L4, A100, H100); chosen in the runtime dialog |

Placeholders: `{task} {num_envs} {iters} {max_iters} {save_every} {run_id} {run_name} {wandb_entity} {wandb_project}
{repo_dir} {video_flags} {resume_args}` (train command), `{resume_run} {resume_checkpoint} {resume_path} {resume_iter}`
(resume args), `{experiment} {run_name} {run_id}` (restore_dir), `{checkpoint} {bench_json}` (bench). `validate`
rejects unknown placeholders and unknown keys, with the closest valid name as a hint.

Resume sources: with `wandb`, the trainer must upload its checkpoints as W&B run files. With `drive`, checkpoints are
mirrored to `My Drive/colab-train/<run id>/` and the newest one is passed through `resume.args`. With `none`, every run
starts fresh.

## How a run works on Colab

The notebook embeds the runner and the job as JSON. **Run all** does this: GPU report; Colab Secrets to environment
variables (values are never printed); optional Drive mount; clone at the branch; the `python` version through uv; the
`install` commands; the MuJoCo GL probe when `mujoco_gl` is set; restore the newest checkpoint; train with the resume
arguments and copy new checkpoints to Drive every 60 seconds (Drive resume only); optional benchmark. Training output is echoed and parsed with
`progress.regex`. A disconnect ends the cell: reconnect, press Run all, and the run resumes.

## Limits

- **Colab's terms and free-tier limits apply.** Free sessions are time-limited, idle sessions time out, and GPU
  availability and daily quotas are not published and change. Check Colab's current terms before you rely on this. Colab
  prioritizes interactive use, so long unattended training is outside the free tier's intent. This tool keeps a person
  in the loop and adds no keep-alive.
- **One Colab session per account** at a time. The queue waits for it; it does not work around it.
- **Browser mode depends on Colab's page.** Button labels and layout can change and break the automation. The tool
  is tested against English labels (`?hl=en`). When something changes, the tool stops and the screenshot shows where.
- **No multi-account or multi-runtime tricks**, no SSH or tunnels, no captcha solving.
- Colab's Python version and preinstalled packages change over time. Pin your environment in `install`.

Open problems and ideas: [docs/OPEN_PROBLEMS.md](docs/OPEN_PROBLEMS.md).

## Development

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH= uvx --with pyyaml --with wandb --with nbformat --with playwright pytest -q tests
```

The browser tests use a fake Colab page in headless Chrome over CDP and skip themselves when Playwright or a Chrome
binary is missing. `tests/test_no_personal_data.py` checks that no tracked file contains private words.

## License

MIT. See [LICENSE](LICENSE).
