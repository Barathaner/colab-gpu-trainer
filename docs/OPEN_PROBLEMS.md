# Open problems and ideas

State as of 2026-10-21. Ordered by how much they matter to a user.

## Open problems

1. **Paths not yet tested live on Colab.** The unit tests and a fake Colab page cover them, but no real run has used:
   a GPU other than the T4 (L4 and A100 need a paid plan), `accelerator: none`, Drive as the resume source, the
   CleanRL and SB3 presets' progress regex on real output, and the worker's notify hook. A live end-to-end run of the
   `mjlab-rsl-rl` preset through `submit` and `worker` (two jobs, serial, with one resume) is planned next.
2. **Browser mode depends on Colab's page.** Labels, shadow DOM and output iframes can change without notice. The tool
   is tested against English labels (`?hl=en`) and stops with a screenshot when a step fails, but a Colab UI change
   needs a code fix.
3. **The daily GPU quota is invisible until launch.** Colab publishes no quota and no API for it. A used-up quota shows
   up only as a refused GPU after `launch`. The worker then backs off (2 h by default) and retries.
4. **Progress without W&B needs the Colab tab.** For a job with `wandb.mode: disabled`, `status`, `logs` and the worker
   read the runner lines from the saved browser tab. If that tab is closed or was opened by hand, they can only report
   "unknown", and the worker's stall timer is the only signal.
5. **Colab Secrets are set by hand.** There is no API for the Secrets panel, so each secret name has to be added once
   in Colab with Notebook access on. `doctor` checks local values and repo access, not what Colab holds.
6. **Fine-grained GitHub tokens read only the repos they list.** A new private repo needs to be added to the token's
   Repository access on GitHub; `doctor` reports a 403 and names this fix.

## Ideas

- Show the newest benchmark result in `status` when the job has a `bench` step.
- Queue priorities and a per-user limit for shared accounts (today the queue is first in, first out).
- A `doctor` check that Colab holds the secret names (would need the browser mode).
- Record the GPU type and the speed (s/iter) per run in the queue history, to plan the next runs.
- Detect the "GPU quota used up" page in browser mode and report it as its own state instead of a generic safety stop.
