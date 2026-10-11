JEV: AI-Assisted Model Finetuning
=================================

> **Status: early exploration.** This repo is a place to find out how far an AI coding agent can speed up
> model finetuning and training work. The scaffolding (workflow, experiment tracking, a finetuning smoke test)
> is in place and was built with Claude Code. No real model has been trained yet, and nothing here has been
> reviewed or hardened beyond the smoke test described below.

Project summary
---------------

JEV combines three tools so that an AI agent can carry model-development work from idea to tracked experiment
while a human approves each step:

- **AI-DLC** (from AWS) gives the agent a structured workflow: requirements, design, code, test, with approval gates.
- **ClearML** records every run, so nothing the agent does goes untracked.
- **Unsloth** does the finetuning itself.

The first target is a transaction decision model, which takes transaction features and outputs
approve, decline or review. That use case is why the ClearML project is named `jev-transaction-decision-model`.

```
idea --> /aidlc workflow --> code (jev/) --> Unsloth finetune --> ClearML task --> review
         (human approves      (agent writes)   (local or GPU      (config, loss,    (human
          each stage)                           machine)           artifacts)        decides)
```

What we want to learn
---------------------

The point of the repo is the questions it lets us answer:

- **Speed:** how much faster is a finetuning experiment, from idea to tracked result, with an agent than by hand?
- **Quality:** where does the agent need correcting, and which approval gates catch the mistakes?
- **Cost:** how much agent effort goes into setup compared with actual training?
- **Limits:** which parts (hardware quirks, data judgment) still need a person?

No measurements have been collected yet. When they are, they will go in a section here, with the method used.

What it does
------------

- **Structured agent workflow:** AI-DLC stages and approval gates are installed in `.claude/` and `aidlc/`. Start work with `/aidlc <description>` in Claude Code.
- **Tracked runs:** `jev/tracking.py` opens a ClearML task in the project and records the full run config before any work starts.
- **Local finetuning:** `jev/unsloth_smoke.py` runs a LoRA finetune of a 0.6B model on a few made-up examples. It checks the toolchain works end to end.
- **Versioned history:** the trunk-based git workflow is in `aidlc/spaces/default/memory/org.md`, and the AI-DLC audit trail is committed with the code.

Architecture
------------

| Piece | Role | Talks to |
|---|---|---|
| `.claude/` | AI-DLC skills, agents, hooks and settings for Claude Code | Claude Code, the `aidlc` CLI |
| `aidlc/` | Workflow records, team rules and audit trail for each piece of work | `.claude/` hooks |
| `jev/tracking.py` | Opens a ClearML task and connects the run config | ClearML server (`app.clear.ml`) |
| `jev/unsloth_smoke.py` | Finetuning smoke test | Unsloth, Hugging Face (model download), `tracking.py` |
| `.venv/` | Python 3.9 environment with `clearml` only | ClearML |
| `.venv-unsloth/` | Python 3.12 environment with Unsloth, PyTorch and ClearML | Unsloth, Hugging Face |

Both virtualenvs are local only and are not committed.

Prerequisites
-------------

- **Claude Code**, with the AI-DLC CLI installed (`aidlc` on your PATH). Install it from the [awslabs/aidlc-workflows](https://github.com/awslabs/aidlc-workflows) releases.
- **A ClearML account** and API credentials from your workspace settings.
- **Python 3.12** for the Unsloth environment. [`uv`](https://docs.astral.sh/uv/) is the easiest way to get it.
- **Hardware:** the smoke test was run on an Apple M2 Max with 64 GB. NVIDIA GPUs are the main Unsloth target and should be used for anything larger.

Quick Start
-----------

1. **Clone the repo and enter it.**

   ```bash
   git clone https://github.com/jax79sg/jev.git
   cd jev
   ```

2. **Add your ClearML credentials.** Put them in `~/clearml.conf`, outside the repo, and never commit them. Create the file from the ClearML workspace settings page (Settings, Workspace, Create new credentials).

3. **Create the Unsloth environment.**

   ```bash
   uv venv .venv-unsloth --python 3.12
   VIRTUAL_ENV=.venv-unsloth uv pip install -r requirements-unsloth.txt --torch-backend=auto
   ```

4. **Run the smoke test.** It downloads `unsloth/Qwen3-0.6B` (about 1 GB) on first run.

   ```bash
   .venv-unsloth/bin/python -m jev.unsloth_smoke
   ```

   A successful run prints a falling loss for 10 steps and a ClearML results link. It only shows the pipeline works. The model learns nothing useful from 32 made-up rows.

5. **Check the AI-DLC setup, then start work.**

   ```bash
   aidlc doctor
   claude
   ```

   In Claude Code, run `/aidlc <what you want to build>`. The agent will ask questions and stop for your approval at each stage.

Apple Silicon notes
-------------------

On a Mac, Unsloth does not use its CUDA path. It detects Apple Silicon and trains through MLX. In practice:

- Some standard Hugging Face training options are rejected. `use_cpu` raised an error, so check each option against what the backend accepts.
- CUDA-only features such as bitsandbytes 4-bit loading may not be available. The smoke test loads full-precision weights.
- Treat the Mac as a place to test pipelines. Run real training on a CUDA machine, using `clearml-agent` and queues.

Configuration Reference
-----------------------

- `~/clearml.conf`: ClearML server addresses and credentials. Not stored in the repo.
- `jev/tracking.py`, `PROJECT_NAME`: the ClearML project all runs go to (`jev-transaction-decision-model`).
- `jev/unsloth_smoke.py`, `config` dict: model name, sequence length, LoRA rank and alpha, steps, learning rate, batch size. It is connected to the ClearML task, so each run records what it used.
- `aidlc/spaces/default/memory/*.md`: team and project rules the agent follows. Edit these to change how the agent works.

Development
-----------

```bash
# tracking-only environment
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# training environment
uv venv .venv-unsloth --python 3.12
VIRTUAL_ENV=.venv-unsloth uv pip install -r requirements-unsloth.txt --torch-backend=auto
```

There is no automated test suite or linter yet. The smoke test is the only check. Both belong on the list of things to add once real code exists.

Working rules for the agent:

- Every training or evaluation run opens a ClearML task through `jev/tracking.py`.
- Dataset changes become new ClearML dataset versions, not in-place edits.
- Work is merged to `main` in small steps (trunk-based).

Troubleshooting
---------------

**`aidlc: command not found`**: open a new terminal, or add `~/.local/bin` to your PATH.

**ClearML says the credentials are wrong, or tasks land in the wrong workspace**: check `~/clearml.conf`. A leftover file from another project is the usual cause.

**`NotImplementedError: Unsloth MLX: unsupported TrainingArguments/SFTConfig kwargs`**: you are on a Mac and passed an option the MLX backend does not implement. Remove it.

**Import errors under Python 3.9**: the Unsloth environment needs Python 3.12. Use `.venv-unsloth/bin/python`, not the system Python.

**Model download is slow or fails**: the first run downloads from Hugging Face. Rerun it, because partial downloads resume.

Project Structure
-----------------

```
jev/
├── README.md
├── jev/
│   ├── tracking.py          # ClearML task helper used by every run
│   └── unsloth_smoke.py     # LoRA finetuning smoke test
├── requirements.txt         # tracking-only environment
├── requirements-unsloth.txt # training environment
├── .claude/                 # AI-DLC skills, agents, hooks, settings
└── aidlc/                   # workflow records, team rules, audit trail
```

Local only, ignored by git: `.venv/`, `.venv-unsloth/`, `outputs/`.

Further reading
---------------

- [AI-DLC workflows](https://github.com/awslabs/aidlc-workflows)
- [Unsloth](https://github.com/unslothai/unsloth)
- [ClearML docs](https://clear.ml/docs)
