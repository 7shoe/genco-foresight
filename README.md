# GENCO Foresight
Post-Training the Grid Solver of Tomorrow

## Installation

Expand the section for your machine.

<details>
<summary><strong>Aurora — Intel XPU</strong></summary>

Use Aurora’s provided PyTorch environment:

```bash
# current aurora env
# module load frameworks/2026.1.0

# all-purpose venv
source /home/siebenschuh/Projects/reasonABLE/env/Aurora/activate_reason_venv.sh

```

Check accelerator access on an allocated compute node:

```bash
python -c "import torch; print(torch.xpu.is_available())"
```

Set `device: xpu` in `config.yaml`.

</details>

<details>
<summary><strong>NVIDIA — CUDA</strong></summary>

Create an environment:

```bash
TBD
```

Install CUDA-enabled PyTorch using the
[official installation selector](https://pytorch.org/get-started/locally/).

Check accelerator access:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Set `device: cuda` in `config.yaml`.

</details>
