# MODEL-AGNOSTIC ADVERSARIAL ATTACK AND DEFENSE FOR VISION-LANGUAGE-ACTION MODELS

This project contains the official implementation of our work "Model-Agnostic Adversarial Attack and Defense for Vision-Language-Action Models".

## 🛠️ Pre-requisites

Before running this repo, you need to install several dependent repositories:
**Important:** All sub-repositories should be cloned and installed **inside the root directory of this project**.

### 1️⃣ LIBERO Simulation Benchmark
Clone and install the LIBERO repository:
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

### 2️⃣ VLA Models

This project supports **three different VLA models**. You can choose to install all of them or only a subset according to your needs.  

#### Option 1: OpenVLA
Install OpenVLA by following the installation instructions in its repository:  
[OpenVLA Repository](https://github.com/openvla/openvla)

#### Option 2: Pi0
Install Pi0 by following the installation instructions in its repository:  
[Pi0 Repository](https://github.com/Physical-Intelligence/openpi)

> ⚠️ **Pi0 must use the PyTorch checkpoint**, not the default JAX one shipped by
> upstream `openpi`. Convert an existing JAX training checkpoint to PyTorch first,
> e.g. via `openpi/examples/convert_jax_model_to_pytorch.py`, then pass the
> resulting directory (containing `model.safetensors` + `config.json` +
> `assets/local/<repo>/norm_stats.json`) as `--vla_path` to the attack /
> fine-tuning scripts below.

#### Option 3: OpenVLA-OFT
Install OpenVLA-OFT by following the installation instructions in its repository:  
[OpenVLA-OFT Repository](https://github.com/moojink/openvla-oft)

> ⚠️ You can choose to install all three VLA models or only the ones you need.  
> Make sure at least one selected model is installed **inside the project root directory** before running the main scripts.

## 📂 Dataset Support

We natively support **arbitrary datasets in RLDS format** for adversarial patch generation, and fine-tuning.

The dataset we typically used in our manuscript comes from: https://huggingface.co/datasets/openvla/modified_libero_rlds

## ▶️ Get Started

After installing the required sub-repositories, you can start using this project.  
> ⚡ Note: Our implementations generally do **not require additional dependencies**—you can use the environment provided by the selected VLA model directly.

### 1️⃣ Adversarial patch generation via EDPA

To generate adversarial patches via EDPA on OpenVLA:

```bash
python -m cp_openvla \
  --vla_path <PATH TO THE CHECKPOINT> \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name <NAME OF THE DATASET> \
  --batch_size 4 \
  --perturbation_ratio 0.05 \
  --alpha 0.2
```
For Pi0 and OpenVLA-OFT, use the corresponding scripts: `cp_pi0.py` and `cp_openvla_oft.py`, with the same arguments. `EDPA` now auto-dispatches on `cfg.model_family` internally, so no source-level edits are required to switch between OpenVLA / OpenVLA-OFT / Pi0.

> ⚠️ `--alpha` now weights the **alignment** term (`(1 - alpha)` weights the
> contrastive patch term). The default `0.2` reproduces our reported
> patch-dominant setting. Callers that previously passed `--alpha 0.8` should
> switch to `--alpha 0.2` for the same behavior.

> ⚡ Note: If both the model and dataset support arm-mounted cameras, you can select which camera view to use with the `--camera_view` argument.  
> Set it to `primary` to use the main camera or `wrist` to use the wrist-mounted camera.

> ⚠️ For **Pi0**, `--vla_path` must point at the **PyTorch** checkpoint dir
> (converted from JAX; see the Pi0 pre-requisite note above), not the JAX
> orbax checkpoint shipped by upstream `openpi`.

### 2️⃣ Evaluating the VLA Performance on LIBERO Simulation Benchmark

After generating adversarial patches, you can evaluate the VLA model performance on the LIBERO simulation benchmark **with or without patch attacks**:

```bash
# OpenVLA
python eval/simulation/Libero/openvla.py \
  --pretrained_checkpoint <MODEL_CHECKPOINT> \
  --task_suite_name <TASK_SUITE> \
  --patch_attack True \
  --perturbation_path <PERTURBATION_FILE>

# OpenVLA-OFT
python eval/simulation/Libero/openvla_oft.py \
  --pretrained_checkpoint <MODEL_CHECKPOINT> \
  --task_suite_name <TASK_SUITE> \
  --patch_attack True \
  --perturbation_primary_path <PERTURBATION_FILE> \
  --perturbation_wrist_path <PERTURBATION_FILE>

# Pi0
python eval/simulation/Libero/pi0.py \
  --task_suite_name <TASK_SUITE> \
  --patch_attack True \
  --perturbation_primary_path <PERTURBATION_FILE> \
  --perturbation_wrist_path <PERTURBATION_FILE>

```

### 3️⃣ Adversarial Fine-tuning of the Visual Encoder

To perform adversarial fine-tuning on the OpenVLA visual encoder:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 at_openvla.py \
  --vla_path <PATH TO THE CHECKPOINT> \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name <NAME OF THE DATASET> \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 10000 \
  --max_steps 50000
```

For **OpenVLA-OFT** and **Pi0**, use the analogous scripts with the same
arguments:

```bash
# OpenVLA-OFT
torchrun --standalone --nnodes 1 --nproc-per-node 1 at_openvla_oft.py \
  --vla_path <PATH TO THE OFT CHECKPOINT> \
  ...

# Pi0 (PyTorch checkpoint required)
torchrun --standalone --nnodes 1 --nproc-per-node 1 at_pi0.py \
  --vla_path <PATH TO THE PYTORCH pi0 CHECKPOINT> \
  ...
```

> ⚡ Note: Compared to adversarial patch generation via EDPA, adversarial fine-tuning requires significantly more computational resources. We generally recommend running it on GPUs with at least A100 or equivalent.

> ⚠️ **Pi0 adversarial fine-tuning (`at_pi0.py`) requires the PyTorch pi0 checkpoint** (converted from JAX; see the Pi0 pre-requisite note above). The JAX orbax checkpoint shipped by upstream `openpi` is not supported.