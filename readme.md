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
  --alpha 0.8
```
For Pi0 and OpenVLA-OFT, use the corresponding scripts: `cp_pi0.py` and `cp_openvla_oft.py`, with the same arguments. If you want to conduct adversarial patch generation on **OpenVLA-OFT**, you need to manually comment out **line 12** and uncomment **line 13** in `VLAAttacker/pytorch/EDPA.py`. If you want to switch back to **OpenVLA**, you need to uncomment **line 12** and comment out **line 13**.



> ⚡ Note: If both the model and dataset support arm-mounted cameras, you can select which camera view to use with the `--camera_view` argument.  
> Set it to `primary` to use the main camera or `wrist` to use the wrist-mounted camera.

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

### 3️⃣ Adversarial Fine-tuning on OpenVLA Visual Encoder

To perform adversarial fine-tuning on the OpenVLA visual encoder:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 at_openvla.py \ 
  --vla_path <PATH TO THE CHECKPOINT> \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name <NAME OF THE DATASET> \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 10000 \
  --max_steps 50000 \
```
> ⚡ Note: Compared to adversarial patch generation via EDPA, adversarial fine-tuning requires significantly more computational resources. We generally recommend running it on GPUs with at least A100 or equivalent.
> Currently, adversarial fine-tuning scripts are only available for OpenVLA; Pi0 and OpenVLA-OFT are not yet supported.