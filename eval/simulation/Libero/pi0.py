import os
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import collections
import dataclasses
import logging
import math
import pathlib
import torch

from LIBERO.libero.libero import benchmark
from LIBERO.libero import get_libero_path
from LIBERO.libero.libero.envs import OffScreenRenderEnv
import numpy as np

from openpi.training import config
from openpi.shared import download
from openpi_client import image_tools
from openpi.policies import policy_config
from torchvision import transforms

from experiments.robot.robot_utils import (
    DATE_TIME,
)

from experiments.robot.libero.libero_utils import (
    save_rollout_video,
)

from VLAAttacker.UUPP import UUPP

import tqdm
import draccus

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "pi0"                    # Model family
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    local_log_dir: str = "./logs"                    # Local directory for eval logs
    run_id_note: str = None                          # Optional note to append to the run ID
    seed: int = 7                                    # Random Seed (for reproducibility)

    patch_attack: bool = True                        # Whether to use patch-based adversarial attack
    perturbation_primary_path: str = ""              # Path to perturbation file on primary camera
    perturbation_wrist_path: str = ""                # Path to perturbation file on wrist camera

def apply_perturbation_to_raw_images(images, perturbation, position=(-1, -1)):

    image_tensors = torch.stack([transforms.ToTensor()(image) for image in images])
    
    B, C, H, W = image_tensors.shape
    pc, ph, pw = perturbation.shape

    assert pc == C, "Perturbation must have the same number of channels as the input images."

    perturbated_images = torch.zeros_like(image_tensors)

    if position == (-1, -1):
        top, left = torch.randint(0, H - ph + 1, (1,)).item(), torch.randint(0, W - pw + 1, (1,)).item()
    else:
        top, left = position

    assert top >= 0 and left >= 0 and top + ph <= H and left + pw <= W, "Perturbation must fit within the image dimensions."

    for i in range(len(perturbated_images)):

        mask = torch.zeros_like(image_tensors[i])
        mask[:, top:top + ph, left:left + pw] = 1.0

        padded_perturb = torch.zeros_like(image_tensors[i])
        padded_perturb[:, top:top + ph, left:left + pw] = perturbation

        perturbated_images[i] = (1 - mask) * image_tensors[i] + padded_perturb
    
    return [
        np.array(transforms.ToPILImage()(img)) for img in perturbated_images
    ]

@draccus.wrap()
def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize local logging
    run_id = f"EVAL-{Args.task_suite_name}-{Args.model_family}-{DATE_TIME}"
    if Args.run_id_note is not None:
        run_id += f"--{Args.run_id_note}"
    os.makedirs(Args.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(Args.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {Args.task_suite_name}")
    log_file.write(f"Task suite: {Args.task_suite_name}\n")


    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_libero")
    client = policy_config.create_trained_policy(config.get_config("pi0_fast_libero"), checkpoint_dir)

    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")


    if args.patch_attack:
        if args.perturbation_primary_path == "" or args.perturbation_wrist_path == "":
            raise ValueError("If using patch-based attack, please provide valid perturbation paths for both primary and wrist!")
        else:
            # Load primary patch
            if args.perturbation_primary_path.endswith(".npy"):
                perturbation_primary = torch.from_numpy(np.load(args.perturbation_primary_path))
            elif args.perturbation_primary_path.endswith(".pt"):
                perturbation_primary = torch.as_tensor(torch.load(args.perturbation_primary_path))
            else:
                raise ValueError("Unsupported primary perturbation file type! Use .npy or .pt")

            # Load wrist patch
            if args.perturbation_wrist_path.endswith(".npy"):
                perturbation_wrist = torch.from_numpy(np.load(args.perturbation_wrist_path))
            elif args.perturbation_wrist_path.endswith(".pt"):
                perturbation_wrist = torch.as_tensor(torch.load(args.perturbation_wrist_path))
            else:
                raise ValueError("Unsupported wrist perturbation file type! Use .npy or .pt")
    
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
 
            top_primary = np.random.randint(0, args.resize_size - perturbation_primary.shape[1] + 1)
            left_primary = np.random.randint(0, args.resize_size - perturbation_primary.shape[2] + 1)

            top_wrist = np.random.randint(0, args.resize_size - perturbation_wrist.shape[1] + 1)
            left_wrist = np.random.randint(0, args.resize_size - perturbation_wrist.shape[2] + 1)


            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    # Mute the primary camera or wrist camera

                    if args.patch_attack:
                        img = apply_perturbation_to_raw_images([img], perturbation_primary, (top_primary, left_primary))[0]
                        wrist_img = apply_perturbation_to_raw_images([wrist_img], perturbation_wrist, (top_wrist, left_wrist))[0]

                    # Apply perturbation
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    
                    # Save preprocessed image for replay video
                    replay_images.append(wrist_img)

                    img = np.zeros_like(img)


                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        print(f"Episode {task_episodes+1} finished after {t} steps.")
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )
            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()


        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    # logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    # logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero()