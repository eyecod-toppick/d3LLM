import subprocess
import json
import os
from datasets import Dataset
from datasets import load_dataset
import argparse


def main(
    num_gpus=24,
    steps=256,
    gen_length=256,
    block_length=32,
    output_dir="trajectory_output",
    max_data_num=-1,
    rank=0,
    local_rank=0,
    world_size=None,
    save_interval=10,
):
    """Distributed trajectory generation using multiple GPUs on one node.

    `rank` and `world_size` describe the distributed configuration.  They can
    be supplied via arguments or through environment variables (RANK,
    WORLD_SIZE, LOCAL_RANK or SLURM equivalents).  When running without
    any distribution helpers, the script behaves as a single-process job.
    """

    # Determine distributed rank and GPU assignment.
    # This script can run on a single node with multiple GPUs. It supports
    # the following ways of specifying ranks:
    #   * command‑line args (--rank, --local_rank, --world_size)
    #   * environment variables set by torchrun/torch.distributed
    #       (RANK, LOCAL_RANK, WORLD_SIZE)
    #   * SLURM environment variables (SLURM_PROCID, SLURM_LOCALID,
    #       SLURM_NTASKS) for backwards compatibility.
    #
    # The `num_gpus` argument is only used as a default for world_size when
    # nothing else is provided.  The `rank` value determines which slice of
    # the dataset this process will handle.

    # start with values from arguments
    rank = rank
    local_rank = local_rank
    world_size = world_size if world_size is not None else num_gpus

    # override from SLURM if present
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
    if "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
    if "SLURM_NTASKS" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])

    # override from torchrun/torch.distributed env vars
    rank = int(os.environ.get("RANK", rank))
    local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", world_size))

    print(f"Process {rank}/{world_size}, Local GPU {local_rank}")
    
    # Only the first task does dataset loading and final concatenation

    import json
    current_size = 0
    # with open("/sensei-fs-3/users/hyou/wei/d3LLM/d3llm/d3llm_LLaDA/distill_1_data_prepare/trajectory_output/trajectory_part_5.json", "r") as f:
    #     current_size = len(json.load(f))

    dataset = load_dataset("Zigeng/dParallel_LLaDA_Distill_Data", split="train")
    if rank == 0:
        # Load dataset to get total size
        total_size = len(dataset)

        # Apply max_data_num limit
        if max_data_num > 0:
            total_size = min(total_size, max_data_num) - current_size

        os.makedirs(output_dir, exist_ok=True)
        
        # Save total_size to a file for other tasks
        with open(os.path.join(output_dir, "total_size.txt"), "w") as f:
            f.write(str(total_size))
        
        print(f"Total dataset size: {total_size}")
        print(f"Distributing across {world_size} processes (num_gpus={num_gpus})")

    # current_size += 5 * (len(dataset) // 8)
    
    # Barrier: wait for task 0 to write total_size
    # In SLURM with srun, we can use a simple file-based barrier
    import time
    total_size_file = os.path.join(output_dir, "total_size.txt")
    while not os.path.exists(total_size_file):
        time.sleep(1)
    
    with open(total_size_file, "r") as f:
        total_size = int(f.read().strip())
    
    # Calculate this task's chunk
    chunk_size = (total_size + world_size - 1) // world_size
    gpu_id = rank  # global rank corresponds to GPU id on a single node
    start_idx = gpu_id * chunk_size + current_size
    end_idx = min((gpu_id + 1) * chunk_size + current_size, total_size + current_size)
    output_file = os.path.join(output_dir, f"trajectory_part_{gpu_id}.json")

    # Run generation on this GPU
    # Determine the script path and working directory
    # The partly script needs to be run from the d3LLM root directory
    # so that its sys.path manipulation can find the utils module correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    d3llm_root = os.path.abspath(os.path.join(script_dir, '../../..'))
    partly_script = os.path.join(script_dir, 'd3llm_llada_generate_partly.py')
    
    cmd = [
        "python",
        partly_script,
        "--start_idx",
        str(start_idx),
        "--end_idx",
        str(end_idx),
        "--steps",
        str(steps),
        "--gen_length",
        str(gen_length),
        "--block_length",
        str(block_length),
        "--output_file",
        output_file,
        "--max_data_num",
        str(max_data_num),
        "--save_interval",
        str(save_interval),
    ]

    env = os.environ.copy()
    # Use local GPU ID (from either slurm_localid or torchrun's LOCAL_RANK)
    env["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    print(f"GPU {gpu_id}: Processing indices {start_idx}-{end_idx}")
    result = subprocess.run(cmd, env=env, cwd=d3llm_root)
    
    if result.returncode != 0:
        print(f"GPU {gpu_id}: Generation failed with return code {result.returncode}")
        return

    print(f"GPU {gpu_id}: Generation completed")

    # Barrier: wait for all tasks to complete
    # Create a completion flag for this task
    completion_file = os.path.join(output_dir, f"completed_{gpu_id}.flag")
    with open(completion_file, "w") as f:
        f.write("done")
    
    # Only rank 0 does concatenation
    if rank == 0:
        print("Waiting for all tasks to complete...")
        # Wait for all completion flags
        while True:
            completed = sum(
                1 for i in range(world_size)
                if os.path.exists(os.path.join(output_dir, f"completed_{i}.flag"))
            )
            if completed == world_size:
                break
            print(f"Completed: {completed}/{world_size}")
            time.sleep(5)
        
        print("All tasks completed. Concatenating results...")
        
        # Concatenate results
        all_data = []
        for gpu_id in range(world_size):
            part_file = os.path.join(output_dir, f"trajectory_part_{gpu_id}.json")
            if os.path.exists(part_file):
                with open(part_file, "r") as f:
                    data = json.load(f)
                    all_data.extend(data)
                    print(f"Loaded {len(data)} samples from GPU {gpu_id}")
            else:
                print(f"Warning: {part_file} not found")

        # Convert to dataset format with correctness check
        dataset_dict = {
            "idx": [d["idx"] for d in all_data],
            "question": [d["question"] for d in all_data],
            "prompt_ids": [d["prompt_ids"] for d in all_data],
            # "trajectory": [d["trajectory"] for d in all_data],
            "final_output": [d["final_output"] for d in all_data],
            "generated_text": [d["generated_text"] for d in all_data],
            "llm_answer": [d["llm_answer"] for d in all_data],
            "gt_answer": [d["gt_answer"] for d in all_data],
            "is_correct": [d["is_correct"] for d in all_data],
        }

        # Print statistics
        num_correct = sum(dataset_dict["is_correct"])
        total = len(dataset_dict["is_correct"])
        accuracy = num_correct / total if total > 0 else 0
        print(f"Correctness: {num_correct}/{total} = {accuracy:.2%}")

        final_dataset = Dataset.from_dict(dataset_dict)
        final_dataset.save_to_disk(os.path.join(output_dir, "trajectory_dataset"))
        print(
            f"Saved complete dataset with {len(all_data)} samples to {output_dir}/trajectory_dataset"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate trajectories across multiple GPUs on a single node."
    )
    parser.add_argument("--num_gpus", type=int, default=24,
                        help="Total number of GPUs available (used as default world size)")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="trajectory_output")
    parser.add_argument(
        "--max_data_num",
        type=int,
        default=-1,
        help="Max number of samples to generate (-1 for no limit)",
    )
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Interval (in steps) to save intermediate results")
    # distributed args
    parser.add_argument("--rank", type=int, default=0,
                        help="Global rank of this process")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local GPU index for this process")
    parser.add_argument("--world_size", type=int, default=None,
                        help="Total number of processes (defaults to num_gpus)")
    args = parser.parse_args()

    main(
        args.num_gpus,
        args.steps,
        args.gen_length,
        args.block_length,
        args.output_dir,
        args.max_data_num,
        rank=args.rank,
        local_rank=args.local_rank,
        world_size=args.world_size,
        save_interval=args.save_interval,
    )

