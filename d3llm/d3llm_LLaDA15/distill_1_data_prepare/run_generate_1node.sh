torchrun --nproc_per_node=8 \
    d3llm_llada_generate_multinode.py \
    --num_gpus 8 \
    --steps 1024 \
    --gen_length 1024 \
    --block_length 32 \
    --save_interval 20 \
    --output_dir "/sensei-fs-3/users/hyou/wei/d3LLM/d3llm/d3llm_LLaDA15/distill_1_data_prepare/trajectory_output_1024"