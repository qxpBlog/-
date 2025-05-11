
CUDA_VISIBLE_DEVICES=0 python3 train.py \
    --dataset_name uav \
    --aggregation gem \
    --backbone dino \
    --cache_refresh_rate 1000 \
    --neg_samples_num 1000 \
    --queries_per_epoch 5000 \
    --criterion triplet \
    --epochs_num 1000 \
    --resize 322 322 \
    --fc_output_dim 1024 \
    --val_positive_dist_threshold 50000 \
    --similarity_top 10 \
    --patience 20 \
    --epochs_num 100 \
    --datasets_folder /home/qxp/VPR/vpr/datasets_vg/datasets \
    > "Your_code_path/eval_logs/qucik_start.log" 2>&1



