CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12236 \
    train.py \
    --batch_size 32 --test_batch_size 32 --epochs=5 --n_display 100 --evals_per_epoch 5 \
    --arch prompt_clip \
    --loss clip+caption \
    --train_csv /home/xzj/Projects/TeachCLIP/datasets/msrvtt_data/MSRVTT_train.9k.csv \
    --val_csv /home/xzj/Projects/TeachCLIP/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv \
    --test_csv /home/xzj/Projects/TeachCLIP/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv \
    --data_path /home/xzj/Projects/TeachCLIP/datasets/msrvtt_data/MSRVTT_data.json \
    --videos_dir /home/xzj/VisualSearch/msrvtt10k/VideoData \
    --output_dir /home/xzj/Projects/PromptSwitch/logs/ \
    --datatype msrvtt \
    --exp_name msrvtt_promptswitch_vit32