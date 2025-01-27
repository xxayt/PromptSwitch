import os
import argparse
from config.base_config import Config
from modules.basic_utils import mkdirp, deletedir


class AllConfig(Config):
    def __init__(self):
        super().__init__()

    def parse_args(self):
        description = 'Text-to-Video Retrieval'
        parser = argparse.ArgumentParser(description=description)


        parser.add_argument('--data_path', type=str, default='/home/xzj/Projects/TeachCLIP/datasets/msrvtt_data/MSRVTT_data.json', help="data pickle file path")
        parser.add_argument('--train_csv', type=str, default='/home/xzj/Projects/TeachCLIP/datasets/msrvtt_data/MSRVTT_train.9k.csv', help="train csv file path")
        parser.add_argument('--val_csv', type=str, default='/home/xzj/Projects/TeachCLIP/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv', help="val csv file path")
        parser.add_argument('--test_csv', type=str, default='/home/xzj/Projects/TeachCLIP/datasets/msrvtt_data/MSRVTT_JSFUSION_test.csv', help="test csv file path")
        parser.add_argument("--local_rank", default=0, type=int, help="distribted training")

        # data parameters
        parser.add_argument('--datatype', type=str, default='msrvtt', help="Dataset name")
        parser.add_argument('--videos_dir', type=str, default='/home/xzj/VisualSearch/msrvtt10k/VideoData', help="Location of videos")
        parser.add_argument('--msrvtt_train_file', type=str, default='9k')
        parser.add_argument('--num_frames', type=int, default=6)
        parser.add_argument('--num_test_frames', type=int, default=12)
        parser.add_argument('--num_prompts', type=int, default=6)
        parser.add_argument('--video_sample_type', default='rand', help="'rand'/'uniform'")
        parser.add_argument('--video_sample_type_test', default='uniform', help="'rand'/'uniform'")
        parser.add_argument('--input_res', type=int, default=224)


        # ema parameters
        parser.add_argument('--use_ema', action='store_true', default=False)
        parser.add_argument('--model_ema_decay', type=float, default=0.9999)

        # experiment parameters
        parser.add_argument('--exp_name', type=str, required=True, help="Name of the current experiment")
        parser.add_argument('--output_dir', type=str, default='/home/xzj/Projects/PromptSwitch/logs/')
        parser.add_argument('--save_every', type=int, default=1, help="Save model every n epochs")
        parser.add_argument('--n_display', type=int, default=10, help="Print training log every n steps")
        parser.add_argument('--evals_per_epoch', type=int, default=10, help="Number of times to evaluate per epoch")
        parser.add_argument('--load_epoch', type=int, help="Epoch to load from exp_name, or -1 to load model_best.pth")
        parser.add_argument('--eval_window_size', type=int, default=5, help="Size of window to average metrics")

        # model parameters
        parser.add_argument('--arch', type=str, default='prompt_clip')
        parser.add_argument('--clip_arch', type=str, default='ViT-B/32', help="CLIP arch. only when not using huggingface")
        parser.add_argument('--embed_dim', type=int, default=512, help="Dimensionality of the model embedding")

        # training parameters
        parser.add_argument('--loss', type=str, default='clip+caption')
        parser.add_argument('--num_captioner_layers', type=int, default=2)
        parser.add_argument('--frequent_word_weight', type=float, default=0.25)
        parser.add_argument('--caption_loss_mult', type=float, default=0.5)

        parser.add_argument('--clip_lr', type=float, default=1e-6, help='Learning rate used for CLIP params')
        parser.add_argument('--noclip_lr', type=float, default=1e-5, help='Learning rate used for new params')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--test_batch_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay')
        parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion for learning rate schedule')

        # frame pooling parameters
        parser.add_argument('--pooling_type', type=str, default='avg')
        parser.add_argument('--pooling_type_test', type=str, default='avg')
        parser.add_argument('--num_samples', type=int, default=2)
        parser.add_argument('--k', type=int, default=-1, help='K value for topk pooling')
        parser.add_argument('--attention_temperature', type=float, default=0.01, help='Temperature for softmax (used in attention pooling only)')
        parser.add_argument('--num_mha_heads', type=int, default=1, help='Number of parallel heads in multi-headed attention')
        parser.add_argument('--transformer_dropout', type=float, default=0.3, help='Dropout prob. in the transformer pooling')

        # system parameters
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--seed', type=int, default=24, help='Random seed')
        parser.add_argument('--no_tensorboard', action='store_true', default=False)
        parser.add_argument('--tb_log_dir', type=str, default='logs')

        args = parser.parse_args()

        args.model_path = os.path.join(args.output_dir, args.exp_name)
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)

        mkdirp(args.model_path)
        deletedir(args.tb_log_dir)
        mkdirp(args.tb_log_dir)

        return args
