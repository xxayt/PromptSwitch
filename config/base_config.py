from abc import abstractmethod, ABC


class Config(ABC):
    def __init__(self):
        args = self.parse_args()

        self.data_path = args.data_path
        self.test_csv = args.test_csv
        self.train_csv = args.train_csv
        self.local_rank = args.local_rank

        self.datatype = args.datatype
        self.videos_dir = args.videos_dir
        self.msrvtt_train_file = args.msrvtt_train_file
        self.num_frames = args.num_frames
        self.num_test_frames = args.num_test_frames
        self.num_prompts = args.num_prompts
        self.video_sample_type = args.video_sample_type
        self.video_sample_type_test = args.video_sample_type_test
        self.input_res = args.input_res

        self.use_ema = args.use_ema
        self.model_ema_decay = args.model_ema_decay

        self.exp_name = args.exp_name
        self.model_path = args.model_path 
        self.output_dir = args.output_dir
        self.save_every = args.save_every
        self.n_display = args.n_display
        self.evals_per_epoch = args.evals_per_epoch
        self.load_epoch = args.load_epoch
        self.eval_window_size = args.eval_window_size

        self.arch = args.arch
        self.clip_arch = args.clip_arch
        self.embed_dim = args.embed_dim

        self.loss = args.loss
        self.num_captioner_layers = args.num_captioner_layers
        self.frequent_word_weight = args.frequent_word_weight
        self.caption_loss_mult = args.caption_loss_mult

        self.clip_lr = args.clip_lr
        self.noclip_lr = args.noclip_lr
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay
        self.warmup_proportion = args.warmup_proportion

        self.pooling_type = args.pooling_type
        self.pooling_type_test = args.pooling_type_test
        self.num_samples = args.num_samples
        self.k = args.k
        self.attention_temperature = args.attention_temperature
        self.num_mha_heads = args.num_mha_heads
        self.transformer_dropout = args.transformer_dropout

        self.num_workers = args.num_workers
        self.seed = args.seed
        self.no_tensorboard = args.no_tensorboard
        self.tb_log_dir = args.tb_log_dir


    @abstractmethod
    def parse_args(self):
        raise NotImplementedError

