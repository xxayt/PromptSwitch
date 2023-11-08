from transformers import CLIPTokenizer

clip_tokenizer = CLIPTokenizer.from_pretrained("/home/xzj/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/e6a30b603a447e251fdaca1c3056b2a16cdfebeb", TOKENIZERS_PARALLELISM=False)
# clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
