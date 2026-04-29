```bash
curl -LsSf https://hf.co/cli/install.sh | bash
export PATH=$HOME/.local/bin:$PATH

hf download Qwen/Qwen3.5-0.8B
sudo mv $HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B /scratch/model_weights/

hf download Qwen/Qwen3.5-2B
sudo mv $HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B /scratch/model_weights/

# 4B and largers model have `linear_num_key_heads != linear_num_value_heads"` to test "groupvalue" kernels

hf download Qwen/Qwen3.5-4B
sudo mv $HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B /scratch/model_weights/

hf download Qwen/Qwen3.5-9B
sudo mv $HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B /scratch/model_weights/
```

Bigger models need W8 quant to fit into single device
- https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-27B-w8a8
- https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8
