```bash
curl -LsSf https://hf.co/cli/install.sh | bash
export PATH=$HOME/.local/bin:$PATH

hf download Qwen/Qwen3.5-0.8B
sudo mv $HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B /scratch/model_weights/

hf download Qwen/Qwen3.5-9B
sudo mv $HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B /scratch/model_weights/
```
