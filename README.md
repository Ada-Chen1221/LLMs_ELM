# LLM Lab: Transformers 推理与 LoRA/QLoRA 训练脚手架

这是一个最小可运行的 HuggingFace Transformers 原生脚手架，用于在实验室 RTX 3090 环境中先跑通：

- Transformers 原生 causal LM 推理
- LoRA SFT 训练
- QLoRA 4-bit SFT 训练

默认模型是 `Qwen/Qwen3-1.7B`。如果显存或下载条件不足，可以把命令中的 `--model_name_or_path` 切换为 `Qwen/Qwen3-0.6B`；后续也可以切换到 `Qwen/Qwen3-4B`、`Qwen/Qwen3-8B` 或 `Qwen/Qwen3-14B`，同时相应调小 `--max_length`、`--per_device_train_batch_size`，或使用 QLoRA。

## 项目结构

```text
configs/
  infer_qwen3_1p7b.yaml
  train_lora_qwen3_1p7b.yaml
  train_qlora_qwen3_1p7b.yaml
data/
  toy_sft.jsonl
  sft_prompt_groundtruth.json
  prompts.jsonl
  prompts.json
scripts/
  check_cuda.py
  download_model.py
  infer_transformers.py
  infer_lora.py
  batch_infer_transformers.py
  batch_api_infer.py
  train_lora.py
  train_qlora.py
src/llm_lab/
  __init__.py
  data.py
  model_utils.py
  train_utils.py
requirements.txt
README.md
```

> 当前脚本以命令行参数为主；`configs/` 中的 YAML 文件用于记录推荐配置，方便后续接入配置加载器。

## 创建环境与安装依赖

建议在服务器上创建独立 conda 环境：

```bash
conda create -n llm-lab python=3.10 -y
conda activate llm-lab
pip install --upgrade pip
pip install -r requirements.txt
```

重要：不要盲目安装最新的 `torch` wheel。`torch` wheel 自带的 CUDA runtime 必须不高于当前 NVIDIA 驱动能支持的 CUDA 版本。比如你看到的日志里 `PyTorch version: 2.12.0+cu130` / `PyTorch CUDA build: 13.0`，但驱动只报告 `found version 12020`（约等于驱动支持 CUDA 12.2），因此 PyTorch 会判定 CUDA 不可用。当前 `requirements.txt` 默认固定到 `torch==2.5.1+cu121`，更适合这类 CUDA 12.2 驱动的 RTX 3090 服务器。

如果你已经装到了不兼容的 `torch+cu130`，优先这样重装：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/check_cuda.py
```

如果管理员能升级驱动，也可以保留新版 PyTorch，但需要把 NVIDIA 驱动升级到支持该 PyTorch CUDA build 的版本。CUDA 13.0 需要 580 系列或更新的 Linux 驱动；CUDA 12.2 驱动无法运行 `+cu130` 的 PyTorch。

如果 `bitsandbytes` 与 CUDA/PyTorch 版本不匹配，请根据服务器 CUDA 版本重新安装兼容的 PyTorch 和 bitsandbytes。脚本不会写入 HuggingFace token；如需访问 gated/private 模型，请在服务器上用 `huggingface-cli login` 登录。

## 检查 CUDA

```bash
python scripts/check_cuda.py
```

该命令会输出 PyTorch 版本、PyTorch CUDA build、`nvidia-smi` 驱动版本、驱动支持的 CUDA Version、CUDA 是否可用、可见 GPU 数量、GPU 名称与显存。脚本不会设置 `CUDA_VISIBLE_DEVICES`，请在命令前自行指定可见 GPU。若 PyTorch CUDA build 高于驱动支持版本，脚本会给出重装 cu121 PyTorch 或升级驱动的提示。

## 单卡 Transformers 推理

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --prompt "请用一句话解释什么是大语言模型。"
```

常用参数：

- `--dtype auto|float16|bfloat16|float32`
- `--device_map auto`
- `--system_prompt "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`
- `--max_new_tokens 128`
- `--temperature 0.7`
- `--top_p 0.9`

推理脚本按 Qwen 官方 quickstart 风格实现：如果 tokenizer 支持 `chat_template`，先用 `apply_chat_template(..., tokenize=False, add_generation_prompt=True)` 渲染 messages，再调用 `tokenizer([text], return_tensors="pt")`，最后使用 `model.generate(**model_inputs)`；否则回退到普通 prompt。

如果命令停在 `[1/5] Loading tokenizer...`，通常说明 `AutoTokenizer.from_pretrained()` 正在联网解析/下载 tokenizer、config 或 chat template。`Qwen/Qwen3-1.7B` 这种写法不会在当前目录创建 `Qwen/` 文件夹；默认缓存位置一般是 `~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/`。如果想看到一个直观的模型目录，建议先预下载到 `models/Qwen3-1.7B`，再用本地路径推理：

```bash
python scripts/download_model.py \
  --repo_id Qwen/Qwen3-1.7B \
  --local_dir models/Qwen3-1.7B
CUDA_VISIBLE_DEVICES=0 python scripts/infer_transformers.py \
  --model_name_or_path models/Qwen3-1.7B \
  --prompt "请用一句话解释什么是大语言模型。"
```

也可以直接使用 HuggingFace CLI：

```bash
huggingface-cli download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
```

如果模型已经在缓存中，想避免脚本联网等待，可以加 `--local_files_only`；如果本地没有缓存，它会快速失败并提示你先下载：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --local_files_only \
  --prompt "请用一句话解释什么是大语言模型。"
```

在国内网络环境中，也可以在运行前设置可用的 HuggingFace 镜像端点，例如 `export HF_ENDPOINT=https://hf-mirror.com`，或让服务器管理员配置代理/缓存。建议先执行 `python scripts/download_model.py ...` 确认下载进度，下载完成后再运行推理/训练脚本。



## 调用下载好的本地模型批量处理 prompt

当前默认推荐：**直接加载你已经下载好的模型目录**，例如 `models/Qwen3-4B-Instruct-2507`，然后从 `.json` 或 `.jsonl` 文件批量读取 `prompt`，生成结果并保存到新的文件。

你的数据是一个 JSON array，每个元素里都有 `prompt` 字段，这种格式已支持：

```json
[
  {
    "prompt_id": "claim001_highInv_highExpert_StrongArg",
    "claim_id": 1,
    "prompt": "You are a cinema customer ... Do not include any explanation."
  },
  {
    "prompt_id": "claim001_highInv_highExpert_WeakArg",
    "claim_id": 1,
    "prompt": "You are a cinema customer ... Do not include any explanation."
  }
]
```

仓库提供了同格式示例输入 `data/prompts.json`。用你现在下载好的 `models/Qwen3-4B-Instruct-2507` 可以这样跑：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_infer_transformers.py \
  --model_name_or_path models/Qwen3-4B-Instruct-2507 \
  --input_file data/prompts.json \
  --output_file outputs/qwen3_4b_batch_outputs.json \
  --batch_size 4 \
  --max_new_tokens 64 \
  --output_field output \
  --overwrite
```

输出默认会保持 JSON array 格式，并在每个原始对象上新增一个 `output` 字段保存模型输出。例如：

```json
[
  {
    "prompt_id": "claim001_highInv_highExpert_StrongArg",
    "claim_id": 1,
    "prompt": "...",
    "output": "My attitude score toward this proposal is: 9"
  }
]
```

如果每条数据需要重复生成多次，可以加 `--num_repeats`。此时 `output` 字段会变成一个 list，按生成顺序保存多次输出：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_infer_transformers.py \
  --model_name_or_path models/Qwen3-4B-Instruct-2507 \
  --input_file data/prompts.json \
  --output_file outputs/qwen3_4b_batch_outputs_repeat5.json \
  --batch_size 4 \
  --max_new_tokens 64 \
  --num_repeats 5 \
  --output_field output \
  --overwrite
```

重复 5 次后的输出示例：

```json
[
  {
    "prompt_id": "claim001_highInv_highExpert_StrongArg",
    "prompt": "...",
    "output": [
      "My attitude score toward this proposal is: 9",
      "My attitude score toward this proposal is: 8",
      "My attitude score toward this proposal is: 9",
      "My attitude score toward this proposal is: 10",
      "My attitude score toward this proposal is: 8"
    ]
  }
]
```

仍然兼容 JSONL：如果输入或输出想用 `.jsonl`，把 `--input_file` / `--output_file` 改成 `.jsonl` 即可；输出格式也可以用 `--output_format json|jsonl` 强制指定。

常用参数：

- `--prompt_field prompt`：指定从 JSON 哪个字段读取普通 prompt。
- `--messages_field messages`：如果每条数据已经是 chat messages，用这个字段。
- `--output_field output`：指定模型输出写回到哪个字段；默认就是 `output`。
- `--adapter_path outputs/...`：可选；加载训练好的 LoRA/QLoRA adapter 做批量推理。
- `--merge_and_unload`：可选；配合 `--adapter_path`，在内存中先合并 adapter 再生成。
- `--num_repeats 5`：每条 prompt 重复生成 5 次；当大于 1 时，`output` 字段保存为 list。
- `--always_list_output`：即使 `--num_repeats 1`，也把 `output` 保存为 list。
- `--seed 42`：可选随机种子，方便复现实验。
- `--batch_size 4`：单次送入模型的生成样本数；如果 `num_repeats` 很大，显存不够就调小到 1 或 2。
- `--max_new_tokens 64`：你的 Likert 任务只需要很短输出，建议先用 32 或 64。
- `--dtype auto|float16|bfloat16|float32`：RTX 3090 通常用 `auto` 或 `float16`。
- `--device_map auto`：默认自动放到可见 GPU。

## 可选：调用已部署模型并批量处理 prompt

如果模型已经通过 vLLM、TGI 或其他服务部署成 OpenAI-compatible API，可以用 `scripts/batch_api_infer.py` 批量请求 `/v1/chat/completions`。输入文件是 JSONL，每行一个样本，支持两种格式：

```json
{"id": "p1", "prompt": "请用一句话解释什么是大语言模型。"}
{"id": "p2", "messages": [{"role": "user", "content": "LoRA 适合什么场景？"}]}
```

仓库提供了示例输入 `data/prompts.jsonl`。假设你的服务地址是 `http://127.0.0.1:8000/v1`，模型名是部署时暴露的名字，可以这样批量跑：

```bash
python scripts/batch_api_infer.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model Qwen3-1.7B \
  --input_file data/prompts.jsonl \
  --output_file outputs/batch_api_results.jsonl \
  --concurrency 4 \
  --max_tokens 512 \
  --overwrite
```

输出也是 JSONL，每行包含 `ok`、`response`、`latency_sec`、原始 `raw_response` 或错误信息。若你的服务需要鉴权，可以设置：

```bash
export OPENAI_API_KEY=你的服务token
```

如果是本地 vLLM，一般服务端类似：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3-1.7B \
  --served-model-name Qwen3-1.7B \
  --host 0.0.0.0 \
  --port 8000
```

## LoRA/QLoRA 训练后如何推理

是的，训练完成后通常需要“基座模型 + adapter”一起加载再推理（不一定必须先离线合并权重）。本仓库提供了 `scripts/infer_lora.py`：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_lora.py \
  --base_model_name_or_path Qwen/Qwen3-1.7B \
  --adapter_path outputs/qwen3_1p7b_lora_test \
  --prompt "请用一句话解释什么是大语言模型。"
```

如果你希望把 LoRA 权重合并回基座模型后再推理（得到纯 Transformers 模型对象），可加：

```bash
--merge_and_unload
```

同样适用于 QLoRA 训练出来的 adapter（前提是 adapter 与 base model 对应）。

如果训练完 LoRA/QLoRA 后要继续做 JSON 批量推理，直接用同一个批量脚本并加上 `--adapter_path`。这里的 `--model_name_or_path` 是**原始基座模型目录**，`--adapter_path` 是训练输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_infer_transformers.py \
  --model_name_or_path models/Qwen3-4B-Instruct-2507 \
  --adapter_path outputs/qwen3_4b_qlora_sft_maskedlowinv \
  --input_file data/prompts.json \
  --output_file outputs/qwen3_4b_qlora_batch_outputs.json \
  --prompt_field prompt \
  --output_field output \
  --batch_size 4 \
  --max_new_tokens 64 \
  --overwrite
```

如果想在内存中先把 adapter 合并到 base model 再生成，可以加 `--merge_and_unload`：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_infer_transformers.py \
  --model_name_or_path models/Qwen3-4B-Instruct-2507 \
  --adapter_path outputs/qwen3_4b_qlora_sft_maskedlowinv \
  --merge_and_unload \
  --input_file data/prompts.json \
  --output_file outputs/qwen3_4b_qlora_merged_batch_outputs.json \
  --prompt_field prompt \
  --output_field output \
  --batch_size 4 \
  --max_new_tokens 64 \
  --overwrite
```

`--merge_and_unload` 只是当前进程内合并，方便得到普通 Transformers 模型对象做推理；不会覆盖你的原始模型目录或 adapter 目录。


## 使用 prompt / groundtruth JSON 文件做 SFT

你的 SFT 数据可以是一个 `.json` 文件，顶层是 list，每条样本包含 `prompt` 和 `groundtruth` 字段，例如：

```json
[
  {
    "prompt_id": "claim001_lowInv_highExpert_StrongArg",
    "prompt": "You are a cinema customer ... Do not include any explanation.",
    "split": "train",
    "groundtruth": "My attitude score toward this proposal is: 6"
  }
]
```

训练脚本会把它转换成 chat SFT 文本：`user=prompt`，`assistant=groundtruth`。如果 tokenizer 支持 chat template，会自动使用 `apply_chat_template(..., add_generation_prompt=False)`。仓库提供了同格式示例 `data/sft_prompt_groundtruth.json`。

LoRA 训练示例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --model_name_or_path models/Qwen3-4B-Instruct-2507 \
  --train_file data/sft_prompt_groundtruth.json \
  --prompt_field prompt \
  --response_field groundtruth \
  --split_field split \
  --split train \
  --output_dir outputs/qwen3_4b_lora_sft_prompt_groundtruth \
  --num_train_epochs 1 \
  --max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

QLoRA 训练示例（显存更省，RTX 3090 上建议优先试这个）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora.py \
  --model_name_or_path models/Qwen3-4B-Instruct-2507 \
  --train_file data/sft_prompt_groundtruth.json \
  --prompt_field prompt \
  --response_field groundtruth \
  --split_field split \
  --split train \
  --output_dir outputs/qwen3_4b_qlora_sft_prompt_groundtruth \
  --num_train_epochs 1 \
  --max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

相关参数：

- `--prompt_field prompt`：输入 prompt 字段名。
- `--response_field groundtruth`：监督目标字段名。
- `--split_field split --split train`：可选，只训练 `split == "train"` 的样本。
- 如果你的字段名不同，只需要改这几个参数，不需要改代码。

## 从 checkpoint 或已有 adapter 继续训练

如果测试结果还不理想，可以继续训练。这里有两种常见情况：

### 方式 A：从 `checkpoint-*` 精确恢复训练状态（推荐）

如果 `output_dir` 里有 Trainer 自动保存的 checkpoint，例如：

```text
outputs/qwen3_4b_qlora_sft_maskedlowinv/checkpoint-435
```

优先用 `--resume_from_checkpoint`。这种方式会恢复 adapter 权重、optimizer、scheduler、global step 等训练状态。注意 `--num_train_epochs` 要设置成**总 epoch 数**，不是“再训练几个 epoch”。例如之前已经训练 1 个 epoch，现在想继续到 3 个 epoch，就设置 `--num_train_epochs 3`。

精确恢复 Trainer checkpoint 时，建议保持 batch size、gradient accumulation 等训练参数与原 run 一致。另外，较新的 Transformers 会因为 CVE-2025-32434 要求 `torch>=2.6` 才能加载 `optimizer.pt` / `scheduler.pt` 这类 Trainer 状态文件；如果你的环境还是 `torch 2.5.x`，请看下面“方式 B”。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora.py \
  --model_name_or_path models/Qwen3-4B-Instruct-2507 \
  --train_file data/split_data/train_lowinv_sft.json \
  --prompt_field prompt \
  --response_field groundtruth \
  --output_dir outputs/qwen3_4b_qlora_sft_maskedlowinv \
  --resume_from_checkpoint outputs/qwen3_4b_qlora_sft_maskedlowinv/checkpoint-435 \
  --num_train_epochs 3 \
  --max_length 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

LoRA 脚本同理，把 `scripts/train_qlora.py` 换成 `scripts/train_lora.py` 即可。

### 方式 B：从 checkpoint 只加载 adapter 权重继续训（适合 torch<2.6 或想改 batch size）

如果遇到 `torch.load` / CVE-2025-32434 报错，或者你想像下面这样把 batch size 从 1 改成 10，不要精确恢复 optimizer/scheduler。可以加 `--resume_checkpoint_as_adapter`：脚本会把 `--resume_from_checkpoint` 指向的 checkpoint 当作 adapter 权重加载，然后重新初始化 optimizer/scheduler。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora.py \
  --model_name_or_path models/Qwen3-4B-Instruct-2507 \
  --train_file data/split_data/train_lowinv_sft.json \
  --prompt_field prompt \
  --response_field groundtruth \
  --output_dir outputs/qwen3_4b_qlora_sft_maskedlowinv_continue \
  --resume_from_checkpoint outputs/qwen3_4b_qlora_sft_maskedlowinv/checkpoint-435 \
  --resume_checkpoint_as_adapter \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --max_length 256 \
  --per_device_train_batch_size 10 \
  --gradient_accumulation_steps 10 \
  --gradient_checkpointing
```

这种方式不是“精确恢复 step”，所以 `--num_train_epochs 1` 表示从当前 adapter 权重开始重新跑 1 个 epoch。继续训练时建议把学习率调小一些，例如 `5e-5` 或 `1e-5`。

### 方式 C：从最终 adapter 继续微调（不恢复 optimizer）

如果你只有最终保存的 adapter 目录，或者想换一个新的 `output_dir` 继续训，可以用 `--adapter_path`。这种方式会加载已有 adapter 权重并继续更新它，但 optimizer/scheduler 会重新开始：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora.py \
  --model_name_or_path models/Qwen3-4B-Instruct-2507 \
  --adapter_path outputs/qwen3_4b_qlora_sft_maskedlowinv \
  --train_file data/split_data/train_lowinv_sft.json \
  --prompt_field prompt \
  --response_field groundtruth \
  --output_dir outputs/qwen3_4b_qlora_sft_maskedlowinv_continue \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --max_length 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

建议继续训练时把学习率调小一些，例如从 `2e-4` 降到 `5e-5` 或 `1e-5`，避免把已经学到的 adapter 权重冲坏。

如果你仍然想使用 `--resume_from_checkpoint` 精确恢复，请升级到与服务器 NVIDIA driver 兼容的 `torch>=2.6`，并尽量不要改变原 checkpoint 的 batch size / accumulation 等关键训练参数。

## 单卡 LoRA 训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_lora_test \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

LoRA 默认 target modules 适配 Qwen/LLaMA 类 causal LM：

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

训练结束后会把 adapter 和 tokenizer 保存到 `--output_dir`。

## 单卡 QLoRA 训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_qlora_test \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

QLoRA 默认使用：

- 4-bit `BitsAndBytesConfig`
- `nf4`
- double quant
- compute dtype `float16`

RTX 3090 上默认使用 fp16 更稳；如确有需要可增加 `--bf16`，脚本会关闭 fp16。

> 说明：部分 Qwen 本地 checkpoint 的 config 里可能带有 `bfloat16` dtype，PEFT/TRL 注入 LoRA adapter 后可能让可训练 adapter 参数或梯度变成 bf16。脚本在 fp16 训练模式下会自动把可训练参数转成 fp32，避免 PyTorch GradScaler 报 `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'`。这不会把 4-bit base model 反量化，只影响很小的可训练 adapter 权重。

## 双卡训练示例

LoRA 双卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 scripts/train_lora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_lora_2gpu \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

QLoRA 双卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 scripts/train_qlora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_qlora_2gpu \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

如果只想让脚本看到第 2、3 张物理卡，可以使用：

```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 scripts/train_lora.py ...
```

## 切换模型

只需要修改命令中的 `--model_name_or_path`：

```bash
--model_name_or_path Qwen/Qwen3-0.6B
--model_name_or_path Qwen/Qwen3-4B
--model_name_or_path Qwen/Qwen3-8B
--model_name_or_path Qwen/Qwen3-14B
```

建议：

- 0.6B/1.7B：优先用于 smoke test。
- 4B/8B：单卡 24GB 可优先尝试 QLoRA，LoRA 需要更谨慎调小 batch/length。
- 14B：建议 QLoRA + 多卡，先从 `--max_length 512` 和 batch size 1 开始。

## 数据格式

`data/toy_sft.jsonl` 支持两种格式：

```json
{"text": "用户：...\n助手：..."}
```

或：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

如果 tokenizer 支持 chat template，`messages` 会通过 `tokenizer.apply_chat_template(..., tokenize=False)` 转成训练文本；否则退化为简单的 `role: content` 文本。

## 必跑验收命令

```bash
python scripts/check_cuda.py
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --prompt "请用一句话解释什么是大语言模型。"
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_lora_test \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_qlora_test \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

## 常见问题

- **CUDA 不可用**：运行 `python scripts/check_cuda.py`，优先确认 PyTorch CUDA build 是否高于 `nvidia-smi` 显示的 CUDA Version。若出现 `torch+cu130` 但驱动只支持 CUDA 12.2，请重装 `torch==2.5.1+cu121` 或升级 NVIDIA 驱动。
- **推理卡在加载阶段**：如果停在 `[1/5] Loading tokenizer...`，通常是在联网下载 tokenizer/config；`Qwen/Qwen3-1.7B` 默认会进 HuggingFace cache，不会在当前目录生成 `Qwen/`。建议用 `python scripts/download_model.py --repo_id Qwen/Qwen3-1.7B --local_dir models/Qwen3-1.7B` 预下载，再把 `--model_name_or_path` 指向本地目录。
- **`AttributeError: shape` 出现在 `model.generate()`**：这是 tokenizer 返回了 `BatchEncoding`，但旧脚本把它当作 `input_ids` Tensor 传给了 `generate()`。新版脚本已改成 Qwen 官方 quickstart 风格：先渲染 chat template，再用普通 tokenizer 调用生成 `model_inputs`，最后执行 `model.generate(**model_inputs)`，适配 Qwen2.5/Qwen3 等模型。
- **模型下载失败**：确认服务器网络、模型名、HuggingFace 登录状态和本地缓存权限。
- **数据格式不合法**：确认 JSONL 每行都是 JSON object，且包含非空 `text` 或非空 `messages`。
- **bitsandbytes 报错**：优先检查 CUDA 版 PyTorch、bitsandbytes 版本和 GPU 是否可见；QLoRA 通常需要 CUDA GPU。
