# LLM Lab: Unsloth SFT / RL 训练脚手架

这个仓库现在改成 **Unsloth 优先** 的实验脚手架，目标是只做模型外部的 LoRA/QLoRA SFT 与 RL（GRPO）训练，不修改模型内部结构。默认模型路径是你已下载好的本地目录 `models/Qwen3-1.7B`；训练时仍可通过 `--load_in_4bit` 让 Unsloth 以 4-bit 方式加载，适合在 RTX 3090 这类单卡环境中先跑通 LoRA/QLoRA。

参考的官方 Unsloth conversational notebook 使用了以下核心流程：

1. `FastLanguageModel.from_pretrained(...)` 加载模型与 tokenizer。
2. `FastLanguageModel.get_peft_model(...)` 注入 LoRA adapter。
3. 使用 TRL 的 `SFTTrainer` / `GRPOTrainer` 训练。
4. 用 `model.save_pretrained(...)` 保存 LoRA adapter，或用 `save_pretrained_merged(...)` 导出合并模型。

## 项目结构

当前 git 仓库中的代码与配置文件结构如下（`models/`、`data/`、`outputs/` 等本地大文件/运行产物目录不提交到 git）：

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── infer_qwen3_1p7b.yaml
│   ├── train_grpo_qwen3_1p7b.yaml
│   ├── train_lora_qwen3_1p7b.yaml
│   ├── train_qlora_qwen3_1p7b.yaml
│   └── train_reinforce_qwen3_4b.yaml
├── notebooks/
│   └── unsloth_sft_grpo_qwen3_v2.ipynb
├── scripts/
│   ├── batch_api_infer.py
│   ├── batch_infer_lora.py
│   ├── batch_infer_transformers.py
│   ├── check_cuda.py
│   ├── download_model.py
│   ├── infer_lora.py
│   ├── infer_transformers.py
│   ├── register_jupyter_kernel.py
│   ├── train_grpo.py
│   ├── train_lora.py
│   ├── train_qlora.py
│   └── train_reinforce_rl.py
└── src/llm_lab/
    ├── __init__.py
    ├── data.py
    ├── elm_eval.py
    ├── model_utils.py
    ├── rl_reinforce.py
    ├── train_utils.py
    └── unsloth_utils.py
```

> `scripts/train_lora.py` 是主 SFT 入口；`scripts/train_qlora.py` 作为兼容入口，默认打开 `--load_in_4bit` 并调用同一套 Unsloth SFT 逻辑。

如果你想像官方 Colab 参考代码那样在 notebook 里逐 cell 跑，Notebook 默认会从 `models/Qwen3-1.7B` 读取本地模型。**不要重新创建环境**；把你已经装好依赖的当前 conda 环境注册成 Jupyter kernel 即可：

```bash
conda activate llm-lab   # 换成你现在已经装好依赖的环境名
python scripts/register_jupyter_kernel.py --name llm-lab --display-name "Python (llm-lab)"
```

如果提示 `No module named ipykernel`，只需要在这个已装好的环境里补一个很小的 kernel 包，不需要重装 Unsloth/Torch：

```bash
python -m pip install ipykernel
python scripts/register_jupyter_kernel.py --name llm-lab --display-name "Python (llm-lab)"
```

然后用任意已有 Jupyter 服务打开 notebook，并在页面菜单里选择：`Kernel -> Change Kernel -> Python (llm-lab)`。如果当前环境本身也装了 notebook，可以这样启动；若没有 `jupyter` 命令，优先用 `python -m notebook`：

```bash
# 推荐：启动 notebook 前指定物理 GPU，例如使用 1 号卡
CUDA_VISIBLE_DEVICES=1 python -m notebook notebooks/unsloth_sft_grpo_qwen3.ipynb
```

如果服务器已经有公共 Jupyter/JupyterLab，也可以不用在 `llm-lab` 里安装 notebook；只要上面注册了 kernel，打开页面后切到 `Python (llm-lab)` 即可。GPU 仍然建议在启动 Jupyter 服务前指定，或者在 notebook 第 0 个代码 cell 里设置 `SELECTED_GPU = "1"`；注意必须在 import `torch` / `unsloth` 之前设置，若已经运行过后面的 cell，请先 Restart Kernel。

这个 notebook 里保留了 Unsloth 原生写法：`FastLanguageModel.from_pretrained(...)`、`FastLanguageModel.get_peft_model(...)`、TRL `SFTTrainer/GRPOTrainer`、response-only 与 full-loss SFT、保存/合并 adapter、单条推理、批量推理、JSON/JSONL 读写与 GRPO reward 示例。

> 你的实验数据如果已经放在 `data/split_data/toy_train.json` 这类 JSON array，推荐先用 notebook：它会先把原始数据清洗成只保留 `prompt_id` / `claim_id` / `prompt` / `groundtruth` / `condition` 的 processed JSON，再用 processed train 做 SFT、processed test 做批量推理；预测输出也只保留这些关键字段并新增 `model_output` / `parsed_score`。Qwen3 的 `<think>...</think>` 会默认通过 `enable_thinking=False` 关闭，避免训练文本里混入空 thinking 标签。

## 创建环境与安装依赖

建议在服务器上创建独立 conda 环境：

```bash
conda create -n llm-unsloth python=3.10 -y
conda activate llm-unsloth
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 仍默认使用 `torch==2.5.1+cu121`，用于兼容常见 CUDA 12.x 驱动的 RTX 3090 服务器。如果你的驱动更新、或 Unsloth 当前版本要求不同的 PyTorch/CUDA 组合，请优先按 Unsloth 官方安装页生成的命令重装 PyTorch / Unsloth。

安装后先检查 CUDA：

```bash
python scripts/check_cuda.py
```

## 数据格式

SFT 支持三种输入：

```json
{"prompt": "问题", "groundtruth": "答案"}
```

```json
{"messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}]}
```

```json
{"text": "已经渲染好的完整训练文本"}
```

默认会做 **response-only loss**：`prompt` / user messages 只作为上下文，label 会被 mask 为 `-100`，loss 只算 assistant answer。若需要对 prompt 也计算 loss，训练时加 `--loss_on_prompt`。

GRPO / RL 使用同一份 `prompt + groundtruth` 或 `messages` 数据。脚本会生成 TRL `GRPOTrainer` 需要的 `prompt` 列，并把答案放到 `answer` 列供 reward function 使用。


## Post-SFT 自定义 REINFORCE RL

SFT 流程不需要改动。SFT 结束后可以单独运行自定义 claim-group REINFORCE：

```bash
python scripts/train_reinforce_rl.py \
  --model_name_or_path models/Qwen3-4B \
  --train_file data/processed_data/processed_train_messages.json \
  --output_dir outputs/qwen3_4b_elm_reinforce
```

`--model_name_or_path` 也可以指向 SFT 后 merge LoRA 得到的模型目录。RL 单位是完整 claim group：脚本会按 `prompt_id`/`claim_id` 分组，只保留同时包含 HHs、HHw、HLs、HLw、LHs、LHw、LLs、LLw 8 个条件的 group。生成时会删除最后一轮 assistant gold answer，只保留 system/user 并加 assistant generation prompt；completion 仍是完整 assistant response，reward 只从 `My attitude score toward this proposal is: X` 中解析 1–11 分数。

每条 prompt 最多重生成 3 次；同一 rollout 内 8 个 condition 会按 batch 一次性生成，解析失败的 condition 再批量重试，8 个条件全部解析成功后才计算 ELM reward；否则该 rollout 使用解析失败 group reward。训练日志持续写入 `outputs/.../rl_training_history.json`，每个 epoch 保存 `checkpoint-epoch-{epoch}`，训练结束保存 `final_checkpoint`。RL 默认使用更随机的 sampling（temperature=1.0/top_p=0.95）、`logprob_reduction=sum`，并默认使用 `reward_baseline_mode=group_mean` 按同一 claim group 的 rollout reward 计算 advantage（也可切到 `global_reward_baseline=0.0`）。若配置 `valid_file`，每个 epoch 后会保存 valid predictions、valid ELM stats、`rl_valid_eval_history.json`，并按 valid `Delta_Arg`/`Delta_Src` 保存 `best_rl_checkpoint`。

如果 notebook 中 RL cell 长时间没有输出，通常是在 `AutoModelForCausalLM.from_pretrained(...)` 加载 4B 模型或分配显存；当前 notebook 已在 tokenizer/model/group loading、每个 epoch、每个 group/rollout/condition generation attempt 前加入 `flush=True` 进度输出，便于区分“正在加载/生成”和“卡住”。

注意：不要在 24GB 单卡上做 Qwen3-4B full-parameter REINFORCE。RL 需要 generation 后再对 completion 做带梯度 forward/backward，并且 AdamW optimizer state 会额外占用大量显存；这比 SFT LoRA/QLoRA 更容易 OOM。默认 RL 配置现在会在 policy model 上再挂一个新的 LoRA adapter，只训练 adapter 参数；如果仍然 OOM，再把 `load_in_4bit` 改成 `true`，并保持 `use_lora: true`。

## Unsloth SFT 训练

默认 4-bit LoRA SFT：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --model_name_or_path model/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_unsloth_lora \
  --max_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4
```

兼容旧命令的 QLoRA 入口：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora.py \
  --model_name_or_path model/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_unsloth_qlora
```

常用参数：

- `--load_in_4bit / --no-load_in_4bit`：是否 4-bit 加载。
- `--dtype auto|float16|bfloat16|float32`：`auto` 让 Unsloth 自动选择。
- `--lora_r`、`--lora_alpha`、`--lora_dropout`、`--target_modules`：LoRA 设置。
- `--chat_template llama-3.1|chatml|...`：可选，用 Unsloth 的 chat template 覆盖 tokenizer 默认模板。
- `--save_method lora|merged_16bit|merged_4bit`：默认只保存 LoRA adapter；需要部署时可导出合并模型。

## Unsloth GRPO / RL 训练

仓库新增了一个轻量 GRPO 入口，适合先验证 RL 管线。默认内置 reward 是 `contains`：如果标准答案出现在模型输出中，reward 为 1，否则为 0。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py \
  --model_name_or_path model/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_unsloth_grpo \
  --max_length 1024 \
  --max_prompt_length 768 \
  --max_completion_length 256 \
  --num_generations 2 \
  --max_steps 100 \
  --reward_type contains
```

内置 reward 类型：

- `contains`：标准答案是生成结果的子串即得分。
- `exact`：归一化后完全一致才得分。
- `numeric`：抽取最后一个数字，数字一致得分。

实际研究中建议把 `scripts/train_grpo.py` 里的 `make_reward_func(...)` 替换成你的任务 reward，例如格式 reward、事实一致性 reward、外部 evaluator reward 或多 reward 加权组合。

如果你安装并配置了 vLLM，可以加 `--fast_inference --gpu_memory_utilization 0.6` 让 Unsloth/GRPO 使用更快的生成路径。

## Unsloth LoRA 推理

训练后直接加载 adapter 目录做单条推理：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_lora.py \
  --model_name_or_path outputs/qwen3_1p7b_unsloth_lora \
  --prompt "请用一句话解释什么是大语言模型。" \
  --max_new_tokens 128
```

如果需要使用特定 chat template：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_lora.py \
  --model_name_or_path outputs/qwen3_1p7b_unsloth_lora \
  --chat_template chatml \
  --prompt "请给出一个简短回答。"
```

### Unsloth 批量推理

batch infer 没有删：原来的 `scripts/batch_infer_transformers.py` 还在；这次另外补了 Unsloth/LoRA adapter 版本 `scripts/batch_infer_lora.py`，用于直接加载 Unsloth 训练保存的 adapter 目录。输入支持 JSON array 和 JSONL，字段可以是 `prompt` 或标准 `messages`；如果 `messages` 里最后一条是 assistant/respondent 答案，推理时会自动去掉，只把 system/user 作为 prompt。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_infer_lora.py \
  --model_name_or_path outputs/qwen3_1p7b_unsloth_lora \
  --input_file data/processed_data/processed_test_messages.json \
  --output_file outputs/qwen3_1p7b_unsloth_batch_outputs.json \
  --batch_size 4 \
  --max_new_tokens 64 \
  --output_field output \
  --overwrite
```

如果每条 prompt 需要重复生成多次，用 `--num_repeats`：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_infer_lora.py \
  --model_name_or_path outputs/qwen3_1p7b_unsloth_lora \
  --input_file data/processed_data/processed_test_messages.json \
  --output_file outputs/qwen3_1p7b_unsloth_batch_outputs_repeat5.json \
  --batch_size 4 \
  --max_new_tokens 64 \
  --num_repeats 5 \
  --output_field output \
  --overwrite
```

当 `--num_repeats > 1` 时，输出字段会保存为 list；如果 `--num_repeats 1` 但也想保存 list，可以加 `--always_list_output`。

## 保留的 Transformers 脚本

`infer_transformers.py`、`batch_infer_transformers.py`、`download_model.py`、`batch_api_infer.py` 仍保留，方便下载模型、批量推理或调用 API。但训练主线已经迁移到 Unsloth：

- SFT：`scripts/train_lora.py` / `scripts/train_qlora.py`
- RL：`scripts/train_grpo.py`
- LoRA 推理：`scripts/infer_lora.py`
- 批量推理：`scripts/batch_infer_lora.py`（Unsloth/adapter）或 `scripts/batch_infer_transformers.py`（Transformers/base model）

## 常见问题

### 1. CUDA 不可用或 bitsandbytes 报错

先运行：

```bash
python scripts/check_cuda.py
```

确认 PyTorch CUDA build 不高于 NVIDIA 驱动支持版本。RTX 3090 服务器如果驱动只支持 CUDA 12.2 左右，通常不要安装 `+cu130` 的 torch wheel。


### 2. CUDA out of memory / GPU 只剩几百 MB

如果错误类似：

```text
CUDA out of memory. Tried to allocate 608.00 MiB. GPU 0 has ... 300.62 MiB free.
```

这通常说明 **当前进程看到的 GPU 0 已经被别的进程占满**，而不是 Qwen3-1.7B 本身需要特别大的 batch。先用下面命令看物理 GPU 占用：

```bash
nvidia-smi
```

然后二选一：

```bash
# 推荐：启动 Jupyter/脚本前选择空闲物理 GPU，例如 1 号卡
CUDA_VISIBLE_DEVICES=1 jupyter notebook notebooks/unsloth_sft_grpo_qwen3.ipynb
CUDA_VISIBLE_DEVICES=1 python scripts/train_lora.py --model_name_or_path model/Qwen3-1.7B
```

或在 notebook 第 0 个 code cell 中把 `SELECTED_GPU = "1"` 改成空闲 GPU 号，并 **Restart Kernel** 后从第 0 个 cell 重新运行。脚本和 notebook 现在都会在加载模型前检查可见 GPU 的空闲显存；如果太低，会提前给出选择 GPU 的提示。

为了先跑通，notebook 默认使用较保守参数：`MAX_SEQ_LENGTH=512`、`INFER_BATCH_SIZE=1`。确认 GPU 空闲且流程跑通后，再逐步调大。

### 3. 模型下载慢

可预先下载模型：

```bash
python scripts/download_model.py \
  --repo_id Qwen/Qwen3-1.7B \
  --local_dir model/Qwen3-1.7B
```

然后训练时传本地路径：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --model_name_or_path model/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl
```

### 4. GRPO reward 不涨

先检查 reward 是否过稀疏。`contains` / `exact` 只是为了跑通管线的 baseline；正式实验需要更贴近任务目标的 reward，并且通常要跑足够多 step 才能观察趋势。
