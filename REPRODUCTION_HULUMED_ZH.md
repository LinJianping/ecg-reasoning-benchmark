# Hulu-Med 复现说明

代码位于 `/root/autodl-tmp/ecg-reasoning-benchmark`，大型资源和独立 Python 环境位于 `/root/autodl-tmp/ecg-repro-assets`。已有 vLLM 环境实际是 `/root/vllm_env`，未更改其中的依赖。

## 指标与版本

[论文表 1](https://arxiv.org/html/2603.14326v1) 中 32B 的 99.42% / 97.20% 指 **GT-RDA**：给定完整正确推理历史后最终诊断的准确率。它们不是直接看 ECG 的诊断准确率，也不是完整推理链成功率。7B 的相应对照值为 **86.87% / 85.83%**；7B 的直接诊断 IDA 为 55.87% / 50.79%。

运行使用官方完整 `default` 推理协议：初始诊断保留真实模型回答，中间各步骤使用真实答案替换历史；最终诊断作答前不提供该最终答案。保持原图像绘制、系统提示、condensed chat、BF16、greedy、max_new_tokens=1024。批量运行按样本逐轮执行，32B 用 Transformers/Accelerate 将模型分布到两张卡；当前没有使用 vLLM 的 tensor parallel。

默认题目为 `data/benchmark-paper-v1/`：由官方 tag `0.0.1` 导出，与论文发布附近的 `04a6a0031338dada079ccd0797a769749a35d7d7` 数据 blob 相同。使用修复后的当前推理代码 `90a53f107023c7c196cbf08de183b871b2a5798c`，不使用 0.0.1 中已确认有初始答案注入问题的推理代码。官方后续修订了部分题目和波形，因此不能承诺得到论文完全相同的两位小数。仓库自身 `data/` 保留当前最新版。

| 数据 | 题目数 | 论文时期唯一 ECG | 当前版唯一 ECG | 两版本并集 |
|---|---:|---:|---:|---:|
| PTB-XL | 3084 | 2868 | 2871 | 2976 |
| MIMIC-IV-ECG | 3359 | 3316 | 3317 | 3416 |

每组包含 17 类诊断。JSONL 只包含问题、答案和记录编号；视觉推理还需要原始 ECG 波形。PTB-XL 只下载上述并集用到的 500Hz `.hea/.dat`，不下载完整数据库。

## 下载来源与限制

Hugging Face 配置和代码使用 `HF_ENDPOINT=https://hf-mirror.com`。权重也可从模型卡链接的[作者 ModelScope 发布仓库](https://modelscope.cn/models/Med-Team/Hulu-Med)下载；脚本仅接受与 HF 固定版本分片 SHA256 和大小完全一致的文件，并在下载结束后逐个校验：

- 7B：`258594714a0d3835eb2c9e4cc165a4242e606d71`
- 32B：`dfac09d0653d00ac974c02a09caacaaab22fa819`

代码、processor 和 tokenizer 始终来自 HF 固定版本，不用 ModelScope 中较旧的 processor 替换。

本容器访问 [MIMIC-IV-ECG 官方页面](https://physionet.org/content/mimic-iv-ecg/1.0/)时，文件区显示所在地区因法律或政策限制无法获取数据，实际波形下载返回 HTTP 403。项目虽然标为 Open Access，但当前不能从此容器下载。已保存访问说明与缺失记录清单；如已有合法获得的本地波形，设置 `MIMIC_ECG_DIR` 为包含 `files/` 的目录即可。PTB-XL 下载来源为 [PhysioNet 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)，校验使用官方 SHA256SUMS 和 WFDB checksum。

## 启动与评测

脚本会清除大小写 HTTP/HTTPS/ALL/FTP 代理变量，不修改全局 `.bashrc`。先打开新的终端：

```bash
cd /root/autodl-tmp/ecg-reasoning-benchmark
# 7B，当前单卡；先每类一条，确认完整多轮流程
HULUMED_ATTN_IMPLEMENTATION=sdpa bash scripts/run_hulumed_7b.sh ptbxl --per-diagnosis 1
# 7B，PTB-XL 全集，默认 FlashAttention 2；已完整写出的样本自动跳过
bash scripts/run_hulumed_7b.sh ptbxl
```

`--limit N` 和 `--per-diagnosis N` 会自动写到独立的 smoke 子目录，不会将少量样本误计为全量实验。输出先写临时文件再原子替换；配置指纹不一致时拒绝混写。SDPA 是模型原生支持的注意力实现，使用时与论文 FlashAttention 2 实现有差异，结果清单会记录。

```bash
# 全集本地 heuristic 评测，无 API 费用
HULUMED_VARIANT=7B bash scripts/evaluate_hulumed.sh heuristic ptbxl
# 每类一条 smoke 的本地评测
HULUMED_VARIANT=7B \
RESULTS_DIR=/root/autodl-tmp/ecg-repro-assets/results/paper-v1/7B/smoke-per-diagnosis-1 \
EVAL_DIR=/root/autodl-tmp/ecg-repro-assets/eval-results/paper-v1/7B/smoke-per-diagnosis-1 \
bash scripts/evaluate_hulumed.sh heuristic ptbxl
```

论文用 `gemini-3-flash-preview`、temperature 0 作裁判。Heuristic 结果需单独标记，不能宣称是同款裁判复现。若要使用 Gemini，在本机安全设置 `GOOGLE_API_KEY` 后，把上面 evaluator 改为 `gemini`；工具可能产生 API 费用，且原 preview 型号未来可能不可用。密钥不保存在项目中。

CSV 位于 `$EVAL_DIR/<evaluator>/ptbxl/total.csv`，目标列为 `gt_reasoning_based_diagnosis_accuracy`；各诊断另有 CSV。整体结果是样本微平均。`--per-diagnosis 1` 选择的各类首条均为阳性，仅用于流程检查，不能据其分数推断全量性能。

本地 heuristic 修复了官方代码未同步的类型名：`finding_identification` 映射到已有 finding 校验器，`diagnostic_decision` 映射到已有 decision 校验器。答案匹配函数保持原样，四项回归测试覆盖类型名、三选项决策、单答案 grounding 列表和多导联回答。没有修改模型生成答案。

## 32B 双卡

32B 权重及环境准备完成后，关机/重启容器并选择 **2 张 H800-80GB**，保留当前数据盘及文件。先用 `nvidia-smi` 确认两卡可见，再执行：

```bash
cd /root/autodl-tmp/ecg-reasoning-benchmark
bash scripts/run_hulumed_2gpu.sh ptbxl --limit 2
bash scripts/run_hulumed_2gpu.sh ptbxl
# 提供合法本地 MIMIC 波形后
MIMIC_ECG_DIR=/path/to/mimic-iv-ecg bash scripts/run_hulumed_2gpu.sh mimic_iv_ecg
```

默认使用 FlashAttention 2；须通过运行环境检查。可显式设置 `HULUMED_ATTN_IMPLEMENTATION=sdpa` 使用 SDPA。32B 启动脚本检查至少两张可见 GPU，默认每卡权重预算 72GiB，拒绝 CPU/disk offload。不会在当前单卡自动启动 32B。

要运行最新版题目，显式设置 `BENCHMARK_DIR=$PWD/data`，并为 `RESULTS_DIR` 和 `EVAL_DIR` 选择新的目录，避免混淆数据版本。

## 断点下载

```bash
source scripts/hulumed_env.sh
python scripts/download_hulumed.py --variant 32B --weight-source modelscope --workers 16
python scripts/download_hulumed.py --variant 7B --weight-source modelscope --workers 8
python scripts/download_ecg_data.py --dataset ptbxl --additional-benchmark-dir "$PWD/data"
```

如只使用 HF 镜像下载权重，把 `--weight-source modelscope` 改为 `--weight-source hf`。下载脚本保留分段进度，最终必须通过权重 SHA256 校验才生成正式 `.safetensors` 文件。

## 全量自动任务与进度

`bash scripts/run_7b_full_pipeline.sh` 默认使用 FlashAttention 2，先验证 PTB 波形，再执行 3084 个样本的完整多轮推理，最后自动进行本地 heuristic 评测并写出 `reproduction-summary.json`。完整多轮 HF 推理耗时明显长于单次诊断，可在后台运行：

```bash
nohup bash scripts/run_7b_full_pipeline.sh > /root/autodl-tmp/ecg-repro-assets/logs/7b-full-pipeline.log 2>&1 < /dev/null &
python scripts/status_hulumed.py
```

`results/paper-v1/7B/pipeline-status.json` 记录验证、推理、评测、完成或失败状态；同一结果目录有进程锁，重复启动不会混写结果。后台任务运行期间请勿重启容器。安装版本记录于 `envs/hulumed-installed-freeze.txt`。WFDB 4.3.0 使用 Pandas 2.3.3，避免 Pandas 3.x 导致的导入错误。
