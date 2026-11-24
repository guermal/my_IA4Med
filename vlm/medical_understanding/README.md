# Chest X-ray report generation with Qwen2.5-VL-3B (on hold)

Fine-tunes [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) to write the
**findings section of a radiology report** from a chest X-ray, acting as a radiologist.

> **Status: on hold.** The pipeline is written and has trained for a few steps, but no run has finished.
> This part is due to be redesigned; this README documents the code as it is.

## How it works

| Part | Choice (from `main.py`) |
|---|---|
| Model | Qwen2.5-VL-3B-Instruct, 4-bit NF4 quantisation with double quantisation (bitsandbytes); the vision tower stays unquantised |
| Fine-tuning | LoRA, r = 8, alpha = 16, dropout 0.05, on `q_proj` and `v_proj` |
| Trainer | TRL `SFTTrainer` with a custom collate function |
| Loss | next-token cross-entropy on the **report tokens only**: system prompt, user turn, image and padding tokens are masked out |
| Schedule | 3 epochs, batch 4 × gradient accumulation 8, LR 2e-4, max grad norm 0.3, bf16 |
| Logging | Weights & Biases, evaluation every 10 steps, checkpoint every 20 steps |

Each sample is a 3-turn chat:
1. **system:** act as a radiologist and write the findings section;
2. **user:** the X-ray image + "This is a {Frontal/Lateral} view of a chest x-ray, provide a clear medical diagnosis…";
3. **assistant:** the reference report text (target).

## Data
Expected on disk (paths are constants in `datasat_loader.py`):
- images: `D:\chestxray_dataset\resized_images\256\<image_id>`
- labels: `D:\chestxray_dataset\cleaned_dataset.csv` with columns `image_id`, `projection`
  (`Frontal` / `Lateral`) and `org_caption` (the report).

`split_dataset()` in `datasat_loader.py` makes an 80/20 split per projection (seed 42) and writes
`train_dataset.csv` / `test_dataset.csv` next to the source CSV.

## Running

```bash
conda activate <env with requirements.txt from this folder>
python main.py                                 # default attention: sdpa
python main.py --attn-implementation eager     # or flash_attention_2 if flash-attn is installed
```

## Known issues (to address in the redesign)
- **Too slow on a 6 GB GPU.** The furthest run reached step 10 of 546 in 2 h 49 min (~988 s/step,
  ~147 h projected). The GTX 1660 Ti also has no native bf16.
- **`push_to_hub=True`**: a finished run would upload the model to the Hugging Face Hub.
- **Possible patient leakage**: the split is per image, so frontal and lateral views of the same study can
  land in different splits.
- **No evaluation script**: `generate_text_from_sample()` exists but nothing calls it, and no report
  metrics (e.g. BLEU/ROUGE or clinical-label agreement) are computed yet.
- The output folder `qwen2.5-vl-3b-instruct-trl-sft-chestxray/` is empty (no checkpoint was reached).

## Files

| File | Purpose |
|---|---|
| `datasat_loader.py` | train/test split, `ChestXrayDataset` building the chat messages |
| `main.py` | model loading (4-bit + LoRA), collate function with label masking, `SFTTrainer` setup |
| `requirements.txt` | package versions the runs used |
