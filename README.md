# VLM_projects: medical imaging with computer vision and VLMs

```
computer_vision/                    trained-from-data vision models
  brain_segmentation/               BraTS2020 tumor segmentation: UNet, SwinUNETR, DINOv2/ViT/I-JEPA + DPT
  ChestXRayAnatomySegmentation/     CXAS (third-party clone): chest X-ray anatomy segmentation
vlm/                                vision-language models
  segmentation/                     text-prompted segmentation (BiomedParse v2) on the same BraTS split;
                                    its README also surveys SAT, MedPLIB, UniBiomed, MedSAM(2), nnInteractive…
  medical_understanding/            Qwen2.5-VL chest X-ray report generation (on hold, to be discussed)
```

The CV and VLM segmentation sides share one patient split and one set of metrics
(`computer_vision/brain_segmentation/{data,metrics}.py`), so their results go in the same table.
