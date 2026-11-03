# Final models

Checkpoints of the paper's final models. They are distributed through Google Drive as `CuCaT_Model_Checkpoints.zip`; extract it into the repository root so that this folder holds the files below. The end-to-end pipeline, the evaluation scripts, and `configs/` read them from these paths.

## Timing models (stage 1)

- `cut_transition/best_model.pt`: per-frame cut probability
- `density_stagea/best_model.pt`: keyframe budget (density)
- `keyframe_stage2/best_model.pt`: cut-conditioned transition locator

## Shot-plan models (stages 2 and 4)

- `v3_1_visual_scale_005/best.pt`: shot-plan Transformer
- `v3_1_visual_scale_005/normalization_stats.npz`: feature normalization of the shot-plan Transformer
- `v3_1_interpolation_stage/best.pt`: interpolation (easing) MLP

The training arguments of every checkpoint are in `configs/`.
