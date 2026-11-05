# Public Release Manifest

Included:

- End-to-end unseen inference pipeline: `music_processing/run_unseen_mmd_timing_pipeline.py`
- Local web demo for the same pipeline: `music_processing/serve_unseen_mmd_pipeline.py` with its pages (`unseen_mmd_pipeline_viewer.html` with Generate and Edit tabs and the Edit tab's `unseen_live_preview.js`, `unseen_mmd_debug_viewer.html`, `final_cut_keyframe_viewer.html`)
- V3.1 shot-plan inference modules under `shotplan_transformer/`
- CP-SAT optimization modules under `shotplan_transformer/`
- V3/V3.1 shot-plan solver and VMD writer under `camera_autoshot/`
- `final model/` folder layout and README (checkpoint files themselves are distributed via Google Drive)
- Distance-level duration stats used by the default optimizer
- Training: stage-1 dataset builders and training scripts, shot-plan / interpolation dataset builder and training scripts, V3.1 label extraction (target dump, GT camera pruning, reverse, DCM clip split), and the exact training configs under `configs/`
- Song-level split files under `datasets/`
- DCM test-set evaluation under `evaluation/`: test-clip stage-1 timing from the released timing models, shot-plan inference, top-k export and CP-SAT, VMD export, the DanceCamAnimator metric adapter, and `patch_dca.py` with the evaluation fixes applied to a DanceCamAnimator checkout
- DCM test clip list (`datasets/dcm_test_clips.json`) and the per-song body-target models (`datasets/dcm_target_models.csv`)
- SABA Python render wrappers
- Minimal custom SABA C++ source, viewer resources, required vendored dependencies, notices, and Windows build instructions

Excluded:

- Checkpoint files (`final model/**/*.pt`, `*.npz`), distributed via Google Drive
- Training datasets and the DCM data itself
- Training labels (distributed via Google Drive as CuCaT_Training_Labels.zip)
- Body-target dumps of all DCM songs (distributed via Google Drive as CuCaT_Target_Dumps.zip)
- Generated test-clip timing, shot plans, cameras, and metrics (`evaluation/` regenerates them)
- Full experiment result folders
- Rendered MP4 outputs
- Raw MMD asset folders
- SABA binaries, build caches, tests, optional example applications, and glslang/Vulkan dependencies
- Python caches and temporary worker folders
