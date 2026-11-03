# CuCaT SABA tools

Minimal source release of the SABA snapshot used by CuCaT. It builds two programs:

- `mmd_target_dump`: evaluate PMX/PMD + motion VMD into camera target JSON and optional global bone transforms for dance features.
- `saba_viewer`: preview MMD scenes and export camera videos through Lua commands.

SABA source, viewer shaders/toon textures/font, and the required C/C++ dependencies are bundled. No MMD character, dance, song, prebuilt executable, training data, or build cache is distributed. The existing Python integration wrappers remain in `build/Release/` for compatibility; Git explicitly includes only those three scripts there.

## Build SABA for rendering on Windows

Install Visual Studio 2022 Build Tools with **Desktop development with C++** (MSVC and Windows SDK), and CMake 3.16 or newer on PATH. The bundled dependencies require no network download at configure time.

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\saba_mmd\build_windows.ps1
```

Equivalent commands:

```powershell
cmake -S saba_mmd -B saba_mmd/build -G "Visual Studio 17 2022" -A x64
cmake --build saba_mmd/build --config Release --target mmd_target_dump saba_viewer --parallel 4
```

Outputs: `saba_mmd/build/Release/mmd_target_dump.exe`, `saba_viewer.exe`, and `resource/`. Keep the resource directory when distributing a local build. Run the viewer with its working directory set to `build/Release`, or copy `resource/` into an isolated working directory containing your `init.lua`.

This release supports the tested Windows build. Other platforms have not been validated. Rendering requires a working OpenGL driver plus `ffmpeg` and `ffprobe` on PATH; target extraction does not use ffmpeg or open a viewer window. This minimal build still configures OpenGL because it includes both programs.

## Extract motion and camera targets

Supply your own matching model and motion; their skeletons must be compatible.

```powershell
New-Item -ItemType Directory -Force output | Out-Null
& .\saba_mmd\build\Release\mmd_target_dump.exe `
  "C:/assets/model.pmx" -vmd "C:/assets/motion.vmd" `
  -o "output/target_probe.json" `
  -global-transform "output/dance_GlobalTransform.json" `
  -start 0 -end 299 -step 1
```

Frame bounds are inclusive. The example extracts frames 0 through 299. Use the intended sequence length in your own command. `-legacy-targets-v1` is available for pipelines requiring the older target semantics; the example uses the tool's default semantics.

Global transforms feed dance feature extraction; target probe data feeds shot-plan-to-camera-VMD conversion. If both compatible precomputed JSON files are available, you do not need to run this extractor. The current unseen pipeline accepts `--motion-json`, but looks for camera targets at `<output-root>/<song_id>/intermediate/<song_id>_target_probe.json`.

The repository's unseen pipeline discovers this build automatically:

```powershell
python -m music_processing.run_unseen_mmd_timing_pipeline `
  "C:/assets/input_song" --model-path "C:/assets/model.pmx" `
  --skip-render-preview
```

## Render a short video directly

We used **Nekomata Okayu (猫又おかゆ)** as our default rendering character: [Okayu model on Niconico 3D](https://3d.nicovideo.jp/works/td63648). Download and extract the model separately, preserving its textures and directory structure, and follow the terms provided with the model. The character is not bundled here. Set the example's model path to the extracted PMX file's actual path.

Copy `init.example.lua` to `build/Release/init.lua` and edit its asset/output paths. Install ffmpeg and check `ffmpeg -version` and `ffprobe -version`, then:

```powershell
Push-Location saba_mmd/build/Release
try { .\saba_viewer.exe } finally { Pop-Location }
```

The example opens a model, motion, and camera, exports one second, and quits. Increase the duration as needed. No dome or grid asset is required. Use a separate working directory plus `resource/` for concurrent renders. The Python pipeline's worker leases already provide isolated working directories.

## Scope and provenance

See [THIRD_PARTY.md](THIRD_PARTY.md) for notices, [CHANGES.md](CHANGES.md) for local capabilities and packaging changes, and [VALIDATION.md](VALIDATION.md) for the actual checks performed. `SOURCE_SNAPSHOT.json` records the original copied-file hashes before the packaging edits listed in CHANGES.md. The precise upstream base commit is unknown; this is a source snapshot, not a claimed complete patch against a particular upstream revision.
