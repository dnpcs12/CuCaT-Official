# Validation

2026-09-10, Windows x64, Visual Studio 2022 Build Tools, CMake 4.3.1.

- Configured and compiled from a new build directory using build_windows.ps1 and only the bundled dependency sources. Both mmd_target_dump.exe and saba_viewer.exe built successfully.
- Ran the new extractor on a local PMX and dance VMD for frames 0-29: target JSON had 30 frames and GlobalTransform JSON had 30 frame records.
- Ran the new viewer in a separate working directory containing the copied resource folder and init.lua. Opened a local PMX, motion VMD, and camera VMD; exported a one-second video with audio, then executed animation reset commands and quit.
- ffprobe confirmed 320x180 video, 30 frames, an audio stream, and 1.000000 second duration. Inspected the first decoded frame to confirm scene geometry rendered.
- All imported source/resource files are visible to Git. Generated binaries, intermediate objects, test JSON, media, and logs are ignored.

Local third-party test assets are not included. This smoke test verifies extraction and rendering capability, not paper metric equivalence or full neural inference. No upstream equivalence comparison, long multi-clip render test, or non-Windows build was performed.
