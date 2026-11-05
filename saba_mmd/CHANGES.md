# Source and packaging changes

Imported from the authors' internal saba_mmd working snapshot on 2026-09-10. That snapshot has no Git metadata identifying the original upstream revision.

Confirmed CuCaT integration capabilities (not an exhaustive upstream diff):

- mmd_target_dump.cpp: camera target extraction, global bone transform export, legacy target option.
- Viewer.cpp / Viewer.h and related viewer sources: video export through ffmpeg, animation reset commands, batch command execution.
- Existing Python wrappers: worker leases and isolated render working directories.

Packaging changes in this release:

- Replaced top-level and external CMake configuration with the two required executable targets.
- Included only Bullet 3.27 LinearMath, BulletCollision, BulletDynamics and common headers; added minimal dependency CMake configuration.
- Compiled Bullet with /fp:fast on MSVC, as Bullet's own CMake build does (USE_MSVC_FAST_FLOATINGPOINT). Physics-driven bones depend on it; with it, mmd_target_dump -legacy-targets-v1 reproduces the paper's body targets exactly.
- Retained Saba and SabaViewer library source files without algorithm changes.
- Raised the bundled GLFW CMake minimum from 2.8.12 to 3.5 for CMake 4 compatibility. Other GLFW source files are unchanged.
- Bundled required viewer resources and copied them next to the renderer on build.
- Excluded SABA tests, example applications, mmd2obj, glslang/Vulkan, dependency demos, cached builds and user assets.
- Added a repository-relative Windows build helper and direct Lua render example.
- Video export (`ExportVideo` in Viewer.cpp/Viewer.h) sends raw RGB frames to ffmpeg's stdin instead of writing one PPM file per frame and encoding afterwards, with the same encoder settings. A 162 s, 1280x720 preview took 29 s instead of 46 s and no longer writes about 13 GB of temporary frames. On Windows the command runs through `_wpopen`, so WAV and MP4 paths with non-ASCII characters work.

SOURCE_SNAPSHOT.json contains the hashes at import time. The GLFW CMake edit and the video export change above intentionally change their imported hashes. New packaging/documentation files are not in that imported-file list.
