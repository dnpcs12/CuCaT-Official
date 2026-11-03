# Third-party notices

SABA originates from benikabocha/saba. Its MIT notice is preserved in [LICENCE](LICENCE). This local snapshot has no recoverable upstream base commit. CuCaT's repository-level LICENSE does not replace the notices for bundled components.

Bundled dependencies: Bullet 3.27 (rigid-body subset), GLFW 3.2.1, fcpp, gl3w/Khronos headers, GLI, GLM, Dear ImGui and ImGuizmo, nlohmann/json, Lua, sol2, spdlog, stb, tinyobjloader, tinyddsloader and tinyxfileloader. Existing upstream source/header notices are retained. Standalone notices available in the source snapshot are preserved under `license/`; Bullet's notice is also in `external/bullet/LICENSE.txt`. Header-distributed components retain their embedded notices.

Viewer resources include SABA's toon textures and shaders, and the mgenplus font with its SIL Open Font License in `license/mgenplus/`.

ffmpeg/ffprobe and the Microsoft build tools are external prerequisites, not bundled binaries. Users supply their own PMX/PMD, VMD, audio, and optional scene assets.

## Redistribution notes

Checked on 2026-09-11 against the [upstream SABA license](https://github.com/benikabocha/saba/blob/master/LICENCE). SABA is MIT-licensed: modification and redistribution are permitted provided the original copyright and permission notice accompany copies or substantial portions. Preserve `LICENCE` and source notices when publishing this modified source snapshot; an acknowledgment alone does not replace the license text.

Bundled components keep their own terms. For example, Bullet and GLFW notices require preserved attribution and identification of altered sources; the GLFW build-file change is documented in `CHANGES.md`. The mgenplus font retains its SIL Open Font License. Preserve the `license/` directory and embedded dependency notices rather than treating the entire folder as newly authored CuCaT code.

This check establishes the permission in SABA's license and the presence of the bundled notices; it is not an exhaustive file-by-file provenance or license audit of every third-party resource. The original SABA base commit of the local snapshot is unknown. Character models, dance motion, music, and DCM data are separate assets and are not covered by SABA's MIT permission.
