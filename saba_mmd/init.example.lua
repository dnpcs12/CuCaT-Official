-- Copy to your runtime working directory as init.lua, then edit these paths.
MSAA = { Enable = false, Count = 4 }
Commands = {
    { Cmd = "open", Args = { "C:/assets/model.pmx" } },
    { Cmd = "open", Args = { "C:/assets/motion.vmd" } },
    { Cmd = "open", Args = { "C:/assets/camera.vmd" } },
    -- output, seconds, fps, audio, width, height
    { Cmd = "exportVideo", Args = { "preview.mp4", "1", "30", "C:/assets/audio.wav", "640", "360" } },
    { Cmd = "quit", Args = {} },
}
