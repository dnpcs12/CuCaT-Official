param(
    [string]$Generator = "Visual Studio 17 2022",
    [int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
$sourceRoot = $PSScriptRoot
$buildRoot = Join-Path $sourceRoot 'build'
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw 'Install CMake 3.16 or newer and add it to PATH.'
}
& cmake -S $sourceRoot -B $buildRoot -G $Generator -A x64
if ($LASTEXITCODE -ne 0) { throw 'SABA CMake configure failed.' }
& cmake --build $buildRoot --config Release --target mmd_target_dump saba_viewer --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'SABA build failed.' }
Write-Host "Built target extractor and renderer in $buildRoot/Release"
