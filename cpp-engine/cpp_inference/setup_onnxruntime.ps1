param(
    [string]$Version = "1.24.4",
    [string]$Arch = "x64"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$thirdPartyDir = Join-Path $scriptDir "third_party"
New-Item -ItemType Directory -Force -Path $thirdPartyDir | Out-Null

$zipName = "onnxruntime-win-$Arch-$Version.zip"
$url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$zipName"
$zipPath = Join-Path $thirdPartyDir $zipName
$extractDir = Join-Path $thirdPartyDir "onnxruntime-win-$Arch-$Version"

if (!(Test-Path $extractDir)) {
    Write-Host "Downloading ONNX Runtime SDK from $url"
    Invoke-WebRequest -Uri $url -OutFile $zipPath

    Write-Host "Extracting SDK to $thirdPartyDir"
    Expand-Archive -Path $zipPath -DestinationPath $thirdPartyDir -Force
} else {
    Write-Host "ONNX Runtime SDK already present at $extractDir"
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Use ONNXRUNTIME_ROOT: $extractDir"
Write-Host ""
Write-Host "Example:"
Write-Host "  `$env:ONNXRUNTIME_ROOT='$extractDir'"
Write-Host "  & '.venv/Scripts/cmake.exe' -S cpp_inference -B cpp_inference/build-mingw -G 'MinGW Makefiles'"
Write-Host "  & '.venv/Scripts/cmake.exe' --build cpp_inference/build-mingw -j"
