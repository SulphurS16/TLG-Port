# Downloads and unpacks the project-local toolchain into local\tools (no admin rights, nothing
# installed system-wide). Archives are verified against pinned SHA-256 values.
#   .\tools\bootstrap.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dl = Join-Path $root "local\downloads"
$tools = Join-Path $root "local\tools"
New-Item -ItemType Directory -Force -Path $dl, $tools | Out-Null
$ProgressPreference = 'SilentlyContinue'

$items = @(
  @{ url = "https://github.com/mstorsjo/llvm-mingw/releases/download/20260908/llvm-mingw-20260908-ucrt-x86_64.zip";
     sha = "1bcf74d06b724aeecaa6412ca85f5b26fb1da770e7cdcefa9263c9c5c3ad34b6"; dest = ""; check = "llvm-mingw-20260908-ucrt-x86_64" },
  @{ url = "https://www.python.org/ftp/python/3.13.9/python-3.13.9-embed-amd64.zip";
     sha = "91d828c2da3a029b41699e918674a0cb379c02cf20dab9c501306885f837402a"; dest = "python"; check = "python" }
)
foreach ($it in $items) {
  $file = Join-Path $dl (Split-Path -Leaf $it.url)
  if (-not (Test-Path $file)) { Write-Host "downloading $($it.url)"; Invoke-WebRequest -Uri $it.url -OutFile $file -UseBasicParsing }
  $h = (Get-FileHash -Algorithm SHA256 $file).Hash.ToLower()
  if ($h -ne $it.sha) { throw "hash mismatch for $file ($h)" }
  if (Test-Path (Join-Path $tools $it.check)) { Write-Host "ok $($it.check)"; continue }
  $target = if ($it.dest) { Join-Path $tools $it.dest } else { $tools }
  Expand-Archive -Path $file -DestinationPath $target -Force
  Write-Host "unpacked $($it.check)"
}

# Rust (gnullvm host, minimal profile) into local\tools\rustup + local\tools\cargo.
$env:RUSTUP_HOME = Join-Path $tools 'rustup'
$env:CARGO_HOME = Join-Path $tools 'cargo'
if (-not (Test-Path (Join-Path $tools 'cargo\bin\cargo.exe'))) {
  $ri = Join-Path $dl 'rustup-init.exe'
  $base = 'https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe'
  Invoke-WebRequest -Uri $base -OutFile $ri -UseBasicParsing
  $c = (Invoke-WebRequest -Uri "$base.sha256" -UseBasicParsing).Content
  if ($c -is [byte[]]) { $c = [Text.Encoding]::ASCII.GetString($c) } # pwsh 7 returns bytes for octet-stream
  $want = ($c -split '\s+')[0].Trim().ToLower()
  $h = (Get-FileHash -Algorithm SHA256 $ri).Hash.ToLower()
  if ($h -ne $want) { throw "rustup-init hash mismatch ($h, published $want)" }
  $env:RUSTUP_INIT_SKIP_PATH_CHECK = 'yes'
  # Pinned to the compiler the runtime is built and tested with.
  & $ri -y --no-modify-path --profile minimal --default-host x86_64-pc-windows-gnullvm --default-toolchain 1.99.0
} else { Write-Host "ok rust" }
