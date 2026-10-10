# Runs cargo with the project-local toolchain and cache (local\tools\cargo).
#   .\tools\cargo.ps1 build --release
# Don't set $ErrorActionPreference = 'Stop' around this: cargo writes progress to stderr.
. (Join-Path $PSScriptRoot 'env.ps1')
& cargo @args
exit $LASTEXITCODE
