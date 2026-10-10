# Puts the project-local toolchain (local\tools) on PATH for this PowerShell session.
#   . .\tools\env.ps1
# Nothing here is installed system-wide; see tools\bootstrap.ps1 and docs\PROVENANCE.md.
$root = Split-Path -Parent $PSScriptRoot
$t = Join-Path $root 'local\tools'
$env:RUSTUP_HOME = Join-Path $t 'rustup'
$env:CARGO_HOME = Join-Path $t 'cargo'
$env:RUSTUP_TOOLCHAIN = '1.99.0-x86_64-pc-windows-gnullvm'
$llvm = Join-Path $t 'llvm-mingw-20260908-ucrt-x86_64\bin'
$paths = @((Join-Path $t 'cargo\bin'), $llvm, (Join-Path $t 'python'))
$env:PATH = (($paths + ($env:PATH -split ';' | Where-Object { $_ -and ($paths -notcontains $_) })) -join ';')
# The cc crate (C decoders) uses the llvm-mingw compiler.
$env:CC_x86_64_pc_windows_gnullvm = 'x86_64-w64-mingw32-clang'
$env:AR_x86_64_pc_windows_gnullvm = 'llvm-ar'
$env:CXX_x86_64_pc_windows_gnullvm = 'x86_64-w64-mingw32-clang++'
