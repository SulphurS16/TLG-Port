# Builds the player package:
#   dist\TLG-Port-<version>\     tlg.exe, tlg-launcher.exe, README.txt, LICENSE.txt,
#                                THIRD_PARTY.md, tlg-release.txt
#   dist\TLG-Port-<version>.zip  (the same files)
# The executables carry no debug information, and the paths of the machine that
# builds them (source, Cargo cache) are remapped; the package is refused if one
# of those paths, the user profile or the user name is still found in them.
# Only the files listed here go in (an allowlist): nothing of the game, no keys,
# saves, caches or logs.
#   .\tools\package.ps1 [-NoBuild] [-Offline]
param([switch]$NoBuild, [switch]$Offline)
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')
$ver = (Select-String -Path (Join-Path $root 'Cargo.toml') -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
if (-not $NoBuild) {
  # (Encoded flags: the paths may contain spaces.)
  $env:CARGO_ENCODED_RUSTFLAGS = @('-C', 'target-feature=+crt-static', "--remap-path-prefix=$root=tlg", "--remap-path-prefix=$($env:CARGO_HOME)=cargo") -join [char]0x1f
  $net = if ($Offline) { '--offline' } else { '--locked' }
  & cargo build --profile dist $net -p tlg -p tlg-launcher
  $code = $LASTEXITCODE
  Remove-Item Env:CARGO_ENCODED_RUSTFLAGS
  if ($code -ne 0) { Write-Host "build failed"; exit 1 }
}
$bin = Join-Path $root 'target\x86_64-pc-windows-gnullvm\dist'
$name = "TLG-Port-$ver"
$dist = Join-Path $root 'dist'
$out = Join-Path $dist $name
# (Only this script's own earlier output is replaced.)
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($f in 'tlg.exe', 'tlg-launcher.exe') {
  $src = Join-Path $bin $f
  if (-not (Test-Path $src)) { Write-Host "missing $src"; exit 1 }
  Copy-Item $src $out
}
# Paths of this machine must not be in the executables.
$needles = @($root, $root.Replace('\', '/'), $env:CARGO_HOME, $env:USERPROFILE) | Where-Object { $_ }
if ($env:USERNAME -and $env:USERNAME.Length -ge 4) { $needles += "\$($env:USERNAME)\" }
foreach ($f in 'tlg.exe', 'tlg-launcher.exe') {
  $bytes = [IO.File]::ReadAllBytes((Join-Path $out $f))
  $text = [Text.Encoding]::GetEncoding(28591).GetString($bytes).ToLowerInvariant()
  $wide = [Text.Encoding]::Unicode.GetString($bytes).ToLowerInvariant()
  foreach ($n in $needles) {
    $k = $n.ToLowerInvariant()
    if ($text.Contains($k) -or $wide.Contains($k)) { Write-Host "$f contains a path of this machine ($n); not packaged"; exit 1 }
  }
}
Copy-Item (Join-Path $root 'docs\PLAYER_README.txt') (Join-Path $out 'README.txt')
Copy-Item (Join-Path $root 'LICENSE') (Join-Path $out 'LICENSE.txt')
Copy-Item (Join-Path $root 'THIRD_PARTY.md') $out
Set-Content -Path (Join-Path $out 'tlg-release.txt') -Value "This file marks an installed build: settings and saves go to %APPDATA%\TLG, caches and logs to %LOCALAPPDATA%\TLG.`r`nAn empty portable.txt next to tlg.exe keeps everything in a 'user' folder here instead." -Encoding utf8
# Allowlist: nothing else may be in the package.
$allowed = 'tlg.exe', 'tlg-launcher.exe', 'README.txt', 'LICENSE.txt', 'THIRD_PARTY.md', 'tlg-release.txt'
$extra = @(Get-ChildItem -Recurse -File $out | Where-Object { $allowed -notcontains $_.Name })
if ($extra.Count -gt 0) { Write-Host "unexpected files in the package: $($extra.FullName -join ', ')"; exit 1 }
$zip = Join-Path $dist "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path (Join-Path $out '*') -DestinationPath $zip
$size = (Get-ChildItem -File $out | Measure-Object Length -Sum).Sum
Write-Host ("package {0}: {1} files, {2:N1} MB; zip {3:N1} MB" -f $out, (Get-ChildItem -File $out).Count, ($size / 1MB), ((Get-Item $zip).Length / 1MB))
