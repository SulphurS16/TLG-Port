# Builds the distribution profile and assembles a local player package:
#   dist\TLG-<version>-win64\  tlg.exe, tlg-launcher.exe, tlg-prep.exe, README.txt,
#   LICENSE.txt, THIRD_PARTY.md, CHANGELOG.md, VERSION.txt, tlg-release.txt
#   dist\TLG-<version>-win64.zip   (the same files)
#   dist\symbols-<version>\        (PDBs for crash reports; not in the package)
# Only the files listed here go in: nothing of the game, no keys, saves, caches,
# logs or pipeline caches (an allowlist check refuses anything else). The package
# is for local use; a public release needs the user's separate authorization.
#   .\tools\package.ps1 [-NoBuild]
param([switch]$NoBuild)
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')
$ver = (Select-String -Path (Join-Path $root 'Cargo.toml') -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
if (-not $NoBuild) {
  & cargo build --profile dist --offline -p tlg -p tlg-launcher -p tlg-prep
  if ($LASTEXITCODE -ne 0) { Write-Host "build failed"; exit 1 }
}
$bin = Join-Path $root 'target\x86_64-pc-windows-gnullvm\dist'
$name = "TLG-$ver-win64"
$dist = Join-Path $root 'dist'
$out = Join-Path $dist $name
# (Only this script's own earlier output is replaced.)
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($f in 'tlg.exe', 'tlg-launcher.exe', 'tlg-prep.exe') {
  $src = Join-Path $bin $f
  if (-not (Test-Path $src)) { Write-Host "missing $src"; exit 1 }
  Copy-Item $src $out
}
Copy-Item (Join-Path $root 'docs\PLAYER_README.txt') (Join-Path $out 'README.txt')
Copy-Item (Join-Path $root 'LICENSE') (Join-Path $out 'LICENSE.txt')
Copy-Item (Join-Path $root 'THIRD_PARTY.md') $out
Copy-Item (Join-Path $root 'CHANGELOG.md') $out
$commit = (& git -C $root rev-parse --short HEAD).Trim()
$dirty = & git -C $root status --porcelain --untracked-files=no
$note = if ($dirty) { ' (built with uncommitted changes)' } else { '' }
Set-Content -Path (Join-Path $out 'VERSION.txt') -Value "tlg $ver`r`ncommit $commit$note`r`nbuilt $(Get-Date -Format 'yyyy-MM-dd HH:mm') (local build, not a public release)" -Encoding utf8
Set-Content -Path (Join-Path $out 'tlg-release.txt') -Value "This file marks an installed tlg build: settings and saves go to %APPDATA%\TLG, caches and logs to %LOCALAPPDATA%\TLG.`r`nCreate an empty portable.txt next to tlg.exe to keep everything in a 'user' folder here instead." -Encoding utf8
# Allowlist: nothing else may be in the package.
$allowed = 'tlg.exe', 'tlg-launcher.exe', 'tlg-prep.exe', 'README.txt', 'LICENSE.txt', 'THIRD_PARTY.md', 'CHANGELOG.md', 'VERSION.txt', 'tlg-release.txt'
$extra = @(Get-ChildItem -Recurse -File $out | Where-Object { $allowed -notcontains $_.Name })
if ($extra.Count -gt 0) { Write-Host "unexpected files in the package: $($extra.FullName -join ', ')"; exit 1 }
# Symbols kept beside the package for crash reports.
$sym = Join-Path $dist "symbols-$ver"
New-Item -ItemType Directory -Force $sym | Out-Null
foreach ($f in 'tlg.pdb', 'tlg-launcher.pdb', 'tlg-prep.pdb') { if (Test-Path (Join-Path $bin $f)) { Copy-Item (Join-Path $bin $f) $sym } }
$zip = Join-Path $dist "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path (Join-Path $out '*') -DestinationPath $zip
$size = (Get-ChildItem -File $out | Measure-Object Length -Sum).Sum
Write-Host ("package {0}: {1} files, {2:N1} MB; zip {3:N1} MB" -f $out, (Get-ChildItem -File $out).Count, ($size / 1MB), ((Get-Item $zip).Length / 1MB))
