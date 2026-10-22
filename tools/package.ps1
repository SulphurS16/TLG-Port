# Builds the player package:
#   dist\TLG-Port-<version>\     tlg.exe, tlg-launcher.exe, README.txt
#   dist\TLG-Port-<version>.zip  (the same files)
# README.txt is the player README (with the port's copyright and license
# notice), then the third-party notices of the two programs
# (THIRD_PARTY_NOTICES.txt, made by tools\notices.py from what was just built),
# then the text of the port's license, the GNU GPL version 3. The programs are built as a
# release (TLG_RELEASE_BUILD: an installed layout unless a whole checkout
# surrounds them, crates\config\src\paths.rs), so no marker file is needed.
# The executables carry no debug information, and the paths of the machine that
# builds them (source, Cargo cache) are remapped; the package is refused if one
# of those paths, the user profile or the user name is still found in them.
# Only the files listed here go in (an allowlist): nothing of the game, no keys,
# saves, caches or logs.
#   .\tools\package.ps1 [-NoBuild] [-Offline] [-Pgo <profile>]
# -Pgo: tlg.exe built with that profile (profile-guided optimization, made
# with tools\pgo.ps1 and tools\pgo_train.sh from this source; the launcher
# without it).
param([switch]$NoBuild, [switch]$Offline, [string]$Pgo)
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'env.ps1')
$ver = (Select-String -Path (Join-Path $root 'Cargo.toml') -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
if (-not $NoBuild) {
  # (Encoded flags: the paths may contain spaces.)
  $flags = @('-C', 'target-feature=+crt-static', "--remap-path-prefix=$root=tlg", "--remap-path-prefix=$($env:CARGO_HOME)=cargo")
  $net = if ($Offline) { '--offline' } else { '--locked' }
  $env:TLG_RELEASE_BUILD = '1'
  # (From the root: Cargo reads .cargo\config.toml from the current folder.)
  Push-Location $root
  if ($Pgo) {
    if (-not (Test-Path $Pgo)) { Pop-Location; Write-Host "no profile at $Pgo"; exit 1 }
    $env:CARGO_ENCODED_RUSTFLAGS = ($flags + @("-Cprofile-use=$((Resolve-Path $Pgo).Path)")) -join [char]0x1f
    & cargo build --profile dist $net -p tlg
    $code = $LASTEXITCODE
    if ($code -eq 0) {
      $env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]0x1f
      & cargo build --profile dist $net -p tlg-launcher
      $code = $LASTEXITCODE
    }
  } else {
    $env:CARGO_ENCODED_RUSTFLAGS = $flags -join [char]0x1f
    & cargo build --profile dist $net -p tlg -p tlg-launcher
    $code = $LASTEXITCODE
  }
  Pop-Location
  Remove-Item Env:CARGO_ENCODED_RUSTFLAGS
  Remove-Item Env:TLG_RELEASE_BUILD
  if ($code -ne 0) { Write-Host "build failed"; exit 1 }
  # The third-party notices of what was just built (THIRD_PARTY_NOTICES.txt).
  if ($Offline) { $env:CARGO_NET_OFFLINE = 'true' }
  & python -I (Join-Path $PSScriptRoot 'notices.py')
  $code = $LASTEXITCODE
  if ($Offline) { Remove-Item Env:CARGO_NET_OFFLINE }
  if ($code -ne 0) { Write-Host "notices failed"; exit 1 }
}
$bin = Join-Path $root 'target\x86_64-pc-windows-gnullvm\dist'
$name = "TLG-Port-$ver"
$dist = Join-Path $root 'dist'
$final = Join-Path $dist $name
# (Built in a fresh staging folder, then swapped in: a launcher still
# running from the previous package locks its file, and the zip must not
# take that one. Only this script's own earlier output is replaced.)
$out = Join-Path $dist ".stage-$name"
if (Test-Path $out) { Remove-Item -Recurse -Force $out -ErrorAction Stop }
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($f in 'tlg.exe', 'tlg-launcher.exe') {
  $src = Join-Path $bin $f
  if (-not (Test-Path $src)) { Write-Host "missing $src"; exit 1 }
  Copy-Item $src $out -ErrorAction Stop
  if ((Get-FileHash $src).Hash -ne (Get-FileHash (Join-Path $out $f)).Hash) { Write-Host "$f was not copied whole"; exit 1 }
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
# README.txt: the player README, the notices, the license (CRLF, UTF-8).
$read = { param($p) ([IO.File]::ReadAllText($p)).Replace("`r`n", "`n").TrimEnd() }
$rule = '=' * 78
$readme = (& $read (Join-Path $root 'docs\PLAYER_README.txt')) + "`n`n`n$rule`nThird-party notices`n$rule`n`n" + (& $read (Join-Path $root 'THIRD_PARTY_NOTICES.txt')) + "`n`n`n$rule`nLicense: GNU General Public License, version 3`n$rule`n`n" + (& $read (Join-Path $root 'LICENSE')) + "`n"
[IO.File]::WriteAllText((Join-Path $out 'README.txt'), $readme.Replace("`n", "`r`n"), (New-Object Text.UTF8Encoding $true))
# Allowlist: nothing else may be in the package.
$allowed = 'tlg.exe', 'tlg-launcher.exe', 'README.txt'
$extra = @(Get-ChildItem -Recurse -File $out | Where-Object { $allowed -notcontains $_.Name })
if ($extra.Count -gt 0) { Write-Host "unexpected files in the package: $($extra.FullName -join ', ')"; exit 1 }
$zip = Join-Path $dist "$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path (Join-Path $out '*') -DestinationPath $zip
$size = (Get-ChildItem -File $out | Measure-Object Length -Sum).Sum
$count = (Get-ChildItem -File $out).Count
# The staged files become dist\TLG-Port-<version> unless the earlier folder is
# in use (renaming it fails while a program from it runs): then they stay staged.
if (Test-Path $final) {
  if (Test-Path "$final.old") { Remove-Item -Recurse -Force "$final.old" -ErrorAction SilentlyContinue }
  try { Rename-Item $final "$name.old" -ErrorAction Stop } catch { }
}
if (-not (Test-Path $final)) {
  Move-Item $out $final -ErrorAction Stop
  if (Test-Path "$final.old") { Remove-Item -Recurse -Force "$final.old" -ErrorAction SilentlyContinue }
  $out = $final
} else {
  Write-Host "$final is in use (a program from it is running): the new package's files are in $out; the zip is complete"
}
Write-Host ("package {0}: {1} files, {2:N1} MB; zip {3:N1} MB" -f $out, $count, ($size / 1MB), ((Get-Item $zip).Length / 1MB))
