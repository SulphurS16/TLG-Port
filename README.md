# The Last Guardian: unofficial PC port

A native Windows version of The Last Guardian (PS4, USA, CUSA03627). The game's own PS4 program runs directly on your PC's processor; the PS4's system libraries, graphics, sound and input are reimplemented for Windows, with Vulkan for the graphics. It uses your own copy of the game's files. This project contains no game data, no game code and no keys, and it decrypts nothing.

Not affiliated with, endorsed or sponsored by Sony Interactive Entertainment, genDESIGN or any other rights holder. The Last Guardian is a trademark of Sony Interactive Entertainment.

## Highlights

- Native, not emulated. The game's own x86-64 code runs directly on your processor.
- Windowed, borderless or fullscreen, with a render resolution from 1280x720 to 3840x2160.
- 30 fps (the game's own limit) or 60 fps (experimental), at the game's normal speed.
- Mouse camera with raw input; every keyboard and mouse control can be changed.
- DUALSHOCK 4, DualSense and Xbox controllers.

## How it works

The PS4 has an x86-64 processor, like a PC. The port loads the game's program from your files and links it to its own implementation of the PS4's system libraries: memory, threads, files, saves, sound (the game's NGS2 mixer with ATRAC9 and HE-VAG audio), movies, video output and controllers. The PS4 graphics chip's command stream is translated to Vulkan, and its shaders to SPIR-V as the game uses them.

Nothing from the game is in this repository.

## Install

### What you need

- Windows 10 or 11, 64-bit, and a graphics card with a Vulkan 1.3 driver and about 4 GB of video memory.
- A recent processor with high single-core speed: about 25-30 fps on a 6-core CPU from 2019 or later.
- Your own copy of The Last Guardian for PS4 (US release, CUSA03627) at version 01.00 with update 01.03, as decrypted folders. The game folder holds 'eboot.bin', 'sce_module', 'sce_sys' and the 'trico_*.psarc' archives; the update folder holds the same for 01.03, or the update is merged into the game folder.
- About 25 GB of disk space for those files.

### Steps

1. Download `TLG-Port-<version>.zip` from [Releases](../../releases/latest).
2. Right click the zip, choose **Extract All...**, and pick a folder, for example 'C:\Games\TLG-Port'.
3. Start 'tlg-launcher.exe'. If Windows says "Windows protected your PC" (the launcher is not code-signed), click **More info**, then **Run anyway**.
4. On **Game files**, choose the game folder and the update folder. Leave the update folder empty if the update is merged into the game folder.
5. Click **Play**.

## Update

Unzip the new release over the old one or into a new folder. Settings and saves are kept.

## Uninstall

Delete the folder. Settings and saves are in '%APPDATA%\TLG', logs and caches in '%LOCALAPPDATA%\TLG'.

## Controls

Controllers work like the PS4 pad: Xbox and other XInput controllers, DUALSHOCK 4 and DualSense (USB or Bluetooth).

Keyboard and mouse (defaults):

| Action | Keys |
| --- | --- |
| Move | W A S D |
| Walk | Left Shift |
| Camera | Mouse, or I J K L |
| Jump (Triangle) | Space |
| Pick up, grab, hold on (Circle) | E or left click |
| Tackle, throw (Square) | F or right click |
| Crouch, confirm (Cross) | C or Enter |
| R1 / L1 | Q or mouse button 5 / R or mouse button 4 |
| R2 / L2 | X / Z |
| R3 / L3 | V or middle click / B |
| D-pad | Arrow keys |
| Pause menu, back in menus | Escape |
| Options / touchpad | Tab / T |
| Free the mouse | F1 |
| Fullscreen | F11 or Alt+Enter |

In menus the left mouse button selects, the right one goes back and the wheel moves the selection. All controls can be changed in the launcher.

## Settings and saves

- Display mode, resolution, frame-rate limit, graphics card and volume are set in the launcher and apply the next time the game starts.
- Settings and saves are in '%APPDATA%\TLG'. Before each new save, the previous one is copied to 'save-backups' there (the newest 20).
- An empty file named 'portable.txt' next to 'tlg.exe' keeps everything in a 'user' folder beside it.

## Known limits

- 60 fps is experimental: a few of the game's script timers count frames, so some moments run faster at 60 fps.
- No mouse pointer in the game's menus.
- The title screen's music plays quieter than it should.
- Tested on one NVIDIA laptop; AMD and Intel graphics have not been tested yet.

## Building

Run `tools\bootstrap.ps1` once. It downloads the pinned Rust and llvm-mingw toolchains into 'local\tools' and checks their SHA-256. Then `tools\package.ps1` builds `dist\TLG-Port-<version>.zip`.

## Community

For more ports, questions, discussion and help join the [Discord](https://discord.gg/Z7pbJ2unX).
Please don't share game files, keys, decrypted files or download links here or on the Discord.
