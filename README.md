# The Last Guardian: unofficial PC port

An unofficial PC port of The Last Guardian (PS4, USA, CUSA03627) for Windows. It is not rebuilt from the game's source code: it runs the game's own PS4 program, from your copy of the game's files, on a compatibility layer made for this game. The program's x86-64 code runs directly on your processor, without CPU emulation; the PS4 system libraries it calls are reimplemented for Windows (high-level emulation), and its graphics commands and shaders are translated to Vulkan. This project contains no game data, no game code and no keys, and it decrypts nothing.

Not affiliated with, endorsed or sponsored by Sony Interactive Entertainment, genDESIGN or any other rights holder. The Last Guardian is a trademark of Sony Interactive Entertainment.

## Highlights

- No CPU emulation: the game's own x86-64 code runs directly on your processor.
- Windowed, borderless or fullscreen, with a render resolution from 1280x720 to 3840x2160.
- 30 fps (the game's own limit) or 60 fps (experimental), at the game's normal speed.
- Mouse camera with raw input; every keyboard and mouse control can be changed.
- DUALSHOCK 4, DualSense and Xbox controllers.

## How it works

The PS4 has an x86-64 processor, like a PC, so the game's program can run on a PC processor. The port loads that program (eboot.bin and the libraries in its sce_module folder) from your files and links it to its own implementation of the PS4's system libraries: memory, threads, files, saves, sound (the game's NGS2 mixer with ATRAC9 and HE-VAG audio), movies, video output and controllers. While loading, it adjusts a few kinds of instructions: thread-local storage goes through Windows, and AMD's SSE4a instructions are emulated on processors that lack them.

The game drives the PS4's graphics chip directly, without a graphics API like DirectX. The port executes the game's graphics command stream with Vulkan and translates its shaders to SPIR-V as the game uses them, the same work a PS4 emulator does for graphics. A few patches to the game's code, applied as it loads and only to the exact 01.03 program, add features such as the raw mouse camera and the resolution setting.

Nothing from the game is in this repository.

## Install

### What you need

- Windows 10 or 11, 64-bit, and a graphics card with a Vulkan 1.3 driver and about 4 GB of video memory.
- A recent processor with high single-core speed. Tested: a steady 30 fps, and 55-60 fps with the 60 fps limit, on a Ryzen 9 5900X with an RTX 4090; about 25-30 fps on a 6-core laptop CPU from 2019.
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
- Tested with NVIDIA graphics (a laptop and a desktop); AMD and Intel graphics have not been tested yet.

## Building

Run `tools\bootstrap.ps1` once. It downloads the pinned Rust and llvm-mingw toolchains into 'local\tools' and checks their SHA-256. Then `tools\package.ps1` builds `dist\TLG-Port-<version>.zip`.

## License

The port's source code is MIT licensed: see [LICENSE](LICENSE). The license covers this project's code only. It grants nothing for The Last Guardian or any other game data, keys or system software, which you must supply yourself. Third-party components keep their own licenses: see [THIRD_PARTY.md](THIRD_PARTY.md) and [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

## Community

For more ports, questions, discussion and help join the [Discord](https://discord.gg/Z7pbJ2unX).
Please don't share game files, keys, decrypted files or download links here or on the Discord.
