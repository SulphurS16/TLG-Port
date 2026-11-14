# The Last Guardian: unofficial PC port

An unofficial PC port of The Last Guardian (PS4: the US release CUSA03627 or the European release CUSA03745) for Windows. It is not rebuilt from the game's source code: it runs the game's own PS4 program, from your copy of the game's files, on a compatibility layer made for this game. The program's x86-64 code runs directly on your processor, without CPU emulation; the PS4 system libraries it calls are reimplemented for Windows (high-level emulation), and its graphics commands and shaders are translated to Vulkan. This project contains no game data, no game code and no keys, and it decrypts nothing.

<img width="2560" height="1440" alt="tlg_MUydREqKdH" src="https://github.com/user-attachments/assets/0e0e1ae6-dcae-4b30-b10f-71010a3cc114" />


Not affiliated with, endorsed or sponsored by Sony Interactive Entertainment, genDESIGN or any other rights holder. The Last Guardian is a trademark of Sony Interactive Entertainment.

## Highlights

- No CPU emulation: the game's own x86-64 code runs directly on your processor.
- PC Settings inside the game's own menus (Options > PC Settings, or F10 while playing), most of them taking effect at once.
- Windowed, borderless or fullscreen; 16:9, 16:10, 21:9 and 32:9 without stretching (wider shows more to the sides), with the menus and HUD in their place; render resolutions up to 3840x2160 or 5120x1440.
- 30 fps (the game's own limit), 60 fps or uncapped, at the game's normal speed: its timers count time, not frames.
- NVIDIA DLSS, Intel XeSS and AMD FSR as anti-aliasing or upscaling, switched while you play, fed the game's own motion vectors (DLSS with your own copy of NVIDIA's DLSS file).
- Graphics beyond the PS4's: sharpening, shadows, texture filtering, ambient occlusion, reflections, fog and light shafts, and each camera effect on or off.
- Button prompts for the device you use: your keyboard and mouse bindings, an Xbox pad's buttons or the PS4's, switching at once.
- Mouse camera with raw input, a mouse pointer in the menus; every keyboard and mouse control can be changed, in the game too.
- 5.1 and 7.1 sound, headphone surround with Windows Sonic or Dolby Atmos for Headphones, a choice of sound device.
- Achievements with a pop-up in the game's style and a list of all trophies.
- Extras: content cut from the game that its files still hold, such as Trico's drool and cut narrator hints, each off until you turn it on.
- The languages your copy of the game carries: the US release's English, Spanish, French and Portuguese, or a European release's.
- DUALSHOCK 4, DualSense and Xbox controllers.

## How it works

The PS4 has an x86-64 processor, like a PC, so the game's program can run on a PC processor. The port loads that program (eboot.bin and the libraries in its sce_module folder) from your files and links it to its own implementation of the PS4's system libraries: memory, threads, files, saves, sound (the game's NGS2 mixer with ATRAC9 and HE-VAG audio), movies, video output and controllers. While loading, it adjusts a few kinds of instructions: thread-local storage goes through Windows, and AMD's SSE4a instructions are emulated on processors that lack them.

The game drives the PS4's graphics chip directly, without a graphics API like DirectX. The port executes the game's graphics command stream with Vulkan and translates its shaders to SPIR-V as the game uses them, the same work a PS4 emulator does for graphics. A few patches to the game's code, applied as it loads and only to the exact 01.03 program, add features such as the raw mouse camera and the resolution setting.

Nothing from the game is in this repository.

## Install

### What you need

- Windows 10 or 11, 64-bit, and a graphics card with a Vulkan 1.3 driver (NVIDIA, AMD or Intel) and about 4 GB of video memory.
- A recent processor with high single-core speed. Tested on a Ryzen 9 5900X with an RTX 4090: 30 fps everywhere, and with the 60 fps limit 60 in nearly every area.
- Your own copy of The Last Guardian for PS4 (US release CUSA03627 or European release CUSA03745) at version 01.00 with update 01.03, as decrypted folders. The game folder holds 'eboot.bin', 'sce_module', 'sce_sys' and the 'trico_*.psarc' archives; the update folder holds the same for 01.03, or the update is merged into the game folder.
- About 25 GB of disk space for those files.

### Steps

1. Download `TLG-Port-<version>.zip` from [Releases](https://github.com/SulphurS16/TLG-Port/releases/latest).
2. Right click the zip, choose **Extract All...**, and pick a folder, for example 'C:\Games\TLG-Port'.
3. Start 'tlg-launcher.exe'. If Windows says "Windows protected your PC" (the launcher is not code-signed), click **More info**, then **Run anyway**.
4. On **Game files**, choose the game folder and the update folder. Leave the update folder empty if the update is merged into the game folder.
5. Click **Play**.

## Update

Unzip the new release over the old one or into a new folder, and start its 'tlg-launcher.exe'. There is nothing to set up again: your game folders, settings, key bindings and saves are in '%APPDATA%\TLG', which every version uses, and the shader caches in '%LOCALAPPDATA%\TLG' carry on too.

- Portable install (below): unzipped into a new folder beside the old one, the new version takes over the old one's settings, key bindings and saves at its first start (the old folder is left as it was).
- If your game folders are inside the old release's folder, keep that folder, or click **Move here** on **Game files**: they move into the new one's folder.

## Uninstall

Delete the folder. Settings and saves are in '%APPDATA%\TLG', logs and caches in '%LOCALAPPDATA%\TLG'. If you moved your game folders into the port's folder (**Move here** on **Game files**), move them out first, or they are deleted with it.

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
| PC Settings (pauses the game) | F10 |
| Free the mouse | F1 |
| Fullscreen | F11 or Alt+Enter |

In menus the mouse pointer selects what it points at (the Options lists and PC Settings), the left button confirms, the right one goes back and the wheel moves the selection. All controls can be changed in the launcher or in the game (PC Settings > Controls > Key Bindings).

## Settings and saves

- In the game: Options > PC Settings on the title screen and in the pause menu, or F10 while playing - display, graphics, camera effects, sound, controls, interface, saved games, achievements and extras. Most settings take effect at once.
- Shadows (Low, PS4, High) change at the next start, like the render resolution, the aspect ratio and upscaling.
- Anti-aliasing and upscaling (PC Settings > Graphics): the game's SMAA, or DLSS (DLAA; NVIDIA RTX cards), XeSS or FSR (FSR 4 on the AMD cards it supports, FSR 3.1 on others) - the ones your PC can run are listed, and the choice changes at once. Upscaling (Quality 67%, Balanced 58%, Performance 50%, Ultra Performance 33%; at the next start) renders below the render resolution and lets that choice bring it up. XeSS, FSR and NVIDIA Streamline ship in the 'upscalers' folder with their licenses, loaded only after their makers' digital signatures check out. DLSS itself, NVIDIA's 'nvngx_dlss.dll', is under NVIDIA's own license and not included: copy your own into the 'upscalers' folder (games with DLSS have it in their installation folder; NVIDIA's DLSS SDK at github.com/NVIDIA/DLSS has it in 'lib/Windows_x86_64/rel'), and DLSS appears in the list at the next start. NVIDIA, DLSS and RTX are trademarks of NVIDIA Corporation; Intel and XeSS of Intel Corporation; AMD and FSR of Advanced Micro Devices, Inc.
- In the launcher: display mode, resolution, frame-rate limit, graphics card, volume and language, for the next start.
- Trophy names: the trophies' names and descriptions are encrypted in the game files. Give the port the PS4 trophy key (32 hexadecimal digits) to see them: in the launcher (Game files > Trophy key, then Save key), or copy it and choose Trophy Names at the top of the Achievements page in the game. The key is checked against your game's trophies and kept in 'trophy-key.txt' in '%APPDATA%\TLG'; the port does not include it. Without it the trophies are numbered (icons, pop-ups and dates work the same).
- Settings and saves are in '%APPDATA%\TLG'. Before each new save, the previous one is copied to 'save-backups' there (the newest 20).
- An empty file named 'portable.txt' next to 'tlg.exe' keeps everything in a 'user' folder beside it.

## Known limits

- At 60 fps the game's timers and effects keep their normal speed, but the game has not been played through at 60 fps yet.
- Uncapped is not recommended: at 116-119, 156-179 and 237-359 fps the game runs slower than real time.
- DLSS, XeSS and FSR were tested on an NVIDIA RTX card; FSR 4 and XeSS on Intel or AMD graphics have not been tested. Upscaling was checked for its picture; its frame-rate gain on slower graphics cards has not been measured yet. The first switch to DLSS or FSR in a session pauses the picture for a moment (up to about a second).
- Outdoors the grass may sway faster than on the PS4 ([#9](https://github.com/SulphurS16/TLG-Port/issues/9)), likely with the 60 fps limit. Being looked into.
- In the title and pause menus the mouse pointer does not move the selection (click and wheel work there).
- The title screen's music may be quieter than on the PS4; it has not been compared with a PS4 recording.
- Headphone surround has been tested only without spatial sound turned on in Windows.
- Tested with NVIDIA graphics (a laptop and a desktop). AMD and Intel graphics have not been tested yet.
- The European release is tested with an English-only copy. Its other languages are offered when your game files carry them, but have not been tested. Japanese and the other Asian languages are in the game's program but need a copy that carries them.
- Now and then the game closes with an error just after an area starts loading (seen with the 60 fps limit and with none). Start it again and choose Continue: the game goes on from its last checkpoint.

## Building

Run `tools\bootstrap.ps1` once. It downloads the pinned Rust and llvm-mingw toolchains into 'local\tools' and checks their SHA-256. Then `tools\package.ps1` builds `dist\TLG-Port-<version>.zip`.

## License

Copyright (C) 2026 SulphurS16 and The Last Guardian unofficial PC port contributors.

The port is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version (SPDX: GPL-3.0-or-later, with the additional permission below). It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](LICENSE) for the full text. Releases up to 0.5.1 were MIT licensed.

Additional permission under GNU GPL version 3 section 7: If you modify this program, or any covered work, by linking or combining it with NVIDIA DLSS (the NGX runtime nvngx_dlss.dll and the NVIDIA Streamline libraries) or Intel XeSS (libxess.dll), or modified versions of those libraries, containing parts covered by the terms of their own licenses, the licensors of this program grant you additional permission to convey the resulting work. The Corresponding Source for a non-source form of such a combination need not include the source code of those libraries.

The license covers this project's code only. It grants nothing for The Last Guardian or any other game data, keys or system software, which you must supply yourself. Third-party components keep their own licenses, all compatible with the GPL version 3: see [THIRD_PARTY.md](THIRD_PARTY.md) and [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

## Community

For more ports, questions, discussion and help join the [Discord](https://discord.gg/rSr5JhxxtZ).
Please don't share game files, keys, decrypted files or download links here or on the Discord.
