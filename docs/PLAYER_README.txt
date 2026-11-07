The Last Guardian - unofficial PC port 0.8.1 for Windows (64-bit)
==================================================================

An unofficial PC port of The Last Guardian (PS4: US release CUSA03627 or
European release CUSA03745). It is not rebuilt from the game's source code:
it runs the game's own PS4 program from your copy of the game's files. The
program's x86-64 code runs directly on your processor (no CPU emulation);
the PS4 system libraries it calls are reimplemented for Windows, and its
graphics commands and shaders are translated to Vulkan. Nothing of the game
is included.

Not affiliated with, endorsed or sponsored by Sony Interactive Entertainment,
genDESIGN or any other rights holder.


What you need
-------------
- Your own copy of The Last Guardian for PS4, US release (CUSA03627) or
  European release (CUSA03745): the game (version 01.00) and update 01.03,
  as decrypted folders. The game folder holds eboot.bin, sce_module,
  sce_sys and the trico_*.psarc files; the update folder holds the same for
  01.03 (or the update is merged into the game folder).
- Windows 10 or 11 (64-bit), and a graphics card and driver with Vulkan 1.3
  (NVIDIA, AMD or Intel) and about 4 GB of video memory.
- A recent processor with high single-core speed. Tested on a Ryzen 9 5900X
  with an RTX 4090: 30 fps everywhere, and with the 60 fps limit 60 in
  nearly every area, about 51-57 at the ending's start and 42-45 in the
  final battle; about 25-30 fps on a 6-core laptop CPU from 2019.


Getting started
---------------
1. Unpack this zip into a folder, for example C:\Games\TLG-Port.
2. Start tlg-launcher.exe.
3. On "Game files", choose the game folder and the update folder (leave the
   update folder empty if the update is merged into the game folder).
4. Click Play.


Updating
--------
Unpack the new version over the old one or into a new folder and start its
tlg-launcher.exe. There is nothing to set up again: the game folders,
settings, key bindings and saves are in %APPDATA%\TLG, which every version
uses, and the shader caches in %LOCALAPPDATA%\TLG carry on too.
- Portable install: unpacked into a new folder beside the old one, the new
  version takes over the old one's settings, key bindings and saves at its
  first start (the old folder is left as it was).
- If your game folders are inside the old version's folder, keep that
  folder, or click "Move here" on "Game files": they move into the new
  version's folder.


Controls
--------
Controllers: Xbox-compatible controllers, DUALSHOCK 4 and DualSense (USB or
Bluetooth) work as on the PS4.

Keyboard and mouse (defaults; the launcher's Controls page changes them):
  W A S D            move              Left Shift       walk
  mouse              camera            I J K L          camera (keys)
  Space              jump (Triangle)
  E, left button     pick up, grab, hold on (Circle)
  F, right button    tackle, throw (Square)
  C, Enter           crouch, confirm (Cross)
  Q, mouse button 5  R1                R, mouse button 4    L1
  X                  R2                Z                    L2
  V, middle button   R3                B                    L3
  arrow keys         d-pad
  Escape             pause menu, back in menus
  Tab                Options           T                    touchpad
  F10                PC Settings (pauses the game)
  F1                 free the mouse (click into the window to capture it)
  Alt+Enter or F11   fullscreen / window
In menus the mouse pointer selects what it points at (the Options lists and
PC Settings), the left button confirms, the right one goes back and the
wheel moves the selection. The on-screen prompts show the device you use:
the keys you bound, an Xbox pad's buttons or the PS4's.

When the boy dies, floating symbols fill the screen until you press
buttons: press several different ones quickly (face and shoulder buttons,
the d-pad; on the keyboard e.g. C, E, F, Space, Q and R) and the game goes
back to the last checkpoint. That is the PS4 game's own revive screen, not
a hang.


PC Settings
-----------
Options > PC Settings on the title screen and in the pause menu, or F10
while playing, opens the port's settings in the game's own menus:
- Display: display mode, aspect ratio (Automatic: your screen's shape;
  16:9, 16:10, 21:9, 32:9 - a wider picture shows more to the sides, the
  menus and HUD stay in the middle), render resolution (Display, or a height
  such as 1080p: the width follows the aspect ratio), frame-rate limit,
  vertical sync, HDR, scaling filter.
- Graphics: anti-aliasing (the game's SMAA, or DLSS, XeSS or FSR - those
  your PC can run, DLSS with a file you add, see below; at once), upscaling
  (Off, or Quality 67%, Balanced 58%, Performance 50%, Ultra Performance
  33%: the game renders below the render resolution and the anti-aliasing
  choice brings it up; at the next start),
  sharpening, shadows, texture filtering, ambient occlusion, reflections,
  volumetric fog, light shafts. Camera Effects: motion blur, depth of field,
  bloom, chromatic aberration, vignette.
- Sound: speakers (Automatic, stereo, 5.1, 7.1, or headphones - surround
  through Windows Sonic for Headphones or Dolby Atmos for Headphones, which
  need spatial sound turned on for the device in Windows' sound settings),
  sound device, volume.
- Controls: button prompts, mouse sensitivity, key bindings. Interface:
  menu and HUD size, the mouse pointer, skipping the intro logos.
- Saved Games: the save folder (a new empty folder gets copies of your
  saves), and a button that opens it.
- Achievements: the trophies, with their icons and when you unlocked them;
  a pop-up shows each new one. Their names and descriptions are encrypted
  in the game files. To see them, give the port the PS4 trophy key (32
  hexadecimal digits): in the launcher (Game files > Trophy key, then Save
  key), or copy it and choose Trophy Names at the top of the Achievements
  page in the game. Either checks the key against your game's trophies and
  keeps it in trophy-key.txt in %APPDATA%\TLG (portable use: in the "user"
  folder). The port does not include the key. Without it the trophies are
  numbered; everything else works the same.
Most settings take effect at once; the render resolution, the aspect ratio,
upscaling, shadows and the intro logos at the next start (the page says so).

XeSS and FSR: their libraries are in the "upscalers" folder beside tlg.exe,
as their makers ship them (each checked for its maker's digital signature
before it is loaded), with their licenses: Intel XeSS 3.0 and AMD FSR (FSR 4
on the AMD graphics cards it supports, FSR 3.1 on others). Without the
folder the game keeps its own SMAA.

DLSS (NVIDIA RTX graphics cards) needs one more file, which is under
NVIDIA's own license and does not come with the port: NVIDIA's
nvngx_dlss.dll. Copy your own into the "upscalers" folder (beside
sl.dlss.dll), then start the game: "DLSS (DLAA)" appears in the
Anti-Aliasing list. Games with DLSS have the file in their installation
folder; NVIDIA's DLSS SDK has it too (github.com/NVIDIA/DLSS, in
lib\Windows_x86_64\rel). A recent version is best (tested: 310.9); the
game uses it only if it carries NVIDIA's digital signature. NVIDIA
Streamline 2.14, through which the port runs DLSS, is in the folder
already.

The first switch to DLSS or FSR in a session pauses the picture for a
moment (up to about a second) while it sets up. NVIDIA, DLSS and RTX are
trademarks of NVIDIA Corporation; Intel and XeSS of Intel Corporation; AMD
and FSR of Advanced Micro Devices, Inc.


Settings, saves and logs
------------------------
- Settings and key bindings: %APPDATA%\TLG\settings.txt and input.txt.
- Saves: %APPDATA%\TLG\saves. Before each new save the previous one is
  copied to %APPDATA%\TLG\save-backups (the newest 20).
- Logs and caches: %LOCALAPPDATA%\TLG. A crash leaves crash.txt in the
  session's log folder.
- Portable use: an empty file named portable.txt next to tlg.exe keeps
  everything in a "user" folder beside it.
- HDR: with Windows HDR on for the game's monitor the game sends an HDR10
  picture. The launcher's Display page sets how bright white is (200 nits by
  default; 100-400, or as bright as Windows' SDR content) or turns it off.
  "The game's own HDR" (experimental) uses the PS4 game's HDR mode instead:
  set the brightness with the game's own slider (Options > Screen).
- Language: the launcher's Game files page lists the languages your game
  files carry: in the US release English, Spanish (Latin America), French
  (Canada) and Portuguese (Brazil); in a European release its languages
  (English (UK), ...).


Known limits
------------
- At 60 fps the game's timers and effects keep their normal speed, but the
  game has not been played through at 60 fps yet.
- Uncapped is not recommended: at 116-119, 156-179 and 237-359 fps the
  game runs slower than real time.
- DLSS, XeSS and FSR were tested on an NVIDIA RTX card (DLSS, XeSS's and
  FSR 3.1's paths for any graphics card); FSR 4 and XeSS on Intel or AMD
  graphics have not been tested. Upscaling was checked for its picture; how
  much frame rate it gains on a slower graphics card has not been measured
  yet. With upscaling the upscalers start from the game's finished picture
  (after its own effects), which costs some sharpness.
- Outdoors the grass may sway faster than on the PS4 (GitHub issue #9),
  likely with the 60 fps limit: the game moves it a step per frame. Being
  looked into.
- In the title and pause menus the mouse pointer does not move the
  selection (click and wheel work there).
- The title screen's music may be quieter than on the PS4: the port plays
  it as the game's files and mix have it, but it has not been compared with
  a PS4 recording.
- Headphone surround has been tested only without spatial sound turned on
  (it then plays the device's own channels).
- Tested with NVIDIA graphics (a laptop and a desktop). AMD and Intel
  graphics have not been tested yet.
- HDR output has not been tested on an HDR screen yet (its conversion was
  checked numerically).
- The European release is tested with an English-only copy. Its other
  languages are offered when your game files carry them, but have not been
  tested. Japanese and the other Asian languages are in the game's program
  but need a copy that carries them.
- Now and then the game closes with an error just after an area starts
  loading (seen with the 60 fps limit and with none). Start it again and
  choose Continue: the game goes on from its last checkpoint.


License
-------
Copyright (C) 2026 SulphurS16 and The Last Guardian unofficial PC port
contributors.

The port is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version. It is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License, whose full text follows at the end of this file, for more
details.

Additional permission under GNU GPL version 3 section 7: If you modify this
program, or any covered work, by linking or combining it with NVIDIA DLSS
(the NGX runtime nvngx_dlss.dll and the NVIDIA Streamline libraries) or
Intel XeSS (libxess.dll), or modified versions of those libraries,
containing parts covered by the terms of their own licenses, the licensors
of this program grant you additional permission to convey the resulting
work. The Corresponding Source for a non-source form of such a combination
need not include the source code of those libraries.

The source code of this release is at
https://github.com/SulphurS16/TLG-Port (the tag of this version). The
license covers the port's own code only, not the game. The notices of the
third-party components in the two programs follow below.
