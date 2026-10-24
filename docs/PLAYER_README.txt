The Last Guardian - unofficial PC port 0.7.0 for Windows (64-bit)
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
  F1                 free the mouse (click into the window to capture it)
  Alt+Enter or F11   fullscreen / window
In menus the left mouse button selects, the right one goes back and the
wheel moves the selection.


Settings, saves and logs
------------------------
- Settings and key bindings: %APPDATA%\TLG\settings.txt and input.txt.
- Saves: %APPDATA%\TLG\saves. Before each new save the previous one is
  copied to %APPDATA%\TLG\save-backups (the newest 20).
- Logs and caches: %LOCALAPPDATA%\TLG. A crash leaves crash.txt in the
  session's log folder.
- Portable use: an empty file named portable.txt next to tlg.exe keeps
  everything in a "user" folder beside it.
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
- The picture keeps the game's 16:9 shape.
- No mouse pointer in the game's menus (keys, wheel and buttons work).
- The title screen's music plays quieter than it should.
- Tested with NVIDIA graphics (a laptop and a desktop). AMD and Intel
  graphics have not been tested yet.
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

The source code of this release is at
https://github.com/SulphurS16/TLG-Port (the tag of this version). The
license covers the port's own code only, not the game. The notices of the
third-party components in the two programs follow below.
