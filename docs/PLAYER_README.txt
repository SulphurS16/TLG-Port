The Last Guardian - unofficial PC port 0.4.0 for Windows (64-bit)
==================================================================

An unofficial PC port of The Last Guardian (PS4, USA, CUSA03627). It is
not rebuilt from the game's source code: it runs the game's own PS4 program
from your copy of the game's files. The program's x86-64 code runs directly
on your processor (no CPU emulation); the PS4 system libraries it calls are
reimplemented for Windows, and its graphics commands and shaders are
translated to Vulkan. Nothing of the game is included.

Not affiliated with, endorsed or sponsored by Sony Interactive Entertainment,
genDESIGN or any other rights holder.


What you need
-------------
- Your own copy of The Last Guardian for PS4, US release (CUSA03627): the
  game (version 01.00) and update 01.03, as decrypted folders. The game
  folder holds eboot.bin, sce_module, sce_sys and the trico_*.psarc files;
  the update folder holds the same for 01.03 (or the update is merged into
  the game folder).
- Windows 10 or 11 (64-bit), and a graphics card and driver with Vulkan 1.3
  and about 4 GB of video memory.
- A recent processor with high single-core speed: about 25-30 fps on a
  6-core CPU from 2019 or later.


Getting started
---------------
1. Unpack this zip into a folder, for example C:\Games\TLG-Port.
2. Start tlg-launcher.exe.
3. On "Game files", choose the game folder and the update folder (leave the
   update folder empty if the update is merged into the game folder).
4. Click Play.


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


Known limits
------------
- 60 fps is experimental: a few of the game's script timers count frames,
  so some moments run faster at 60 fps.
- The picture keeps the game's 16:9 shape.
- No mouse pointer in the game's menus (keys, wheel and buttons work).
- The title screen's music plays quieter than it should.


License: the port is MIT licensed (LICENSE.txt). The license covers the
port's own code only, not the game. Third-party components keep their own
licenses: THIRD_PARTY.md, THIRD_PARTY_NOTICES.txt and
RUST_STD_NOTICES.html.
