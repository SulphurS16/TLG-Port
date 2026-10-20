# Changelog

## 0.5.1

- Faster in the game's heaviest moments: the battle against the armoured
  Tricos at Sentinel Tower and the busiest scenes of the ending. The port's
  graphics work no longer waits for earlier work to be copied back before it
  reuses its upload memory, and mid-sized data the game's shaders read now
  stays in video memory instead of being copied for every use. On a Ryzen 9
  5900X with an RTX 4090 the battle now holds 30 fps at the default limit
  (about 38 with the 60 fps limit) and runs about 5% faster than before
  without a limit, its slow frames shorter; Falling Bridge about 4% faster,
  First Antenna about 3%, the ending's busiest stretch about 7%.
- Processors with 8 or fewer threads (for example 4 cores / 8 threads): the
  port's helper threads poll briefly and its graphics thread gets a core of
  its own. On 4 cores / 8 threads of a Ryzen 9 5900X, Falling Bridge runs
  about 7.5% faster.
- AMD graphics: the port keeps far fewer separate video memory allocations
  (First Antenna about 380 instead of about 2,100; AMD's Windows driver
  allows at most 4,096).
- Less waiting between the port's threads in busy areas (the game's memory
  map is shared for reading; checks of memory the game has not written are
  skipped).
- Mouse camera: mouse movement goes to the game camera the moment it
  arrives instead of waiting for the game's next controller read, so the
  camera takes all of it at its next update, and the input thread runs at
  high priority. The camera turns by the raw mouse counts, with no
  smoothing or acceleration; at 60 fps it updates twice as often as at 30.
- Frame rate: 60 fps is no longer marked experimental, and the launcher
  offers Uncapped (not recommended: at some frame rates the game runs slower
  than real time).
- A smaller download: the launcher, the game program and README.txt, with
  the license and the third-party notices at the end of the README.
- A new icon for the launcher and the game: the letters TLG.
- The 30 fps default is unchanged.

## 0.5.0

- 60 fps at the game's normal speed in every respect: some of the game's
  timers counted frames instead of time (switches and elevators in the
  Soldiers' Den, timing windows on the falling bridges and in First Flight,
  the Sentinel Tower rodeo, Trico's petting, gazes and sneezes, some sounds),
  so they ran up to twice as fast at 60 fps. They now count time, as do the
  animated environment effects and particle emission. At 30 fps nothing
  changes.
- Other graphics cards: a review for AMD and Intel graphics removed the
  port's NVIDIA-specific assumptions (how textures are set up, features it
  required but never used, shader results that differ between drivers, very
  large compute jobs, cards with less video memory). The launcher now checks
  each graphics card the same way the game does. Still untested on AMD and
  Intel hardware.
- Language: English, Spanish (Latin America), French (Canada) or Portuguese
  (Brazil), on the launcher's Game files page.
- When Windows runs out of memory for the game, the game now waits a few
  seconds for Windows to make room and otherwise says what happened, instead
  of closing without a word.
- Smoother First Antenna: when new parts of the area come into play, the
  lighting data the game computes on the graphics card is now prepared
  there too, and copying it back no longer holds up the frame - those
  moments used to stall for a few frames of 50-100 ms.

## 0.4.1

- Faster with the 60 fps limit: the game's own short waits are now timed
  precisely (Windows let them run up to half a millisecond late, which held
  the game's frame back), and the graphics work no longer waits for earlier
  frames to be copied back before it reuses its buffers. On a Ryzen 9 5900X
  with an RTX 4090, Trico's cave and First Antenna run at 55-60 fps with the
  60 fps limit (0.4.0: about 50-55).
- AMD Ryzen processors with two core complexes (CCDs), such as the Ryzen 9
  5900X and 5950X: the port's graphics threads share one CCD and the game's
  threads use the other, so each keeps its data in its own cache.
- Fixed a race in which data copied back from the graphics card could
  overwrite memory the game had just written, most likely while a stage
  loads; a possible cause of a rare crash when First Antenna starts.
- The 30 fps default is unchanged.

## 0.4.0 - first public release

- Runs The Last Guardian (PS4, US, version 01.00 with update 01.03) on Windows from the
  player's own decrypted game folders: the game's own x86-64 program runs directly on the
  processor, the PS4 system libraries are reimplemented and the graphics are translated
  to Vulkan.
- Graphics through Vulkan: windowed, borderless or fullscreen; render resolution from
  1280x720 to 3840x2160; 30 fps or 60 fps (experimental) at the game's normal speed.
- Sound: the game's NGS2 mixer with ATRAC9 and HE-VAG decoding, its equalizers,
  compressors, reverbs and echoes.
- Controls: raw mouse camera, rebindable keyboard and mouse, DUALSHOCK 4, DualSense and
  Xbox controllers.
- Launcher: the game folders, display and control settings, Play.
