# Changelog

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
