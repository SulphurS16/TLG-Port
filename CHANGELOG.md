# Changelog

## 0.8.1

- Fixed: leaving the game's Options menu (Back or cancel) locked the menus,
  on the title screen and in the pause menu, also after changing a setting
  there (GitHub issues #12 and #13). The PC Settings item in the list never
  finished the list's closing; the port now takes it out of the list as the
  list closes.
- Fixed: changing Shadows to High or Ultra while playing crashed the game a
  few seconds later (GitHub issue #12): the larger shadow map no longer fit
  the game's graphics memory by then. Shadows now change at the next start,
  and High gets the graphics memory it needs on top. Ultra is gone: its map
  is too large for the game's memory system (a setting of Ultra is High).

## 0.8.0

PC features, in the game itself:

- PC Settings in the game's own menus: Options > PC Settings on the title
  screen and in the pause menu, or F10 while playing (it pauses the game).
  The pages look and work like the game's Options pages, with a controller,
  the keyboard or the mouse. Most settings take effect at once; the render
  resolution, the aspect ratio and the intro logos at the next start. The
  launcher keeps its settings too.
- DLSS, XeSS and FSR (PC Settings > Graphics > Anti-Aliasing, live): the
  game's SMAA, or NVIDIA DLSS (DLAA), Intel XeSS or AMD FSR as anti-aliasing
  at the render resolution, fed the game's own motion vectors and depth.
  With Upscaling (Quality 67%, Balanced 58%, Performance 50%, Ultra
  Performance 33%; at the next start) the game renders below the render
  resolution and the chosen upscaler brings it up. Only the ones your PC can
  run are listed: DLSS needs an NVIDIA RTX graphics card; XeSS and FSR run
  on recent graphics cards of any make (FSR 4 on AMD's newer cards, FSR 3.1
  elsewhere). XeSS, FSR and NVIDIA Streamline come in the "upscalers"
  folder; DLSS itself needs your own copy of NVIDIA's nvngx_dlss.dll there
  (NVIDIA's license; see the readme).
- Graphics beyond the PS4's, live: sharpening (0-200%), shadows (low, PS4,
  high, ultra), texture filtering (2x-16x), ambient occlusion (off, half,
  full resolution), screen-space reflections, volumetric fog and light
  shafts on or off. Camera effects on or off: motion blur, depth of field,
  bloom, chromatic aberration, vignette.
- Wide and tall screens: 21:9, 32:9 and 16:10 (Automatic: your screen's own
  shape). A wider picture shows more to the sides and a taller one more
  above and below, without stretching; the menus, prompts and subtitles keep
  the game's 16:9 place in the middle. Render sizes up to 7680 wide (at most
  3840x2160's pixels, so 32:9 reaches 5120x1440). The fixed render
  resolutions now name a height (1080p at 21:9 is 2560x1080).
- Button prompts follow the device you use and switch at once: Xbox buttons
  with an Xbox controller, the PS4's with a DUALSHOCK 4 or DualSense, and
  with keyboard and mouse each prompt's own keys and mouse buttons - your
  bindings, redrawn when you change them (GitHub issue #11). The keys can be
  changed in the game: PC Settings > Controls > Key Bindings.
- A mouse pointer in the game's menus: in the Options lists and the PC
  Settings pages it selects what it points at; in the title and pause menus
  the click and the wheel work on the selection.
- The frame-rate limit (30, 60 or none) changes while you play.
- Menu and HUD size (60-110%), an option to skip the intro logos, mouse
  sensitivity; mouse look follows the game's own camera inversion options.
- At the first start the port picks settings that suit the PC (from the
  graphics card, processor and monitor): display mode, render resolution,
  frame-rate limit, texture filtering, shadows and ambient occlusion.
- Sound: 5.1 and 7.1 output (Automatic follows Windows' speaker set-up),
  headphone surround through Windows Sonic for Headphones or Dolby Atmos for
  Headphones (with spatial sound on for the device in Windows), and a choice
  of sound device. On Windows' default device the game follows when the
  default changes.
- Achievements: a pop-up in the game's style when you unlock a trophy, and
  the list of all trophies with their icons and unlock dates (PC Settings >
  Achievements). The icons come from your game files. The names and
  descriptions are encrypted in them: give the port the PS4 trophy key -
  in the launcher (Game files > Trophy key) or pasted on the Achievements
  page (Trophy Names) - and it shows them; otherwise the trophies are
  numbered. The port does not include the key.
- Saved games: a choice of save folder (PC Settings > Saved Games). An empty
  new folder gets copies of your saves, and the game uses it from then on.
- Fixed a crash in the cutscene after the cave at 3840x2160 (GitHub issue
  #10): the game ran out of graphics memory there. Its graphics memory now
  grows with the render resolution.

## 0.7.2

- Fixed a crash on the way into the last cutscene (GitHub issue #7): the
  game ran out of its graphics memory while loading the ending on top of the
  area before it ("vram allocation failed!" in the log), then crashed. Slower
  graphics cards, integrated ones especially, hit it most. The game now gets
  the larger graphics memory a PS4 Pro gives it (512 MB more) at every
  resolution, not only above 1080p.
- AMD graphics: the port could treat a batch of graphics work as finished
  before the graphics card had run it - mostly in the first seconds, while
  shaders are being prepared - and hand the game results too early. It now
  waits until the work is submitted and done. (Found in a log from an AMD
  system; not tested on AMD hardware yet.)
- Borderless (GitHub issue #4): some graphics drivers (NVIDIA's) took a
  borderless window that covers the screen for exclusive fullscreen and
  switched the monitor to a resolution and refresh rate of their own.
  Borderless and windowed now tell the driver not to; exclusive fullscreen
  stays as it was. The log now notes any display mode change while the game
  runs.
- Fixed a rare crash while the title screen loads, about half a minute
  after the start: the game could look at Trico's skeleton a moment before
  it existed. It happened more often while another program kept the
  computer busy.
- The ending's start runs about 5% faster: while the graphics card is
  behind, the port hands it larger batches of work. Busy scenes such as the
  battle at Sentinel Tower and the ending run 1-2% faster: less bookkeeping
  on the port's graphics thread.

## 0.7.1

- Fixed the lighting "breaking" in some places (GitHub issue #3): after the
  statue bridge, in courtyards and halls, and by the railing after the first
  eye window, the picture could turn bright and washed out with no shadows,
  and come and go with a few steps. The game keeps the sun's shadows in four
  layers; the port mixed them up, so some areas got sunlight where the PS4
  has shade. They now look as on the PS4 (dim halls, torch-lit stairs).
- HDR monitors (GitHub issue #5): with Windows HDR on, the game now sends a
  real HDR picture instead of one that Windows shows dim and grey. The
  launcher's Display page has an HDR output setting: how bright white is
  (200 nits by default, 100-400, or as bright as Windows' SDR content), or
  Off. Also there, experimental: "The game's own HDR" - the PS4 game's HDR
  mode, as a PS4 with HDR shows it; the game's Options > Screen then says
  "HDR is currently enabled" and its brightness slider sets the brightness.

## 0.7.0

- Fixed the lighting being far too bright in bright places (GitHub issue
  #3): skies and sunlit areas washed out to white - the first outdoor
  areas, the mirror chamber, the Falling Bridge courtyard, the Large Tower,
  the Column Corridor, the condor's arena and more. The game measures how
  bright each frame is on the graphics card and reads that back to set its
  exposure (its eye adaptation) and bloom; the port never handed the
  measurement back, so the game always saw a black frame and turned both up
  to the maximum. Exposure now follows the scene as on the PS4: bright
  places are no longer washed out, and some dark places come out brighter.
- The European release (CUSA03745) works too: the game's program is the
  same in every region. The launcher recognizes its folders, also one with
  the update already merged in.
- Languages: the launcher lists the languages your game files carry - the
  US release's English, Spanish, French and Portuguese, or a European
  release's. The game's program also has Japanese, Korean, Chinese and more,
  for copies that carry them.
- Fixed a rare hang when the game starts in a window: now and then it
  stopped before the title screen.

## 0.6.1

- Fixed colors and effects that were drawn wrong or not at all: Trico's
  broken horns glow teal again instead of red, and the game's small glowing
  sprites are back - the glowing butterflies, the halos around luminous
  markings, beams and haze of light, the ending's floating glyphs. The port
  read the small tables some of the game's shaders carry only in part.
- Updating is unzip and play: a portable install unzipped into a new folder
  beside the old one takes over its settings, key bindings and saves at the
  first start, and when the game folders are inside the old version's
  folder the launcher says so and can move them into the new one ("Move
  here" on "Game files"). The installed layout already kept everything in
  %APPDATA%\TLG, which every version uses.

## 0.6.0

- License: the port is now free software under the GNU General Public
  License, version 3 or (at your option) any later version. Releases up to
  0.5.1 were MIT licensed. Third-party components keep their own licenses.
- Faster in busy areas: the port's graphics thread now runs the game's
  compute work itself instead of handing it over to other threads, reads the
  game's command buffers in place instead of copying them first, and keeps
  mid-sized data the game's shaders write in video memory. tlg.exe is also
  built with profile-guided optimization. On a Ryzen 9 5900X with an RTX
  4090, with the 60 fps limit, every area but the ending's start (51-57)
  and the final battle (42-45; was about 38) now runs at 60 (in 0.5.1 many
  ran at 47-59), and the 30 fps default now holds through the ending's
  busiest scene too.
- A message at start-up when the processor lacks instructions the game's own
  code needs (AVX, BMI1, LZCNT, POPCNT, MOVBE: processors from about 2013 on
  have them).
- The 30 fps default is unchanged.

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
