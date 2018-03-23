# Changelog

## Changes in 1.1.5

#### Buttons and actions
- Added `getButton` method.

#### Episode recording and replaying
- Added `isRecordingEpisode` and `isReplayingEpisode` methods.

#### GameVariables
- `KILLCOUNT` counts all kills, including multilayer kills.
- `HITCOUNT`, `HITS_TAKEN`, `DAMAGECOUNT`, `DAMAGE_TAKEN` game variables added.

#### Labels
- Added appending "Dead" prefix to label's name when actor is a corpse.
- **Added bounding box information to Label object in `x`, `y`, `width` and `height` fields.**
- **Added `objectAngle`, `objectPitch`, `objectRoll`, `objectVelocityX/Y/Z` fields to Label object.**


#### Windows specific
- Fixed problem with building on Windows 8.1 and 10.
- Added scripts for downloading freedoom2.wad and assembling Python packages.

#### Rendering
- Fixed minor rendering issue in depth and labels buffer.
- Fixed order of color values in `RGB/BGR` modes of `ScreenFormat`.


## Changes in 1.1.4

#### Automap
- Added `am_scale` CCMD.

#### Scenarios
- Fixed `KILLCOUNT` GameVariable for ChainsawMarine in `defend_the_center` and `deathmatch` scenarios.

#### Python specific
- **Ported Python binding to pybind11 as a replacement for Boost.Python.**
- Fixed problems with `pip install` detecting Python interpreter, includes and libraries from different Python versions.


## Changes in 1.1.3

#### Rendering options
- Added `setRenderScreenFlashes` and `setRenderAllFrames` methods.
- Added `viz_ignore_render_mode` CVAR which disables overriding rendering settings.

#### GameVariables
- **Added `ANGLE`, `PITCH`, `ROLL`, `VELOCITY_X/Y/Z` GameVariables.**

#### Missing config keys
- Added support for `DEATHCOUNT`, `USER31` - `USER60`, `PLAYER_NUMBER`, `PLAYER_COUNT`, `PLAYER1_FRAGCOUNT` - `PLAYER16_FRAGCOUNT`, `POSITION_X/Y/Z` GameVariables in the config file.
- Added support for `ALTATTACK` Button in the config file.

#### Java specific
- Fixed `makeAction`.
- Added missing `POSITION_X/Y/Z` Game Variables.

#### Python specific
- Added manual GIL management for better performance when used with Python threads.

#### Windows specific
- Fixed building for Windows 10.


## Changes in 1.1.2

#### Multiplayer
- Added `isMultiplayerGame` method.
- Added `viz_respawn_delay` CVAR, which allows controlling the delay between respawns in multiplayer game.
- Added `viz_spectator` CVAR which allows connecting to multiplayer game as a spectator.
- **Maximum number of connected players raised to 16, `PLAYER9_FRAGCOUNT` - `PLAYER16_FRAGCOUNT` GameVariables added.**

#### Missing methods
- Added `isRunning`, `isDepthBufferEnabled`, `isLabelsBufferEnabled` and `isAutomapBufferEnabled` missing methods to Python and Lua bindings.


## Changes in 1.1.1

#### GameState
- Added `tic` field.
- `GameVariable.DEATHCOUNT` fixed.

#### Lua specific
- Fixed crash when calling `getState` in a terminal state.

#### Python specific
- Fixed minor memory leak
- Fixed crash when calling `getState` in a terminal state.


## Changes in 1.1.0

#### Buffers

- Depth buffer is now separate buffer in state and `ScreenFormat` values with it was removed - `is/setDepthBufferEnabled` added.
- Added in frame actors labeling feature -`is/setLabelsBufferEnabled` added.
- Added buffer with in game automap - `is/setAutomapBufferEnabled`, `setAutomapMode`, `setAutomapRoate`, `setAutomapRenderTextures`, `AutomapMode` enum added.


#### GameState

- `getState` will now return `nullptr/null/None` if game is in the terminal state.
- `imageBuffer` renamed to `screenBuffer`.
- Added `depthBuffer`, `labelsBuffer` and `automapBuffer` and `labels` fields.


#### Rendering options

- The option to use minimal hud instead of default full hud - `setRenderMinimalHud` added.
- The option to enable/disable effects that use sprites - `setRenderEffectsSprites` added.
- The option to enable/disable in game messages independently of the console output - `setRenderMessages` added.
- The option to enable/disable corpses - `setRenderCorpses` added.


#### Episode recording and replaying

- The option to record and replaying episodes, based on adapted ZDoom's demo mechanism -
recording `filePath` argument added to `newEpisode`, `replayEpisode` added.
- The option to replay demo from other players' perspective.


#### Ticrate

- The option to set number of tics executed per second in ASNYC Modes.
- New `ticrate` optional argument in `doomTicsToMs`, `msToDoomTics`.
- `doomTicsToSec` and `secToDoomTics` added.


#### Paths

- **Paths in config files are now relative to config file.**
- setting **vizdoom_path** and **doom_game_path** is no longer needed - they default to location(installation) of vizdoom.so.


#### Others

- ZDoom engine updated to 2.8.1.
- **Basic support for multiplayer in PLAYER and SPECTATOR Modes.**
- Improved exceptions messages.
- Bugs associated with paths handling fixed.
- Many minor bugs fixed.
- Possibility to change scenario wad during runtime (only first map from WAD file).
- Added `viz_debug` CVAR to control some diagnostic messages.


#### C++ specific

- A lot of overloaded methods turned into a methods with default arguments.
- `getState()` now returns `GameStatePtr (std::shared_ptr<GameState>)` instead of `GameState`.
- Buffers are now copied.
- GameState's buffer has now `BufferPtr (std::shared_ptr<Buffer>)` type - `Buffer (std::vector<uint8_t>)`.
- GameState's gameVariables are now vector of doubles instead of ints.


#### Lua specific

- Lua binding added.
- Support for LuaRocks installation for Linux and MacOS.


#### Java specific

- GameState buffers type changed to byte[].
- Performance improved.
- Java exceptions handling fixed.
- Few functions fixed.


#### Python specific

- Consts added to Python.
- Aliases for `doom_fixed_to_double` - `doom_fixed_to_float` added.
- Support for pip installation for Linux and MacOS.
