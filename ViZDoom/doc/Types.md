# Types

* [Label](#label)
* [GameState](#gamestate)
* [Enums](#enums)
    * [Mode](#mode)
    * [ScreenFormat](#screenformat)
     * [ScreenResolution](#screenresolution)
     * [AutomapMode](#automapmode)
     * [GameVariable](#gamevariable)
     * [Button](#button)
         * [binary buttons](#binary-buttons)
         * [delta buttons](#delta-buttons)


## C++ only

- `Buffer (std::vector<uint8_t>)`
- `BufferPtr (std::shared_ptr<Buffer>)`
- `GameStatePtr (std::shared_ptr<GameState>)`


## Structures

---
### <a name="label"></a> `Label`
(`C++ type / Lua type / Java type / Python type` **name**)

- `unsigned int / number / unsigned int / int` **objectId / object_id**
- `std::string / string / String / str` **objectName / object_name**
- `uint8_t / number / byte / int` **value**
- `unsigned int / number / unsigned int / int` **x**
- `unsigned int / number / unsigned int / int` **y**
- `unsigned int / number / unsigned int / int` **width**
- `unsigned int / number / unsigned int / int` **height**
- `double / number / double / float` **objectPositionX / object_position_x**
- `double / number / double / float` **objectPositionY / object_position_y**
- `double / number / double / float` **objectPositionZ / object_position_z**
- `double / number / double / float` **objectAngle / object_angle**
- `double / number / double / float` **objectPitch / object_pitch**
- `double / number / double / float` **objectRoll / object_roll**
- `double / number / double / float` **objectVelocityX / object_velocity_x**
- `double / number / double / float` **objectVelocityY / object_velocity_y**
- `double / number / double / float` **objectVelocityZ / object_velocity_z**

**objectId / object_id** - unique object instance ID - assigned when object is seen for the first time
(so object with lower id was seen before object with higher).

**objectName / object_name** - ingame object name, many different objects can have the same name (e.g. Medikit, Clip, Zombie).

**value** - value that represents this particular object in **labelsBuffer**.

**x**, **y**, **width**, **height** - describes bounding box of this particular object in **labelsBuffer**.

---
### <a name="gamestate"></a> `GameState`
(`C++ type / Lua type / Java type / Python type` **name**)

- `unsigned int / number / unsigned int / int` **number**
- `unsigned int / number / unsigned int / int` **tic**
- `std::vector<float> / DoubleTensor / double[] / numpy.double[]` **gameVariables / game_variables**
- `BufferPtr / ByteTensor / byte[] / numpy.uint8[]` **screenBuffer / screen_buffer**
- `BufferPtr / ByteTensor / byte[] / numpy.uint8[]` **depthBuffer / depth_buffer**
- `BufferPtr / ByteTensor / byte[] / numpy.uint8[]` **labelsBuffer / labels_buffer**
- `BufferPtr / ByteTensor / byte[] / numpy.uint8[]` **automapBuffer / automap_buffer**
- `std::vector<Label> / table / Label[] / list` **labels**

**number** - number of the state in the episode.
**tic** - ingame time, 1 tic is 1/35 of second in the game world. Added in 1.1.1.


See also:
- [`DoomGame: getState`](DoomGame.md#getState),
- [examples/python/basic.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/basic.py),
- [examples/python/buffers.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/buffers.py).


## <a name="enums"></a> Enums

---
### <a name="mode"></a> `Mode`

Enum type that defines all supported modes.

- **PLAYER** - synchronous player mode
- **SPECTATOR** - synchronous spectator mode
- **ASYNC_PLAYER** - asynchronous player mode
- **ASYNC_SPECTATOR** - asynchronous spectator mode

In **PLAYER** and **ASYNC_PLAYER** modes, the agent controls ingame character.

In **SPECTATOR** and **ASYNC_SPECTATOR** modes, ingame character should be controlled by the human and the agent gets information about the human action.

In **PLAYER** and **SPECTATOR** modes, the game waits for agent action or permission to continue.

In **ASYNC** modes the game progress with constant speed (default 35 tics per second, this can be set) without waiting for the agent actions.

All modes can be used in singleplayer and multiplayer.

See also:
- [`DoomGame: getMode`](DoomGame.md#getMode),
- [`DoomGame: setMode`](DoomGame.md#setMode),
- [`DoomGame: getTicrate`](DoomGame.md#getTicrate),
- [`DoomGame: setTicrate`](DoomGame.md#setTicrate).


---
### <a name="screenformat"></a> `ScreenFormat`

Enum type that defines all supported **screenBuffer** and **automapBuffer** formats.

- **CRCGCB**    - 3 channels of 8-bit values in RGB order
- **RGB24**     - channel of RGB values stored in 24 bits, where R value is stored in the oldest 8 bits
- **RGBA32**    - channel of RGBA values stored in 32 bits, where R value is stored in the oldest 8 bits
- **ARGB32**    - channel of ARGB values stored in 32 bits, where A value is stored in the oldest 8 bits
- **CBCGCR**    - 3 channels of 8-bit values in BGR order
- **BGR24**     - channel of BGR values stored in 24 bits, where B value is stored in the oldest 8 bits
- **BGRA32**    - channel of BGRA values stored in 32 bits, where B value is stored in the oldest 8 bits
- **ABGR32**    - channel of ABGR values stored in 32 bits, where A value is stored in the oldest 8 bits
- **GRAY8**     - 8-bit gray channel
- **DOOM_256_COLORS8** - 8-bit channel with Doom palette values


In **CRCGCB** and **CBCGCR** format **screenBuffer** and **automapBuffer** store all red 8-bit values then all green values and then all blue values, each channel is considered separately. As matrices they have [3, y, x] shape.

In **RGB24** and **BGR24** format **screenBuffer** and **automapBuffer** store 24 bit RGB triples. As matrices they have [y, x, 3] shape.

In **RGBA32**, **ARGB32**, **BGRA32** and **ABGR32** format **screenBuffer** and **automapBuffer** store 32 bit sets of RBG + alpha values. As matrices they have [y, x, 4] shape.

In **GRAY8** and **DOOM_256_COLORS8** format **screenBuffer** and **automapBuffer** store single 8 bit values. As matrices they have [y, x] shape.

**depthBuffer** and **lablesBuffer** always store single 8-bit values, so they always have [y, x] shape.

See also:
- [`DoomGame: getScreenFormat`](DoomGame.md#getScreenFormat),
- [`DoomGame: setScreenFormat`](DoomGame.md#setScreenFormat),
- [examples/python/buffers.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/buffers.py).


---
### <a name="screenresolution"></a> `ScreenResolution`

Enum type that defines all supported resolutions - shapes of **screenBuffer**, **depthBuffer**, **labelsBuffer** and **automapBuffer** in **State**.

- **RES_160X120** (4:3)
- **RES_200X125** (16:10)
- **RES_200X150** (4:3)
- **RES_256X144** (16:9)
- **RES_256X160** (16:10)
- **RES_256X192** (4:3)
- **RES_320X180** (16:9)
- **RES_320X200** (16:10)
- **RES_320X240** (4:3)
- **RES_320X256** (5:4)
- **RES_400X225** (16:9)
- **RES_400X250** (16:10)
- **RES_400X300** (4:3)
- **RES_512X288** (16:9)
- **RES_512X320** (16:10)
- **RES_512X384** (4:3)
- **RES_640X360** (16:9)
- **RES_640X400** (16:10)
- **RES_640X480** (4:3)
- **RES_800X450** (16:9)
- **RES_800X500** (16:10)
- **RES_800X600** (4:3)
- **RES_1024X576** (16:9)
- **RES_1024X640** (16:10)
- **RES_1024X768** (4:3)
- **RES_1280X720** (16:9)
- **RES_1280X800** (16:10)
- **RES_1280X960** (4:3)
- **RES_1280X1024** (5:4)
- **RES_1400X787** (16:9)
- **RES_1400X875** (16:10)
- **RES_1400X1050** (4:3)
- **RES_1600X900** (16:9)
- **RES_1600X1000** (16:10)
- **RES_1600X1200** (4:3)
- **RES_1920X1080** (16:9)

See also:
- [`DoomGame: setScreenResolution`](DoomGame.md#setScreenResolution),
- [`DoomGame: getScreenWidth`](DoomGame.md#getScreenWidth),
- [`DoomGame: getScreenHeight`](DoomGame.md#getScreenHeight).


---
### <a name="automapmode"></a> `AutomapMode`

Enum type that defines all **automapBuffer** modes.

- **NORMAL**    - Only level architecture the player has seen is shown.
- **WHOLE**     - All architecture is shown, regardless of whether or not the player has seen it.
- **OBJECTS**   - In addition to the previous, shows all things in the map as arrows pointing in the direction they are facing.
- **OBJECTS_WITH_SIZE** - In addition to the previous, all things are wrapped in a box showing their size.

See also:
- [`DoomGame: setAutomapMode`](DoomGame.md#setAutomapMode),
- [examples/python/buffers.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/buffers.py).


---
### <a name="gamevariable"></a> `GameVariable`

Enum type that defines all variables that can be obtained from the game.


    FRAGCOUNT is mostly for multiplayer, e.g. against other players or bots. Does not update from killing NPC(?)
    KILLCOUNT reads "level.killed_monsters", so I would guess it updates on killing monsters. However as you pointed out, this does not seem to be the case.
    SECRETCOUNT seems to count the number of "secret objects" collected (level.found_secrets), possibly for some "locate and collect" task for collecting certain type of pickups.


#### Defined variables
- **KILLCOUNT**     - Counts the number of monsters killed during the current episode. ~Killing other players/bots do not count towards this.~ From 1.1.5 killing other players/bots counts towards this.
- **ITEMCOUNT**     - Counts the number of picked up items during the current episode.
- **SECRETCOUNT**   - Counts the number of secret location/objects discovered during the current episode.
- **FRAGCOUNT**     - Counts the number of players/bots killed, minus the number of committed suicides. Useful only in multiplayer mode.
- **DEATHCOUNT**    - Counts the number of players deaths during the current episode. Useful only in multiplayer mode.
- **HITCOUNT**      - Counts number of hit monsters/players/bots during the current episode.
- **HITS_TAKEN**    - Counts number of hits taken by the player during the current episode.
- **DAMAGECOUNT**   - Counts number of damage dealt to monsters/players/bots during the current episode.
- **DAMAGE_TAKEN**  - Counts number of damage taken by the player during the current episode.
- **HEALTH**        - Can be higher then 100!
- **ARMOR**         - Can be higher then 100!
- **DEAD**          - True if the player is dead.
- **ON_GROUND**     - True if the player is on the ground (not in the air).
- **ATTACK_READY**  - True if the attack can be performed.
- **ALTATTACK_READY**       - True if the altattack can be performed.
- **SELECTED_WEAPON**       - Selected weapon's number.
- **SELECTED_WEAPON_AMMO**  - Ammo for selected weapon.
- **AMMO0** - **AMMO9**     - Number of ammo for weapon in N slot.
- **WEAPON0** - **WEAPON9** - Number of weapons in N slot.
- **POSITION_X**            - Position of the player, not available if `viz_nocheat` is enabled.
- **POSITION_Y**
- **POSITION_Z**
- **ANGLE**                 - Orientation of the player, not available if `viz_nocheat` is enabled.
- **PITCH**                 
- **ROLL**
- **VELOCITY_X**            - Velocity of the player, not available if `viz_nocheat` is enabled.
- **VELOCITY_Y**
- **VELOCITY_Z**
- **PLAYER_NUMBER**         - Player's number in multiplayer game.
- **PLAYER_COUNT**          - Number of players in multiplayer game.
- **PLAYER1_FRAGCOUNT** - **PLAYER16_FRAGCOUNT** - Number of N player's frags


#### User (ACS) variables  
- **USER1** - **USER60**

ACS global int variables can be accessed as USER GameVariables.
global int 0 is reserved for reward and is always threaded as Doom's fixed point numeral.
Other from 1 to 60 (global int 1-60) can be accessed as USER1 - USER60 GameVariables.

See also:
- [ZDoom Wiki: ACS](http://zdoom.org/wiki/ACS),
- [`DoomGame: getAvailableGameVariables`](DoomGame.md#getAvailableGameVariables),
- [`DoomGame: setAvailableGameVariables`](DoomGame.md#setAvailableGameVariables),
- [`DoomGame: addAvailableGameVariable`](DoomGame.md#addAvailableGameVariable),
- [`DoomGame: getGameVariable`](DoomGame.md#getGameVariable),
- [`Utilities: doomFixedToDouble`](Utilities.md#doomFixedToDouble),
- [examples/python/basic.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/basic.py),
- [examples/python/shaping.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/shaping.py).


---
### <a name="button"></a> `Button`

Enum type that defines all buttons that can be "pressed" by the agent.

#### <a name="binary-buttons"></a> Binary buttons

Binary buttons have only 2 states "not pressed" if value 0 and "pressed" if value other then 0.

- **ATTACK**
- **USE**
- **JUMP**
- **CROUCH**
- **TURN180**
- **ALTATTACK**
- **RELOAD**
- **ZOOM**
- **SPEED**
- **STRAFE**
- **MOVE_RIGHT**
- **MOVE_LEFT**
- **MOVE_BACKWARD**
- **MOVE_FORWARD**
- **TURN_RIGHT**
- **TURN_LEFT**
- **LOOK_UP**
- **LOOK_DOWN**
- **MOVE_UP**
- **MOVE_DOWN**
- **LAND**
- **SELECT_WEAPON1**
- **SELECT_WEAPON2**
- **SELECT_WEAPON3**
- **SELECT_WEAPON4**
- **SELECT_WEAPON5**
- **SELECT_WEAPON6**
- **SELECT_WEAPON7**
- **SELECT_WEAPON8**
- **SELECT_WEAPON9**
- **SELECT_WEAPON0**
- **SELECT_NEXT_WEAPON**
- **SELECT_PREV_WEAPON**
- **DROP_SELECTED_WEAPON**
- **ACTIVATE_SELECTED_ITEM**
- **SELECT_NEXT_ITEM**
- **SELECT_PREV_ITEM**
- **DROP_SELECTED_ITEM**


#### <a name="delta-buttons"></a> Delta buttons

Buttons whose value defines the speed of movement.
A positive value indicates movement in the first specified direction and a negative value in the second direction.
For example: value 10 for MOVE_LEFT_RIGHT_DELTA means slow movement to the right and -100 means fast movement to the left.

- **LOOK_UP_DOWN_DELTA**
- **TURN_LEFT_RIGHT_DELTA**
- **MOVE_FORWARD_BACKWARD_DELTA**
- **MOVE_LEFT_RIGHT_DELTA**
- **MOVE_UP_DOWN_DELTA**

In case of **TURN_LEFT_RIGHT_DELTA** and **LOOK_UP_DOWN_DELTA** values correspond to degrees.
In case of **MOVE_FORWARD_BACKWARD_DELTA**, **MOVE_LEFT_RIGHT_DELTA**, **MOVE_UP_DOWN_DELTA** values correspond to Doom Map unit (see Doom Wiki if you want to know how it translates into real life units).

See also:
- [Doom Wiki: Map unit](https://doomwiki.org/wiki/Map_unit),
- [`DoomGame: getAvailableButtons`](DoomGame.md#getAvailableButtons),
- [`DoomGame: setAvailableButtons`](DoomGame.md#setAvailableButtons),
- [`DoomGame: addAvailableButton`](DoomGame.md#addAvailableButton),
- [`DoomGame: setButtonMaxValue`](DoomGame.md#setButtonMaxValue),
- [`DoomGame: getButtonMaxValue`](DoomGame.md#getButtonMaxValue),
- [examples/python/basic.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/basic.py),
- [examples/python/delta_buttons.py](https://github.com/mwydmuch/ViZDoom/tree/master/examples/python/delta_buttons.py),
- [GitHub issue: Angle changes by executing certain commands](https://github.com/mwydmuch/ViZDoom/issues/182).
