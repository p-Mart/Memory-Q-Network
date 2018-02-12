from vizdoom import *
import random
import time

game = DoomGame()

path = 'C:/Users/georg/Documents/GitHub/Vizdoom-MemQN/scenarios/' + \
    'my_way_home.wad'
game.set_doom_scenario_path(path)
game.set_screen_resolution(ScreenResolution.RES_320X240)
game.set_screen_format(ScreenFormat.RGB24)  # [0, 255]^3 for each pixel

# Get position data
game.add_available_game_variable(GameVariable.POSITION_X)
game.add_available_game_variable(GameVariable.POSITION_Y)

# Don't render whats not nessecary
game.set_render_hud(False)
game.set_render_minimal_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(False)
game.set_render_decals(False)
game.set_render_particles(False)
game.set_render_effects_sprites(False)
game.set_render_messages(False)
game.set_render_corpses(False)

# Add buttons
game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.MOVE_FORWARD)
game.add_available_button(Button.TURN_LEFT)
game.add_available_button(Button.TURN_RIGHT)

# Only one action at a time
# actions = [[1, 0, 0],
#            [0, 1, 0],
#            [0, 0, 1]]
actions = [[1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1]]

# Episode and display settings
game.set_episode_timeout(2500)
game.set_episode_start_time(0)
game.set_living_reward(-0.0001)
game.set_mode(Mode.PLAYER)
game.set_window_visible(True)
N_eps = 10

game.init()

# Otherwise everything would be too fast
sleep_time = 1.0 / DEFAULT_TICRATE

for i in range(N_eps):
    print("Episode #%d" % (i + 1))

    game.new_episode()

    while not game.is_episode_finished():
        state = game.get_state()
        n_state = state.number
        vars = state.game_variables
        screen = state.screen_buffer  # 3d np array with shape H x W x C
        print(screen.shape)

        r = game.make_action(random.choice(actions))
        # r = game.make_action(actions[3])  # Pans

        print("State:", state.number)
        print("Game variables:", vars)
        print("Reward:", r)
        print("=====================")

        if sleep_time > 0:
            time.sleep(sleep_time)

# For the sake of completeness
game.close()
