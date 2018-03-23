import vizdoom.*;

import java.util.*;
import java.lang.*;

public class Basic {

    public static void main (String[] args) {

        System.out.println("\n\nBASIC EXAMPLE\n");

        // Create DoomGame instance. It will run the game and communicate with you.
        DoomGame game = new DoomGame();

        // Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
        game.setViZDoomPath("../../bin/vizdoom");

        // Sets path to doom2 iwad resource file which contains the actual doom game-> Default is "./doom2.wad".
        game.setDoomGamePath("../../bin/freedoom2.wad");
        //game.setDoomGamePath("../../bin/doom2.wad");   // Not provided with environment due to licences.

        // Sets path to additional resources iwad file which is basically your scenario iwad.
        // If not specified default doom2 maps will be used and it's pretty much useles... unless you want to play doom.
        game.setDoomScenarioPath("../../scenarios/basic.wad");

        // Set map to start (scenario .wad files can contain many maps).
        game.setDoomMap("map01");

        // Sets resolution. Default is 320X240
        game.setScreenResolution(ScreenResolution.RES_640X480);

        // Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
        game.setScreenFormat(ScreenFormat.RGB24);

        // Sets other rendering options
        game.setRenderHud(false);
        game.setRenderCrosshair(false);
        game.setRenderWeapon(true);
        game.setRenderDecals(false);
        game.setRenderParticles(false);
        game.setRenderEffectsSprites(false);
        game.setRenderMessages(false);
        game.setRenderCorpses(false);

        // Adds buttons that will be allowed.
        Button[] availableButtons = new Button [] {Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.ATTACK};
        game.setAvailableButtons(availableButtons);
        // game.addAvailableButton(Button.MOVE_LEFT); // Appends to available buttons.
        // game.addAvailableButton(Button.MOVE_RIGHT);
        // game.addAvailableButton(Button.ATTACK);

        // Returns table of available Buttons.
        // Button[] availableButtons = game.getAvailableButtons();

        // Adds game variables that will be included in state.
        game.addAvailableGameVariable(GameVariable.AMMO2);
        // game.setAvailableGameVariables is also available.

        // Returns table of available GameVariables.
        // GameVariable[] availableGameVariables = game.getAvailableGameVariables();

        // Causes episodes to finish after 200 tics (actions)
        game.setEpisodeTimeout(200);

        // Makes episodes start after 10 tics (~after raising the weapon)
        game.setEpisodeStartTime(10);

        // Makes the window appear (turned on by default)
        game.setWindowVisible(true);

        // Turns on the sound. (turned off by default)
        game.setSoundEnabled(true);

        // Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        game.setMode(Mode.PLAYER);

        // Enables engine output to console.
        //game.setConsoleEnabled(true);

        // Initialize the game. Further configuration won't take any effect from now on.
        game.init();

        // Define some actions. Each list entry corresponds to declared buttons:
        // MOVE_LEFT, MOVE_RIGHT, ATTACK
        // more combinations are naturally possible but only 3 are included for transparency when watching.
        List<double[]> actions = new ArrayList<double[]>();
        actions.add(new double[] {1, 0, 1});
        actions.add(new double[] {0, 1, 1});
        actions.add(new double[] {0, 0, 1});

        Random ran = new Random();

        // Run this many episodes
        int episodes = 10;

        for (int i = 0; i < episodes; ++i) {

            System.out.println("Episode #" + (i + 1));

            // Starts a new episode. It is not needed right after init() but it doesn't cost much and the loop is nicer.
            game.newEpisode();

            while (!game.isEpisodeFinished()) {

                // Get the state
                GameState state = game.getState();

                int n               = state.number;
                double[] vars       = state.gameVariables;
                byte[] screenBuf    = state.screenBuffer;
                byte[] depthBuf     = state.depthBuffer;
                byte[] labelsBuf    = state.labelsBuffer;
                byte[] automapBuf   = state.automapBuffer;
                Label[] labels      = state.labels;

                // Make random action and get reward
                double r = game.makeAction(actions.get(ran.nextInt(3)));

                // You can also get last reward by using this function
                // double r = game.getLastReward();

                System.out.println("State #" + n);
                System.out.println("Game variables: " + Arrays.toString(vars));
                System.out.println("Action reward: " + r);
                System.out.println("=====================");

            }

            System.out.println("Episode finished.");
            System.out.println("Total reward: " + game.getTotalReward());
            System.out.println("************************");

        }

        // It will be done automatically in destructor but after close You can init it again with different settings.
        game.close();
    }
}
