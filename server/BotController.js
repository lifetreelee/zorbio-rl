let config     = require( '../common/config.js' );
let Zorbio     = require( '../common/zorbio.js' );
let Bot        = require( './Bot.js' );
let CurvePaths = require( '../common/CurvePaths' );
let THREE      = require( 'three' );

let BotController = function(model) {
    //  Scope
    let self = this;

    self.model = model;

    self.bots = [];

    self.currentSpawnCycle = 0;

    self.curvePaths = new CurvePaths();

    // Admin-controlled toggles (see AppServer.js admin_* message handlers)
    self.enabled   = true;  // master switch - if false, no bots spawn and existing bots are removed
    self.rlEnabled = true;  // if true, large bots use the trained 'rl' pattern instead of 'hunt' - defaulted on while actively running PPO training; flip back to false (or via the mod menu) once done watching

    /**
     * Turns bot spawning on/off. Disabling immediately removes all current bots.
     * @param {boolean} enabled
     */
    self.setEnabled = function botControllerSetEnabled(enabled) {
        self.enabled = !!enabled;

        if (!self.enabled) {
            while (self.bots.length) self.popBot();
        }
    };

    /**
     * Toggles whether large bots use the RL sidecar-driven movement pattern.
     * Also live-swaps any already-spawned 'hunt'/'rl' bots to match, since
     * bot.move is bound once at spawn time (see Bot.js) - without this, the
     * toggle would only affect bots spawned *after* the change, and since
     * bots rarely die, an admin could flip it and see nothing happen for
     * minutes.
     * @param {boolean} enabled
     */
    self.setRlEnabled = function botControllerSetRlEnabled(enabled) {
        self.rlEnabled = !!enabled;

        let newPattern = self.rlEnabled ? 'rl' : 'hunt';
        self.bots.forEach(function(bot) {
            if (bot.movementPattern === 'hunt' || bot.movementPattern === 'rl') {
                bot.movementPattern = newPattern;
                bot.move = bot.movementPaterns[newPattern];
            }
        });
    };

    self.spawnBot = function botSpawnBot() {
        if (!self.enabled) return null;

        self.setNextSpawnCycle();
        let scale = self.getNextSpawnScale();
        let bot;

        // Spawn the bot
        if (!self.hasChaserBot() && scale < 10) {
            // Always have at least one medium to small size chaser bot
            bot = new Bot(scale, self.model, 'chase');
            console.log('Spawning chaser bot');
        }
        else if (scale > config.RL_BOT_SIZE_THRESHOLD) {
            // Large bots are big enough to be threatening - hunt actively,
            // or use the trained RL policy if an admin has enabled it
            let pattern = self.rlEnabled ? 'rl' : 'hunt';
            bot = new Bot(scale, self.model, pattern, self.curvePaths.getRandomCurve());
            console.log(self.rlEnabled ? 'Spawning RL bot' : 'Spawning hunter bot');
        }
        else {
            // Spawn all other bots with a random curve pattern
            bot = new Bot(scale, self.model, config.BOT_DEFAULT_MOVEMENT, self.curvePaths.getRandomCurve());
        }

        self.bots.push(bot);
        self.model.players.push(bot.player);
        self.model.addActor(bot.player.sphere);

        console.log('Spawned bot: ', bot.name, bot.player.id, bot.scale, self.currentSpawnCycle);

        return bot;
    };

    self.setNextSpawnCycle = function botSetNextSpawnCycle() {
        if (self.currentSpawnCycle >= config.MAX_BOTS) {
            self.currentSpawnCycle = 0;
        }

        // First find out how many big bots are already in the model
        let big_bots = self.getNumBigBots();

        do {
            self.currentSpawnCycle++;
        }
        while (self.currentSpawnCycle < big_bots);
    };

    self.getNumBigBots = function botGetNumBigBots() {
        let num = 0;
        self.model.players.forEach(function eachPlayer(player) {
            if (player.type === Zorbio.PlayerTypes.BOT && player.sphere.scale >= 90) {
                num++;
            }
        });
        return num;
    };

    /**
     * Returns true if there is at least 1 chaser bot false otherwise
     * @returns {boolean}
     */
    self.hasChaserBot = function botHasChaserBot() {
        for (let i = 0; i < self.bots.length; i++) {
            let bot = self.bots[i];

            if (bot.move === bot.movementPaterns.chase) {
                return true;
            }
        }

        return false;
    };

    /**
     * Gets the scale of the next bot to spawn based on internal counter
     * @returns {number}
     */
    self.getNextSpawnScale = function botGetNextSpawnScale() {
        // just in case
        let scale = (this.currentSpawnCycle === 0) ? 1 : this.currentSpawnCycle;

        // https://www.desmos.com/calculator/fmmedr9kzi
        // this curve always makes cycle 1 (the first bot spawned after a
        // reset) the biggest - by design, one "apex" bot plus many small
        // ones - capped so an admin can flatten that out if they don't want
        // a guaranteed giant bot every time
        return Math.min( config.BOT_MAX_SPAWN_SCALE, (30 / (scale - 0.75)) + 3 );
    };

    /**
     * Removes the latest bot spawned
     * @returns {Bot}
     */
    self.popBot = function botPopBot() {
        let bot = self.bots.pop();

        // administrative removal (max-bots change, reset, disable) - not a
        // death, so end its RL episode as truncated if it had one, never
        // terminated, or the reward pipeline would wrongly treat an admin
        // action as the bot getting eaten
        if (bot.rlEpisode) bot.endRLEpisode(false, 'admin_removal');

        // remove from model
        self.model.removePlayer(bot.player.id);

        console.log('Removed top bot: ', bot.id, bot.name, bot.scale);

        return bot;
    };

    /**
     * Removes a bot by ID
     * @param {number} id
     */
    self.removeBot = function botRemoveBot(id) {
        for (let i = 0; i < self.bots.length; i++) {
            let bot = self.bots[i];

            if (bot.id === id) {
                // found the bot with matching ID now delete it from the array
                self.bots.splice(i, 1);

                // real death - end its RL episode as terminated if it had one
                if (bot.rlEpisode) bot.endRLEpisode(true, 'eaten');

                // remove from model may have already been removed but this won't hurt
                self.model.removePlayer(bot.player.id);

                console.log('Removed bot: ', bot.id, bot.name);

                bot = null;

                return;
            }
        }
    };

    self.update = function botUpdate() {
        for (let i = 0; i < self.bots.length; i++) {
            let bot = self.bots[i];

            // bot.move is bound once at spawn time from the pattern picked
            // in spawnBot (see Bot.js) and nothing re-evaluates it as a bot
            // grows - only setRlEnabled's bulk admin toggle ever rebinds it.
            // That meant a bot spawned small (curve pattern, the vast
            // majority per getNextSpawnScale) that later grew past
            // RL_BOT_SIZE_THRESHOLD through feeding would NEVER become an
            // RL/hunt bot - only the ~3 bots per population cycle that spawn
            // already-big ever did. Once those died, there were zero RL bots
            // generating training data until the next full population
            // reset re-seeded new big ones, producing the ~10-min dead
            // zones between resets. Promoting in-place here means growth
            // itself creates new RL bots continuously, the way it always
            // should have.
            if (bot.movementPattern === config.BOT_DEFAULT_MOVEMENT
                && bot.player.sphere.scale > config.RL_BOT_SIZE_THRESHOLD) {
                let newPattern = self.rlEnabled ? 'rl' : 'hunt';
                bot.movementPattern = newPattern;
                bot.move = bot.movementPaterns[newPattern];
                console.log('Promoted bot to', newPattern, 'pattern after growing past threshold:',
                    bot.name, bot.player.id, bot.player.sphere.scale);
            }

            bot.move();
        }
    };

    self.hasBots = function botHasBots() {
        return self.bots.length > 0;
    };


    /**
     * Checks chase bots for to make sure their chase target is still valid
     */
    self.validateChaseTargets = function botControllerCheckChaseTargets() {
        // For any bots that have the chase movement method, set new chase target
        for (let i = 0; i < self.bots.length; i++) {
            let bot = self.bots[i];

            if (bot.move === bot.movementPaterns.chase) {
                if (!bot.chasePlayer ||
                    !self.model.getPlayerById( bot.chasePlayer.id ) ||
                    !(bot.chasePosition instanceof THREE.Vector3)) {
                    console.log('Bot', bot.id, 'chase target invalid, setting new target..');
                    bot.setChaseTarget();  // Set a new random chase target
                }
            }
        }
    };
};

module.exports = BotController;
