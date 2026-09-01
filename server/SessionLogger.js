/**
 * Logs human gameplay to JSONL so a policy can later be behavioral-cloned
 * from it offline (see server/rl/zorbio_agent.py --train).
 *
 * Schema per ZORBIO_PHASE2_ADDENDUM.md: one nested JSON object per human
 * player per fast tick, built from the shared RL feature set in
 * common/RLState.js, with the human's actual movement that tick recorded
 * as the "action" label for supervised learning.
 */
let fs   = require('fs');
let path = require('path');
let os   = require('os');
let config  = require('../common/config.js');
let Zorbio  = require('../common/zorbio.js');
let RLState = require('../common/RLState.js');

let SessionLogger = function() {
    let self = this;

    let dir = path.join(os.homedir(), 'zorbio', 'training_data');
    fs.mkdirSync(dir, { recursive: true });

    self.sessionId = new Date().toISOString();
    let fileSafeSession = self.sessionId.replace(/[:.]/g, '-');
    self.filePath = path.join(dir, 'session_' + fileSafeSession + '.jsonl');
    self.stream = fs.createWriteStream(self.filePath, { flags: 'a' });

    self.tick = 0;

    console.log('SessionLogger: logging human gameplay to', self.filePath);

    self.logTick = function sessionLoggerLogTick(model) {
        self.tick++;

        model.players.forEach(function logPlayer(player) {
            if (player.type !== Zorbio.PlayerTypes.PLAYER) return;

            // build() also gives us the raw ingredients (nearest human/threat/food)
            // via the flat vector - pull the pieces back out for the nested schema
            let state = RLState.build(model, player);
            let vel   = player._rlVelocity || { x: 0, y: 0, z: 0 };
            let speed = player._rlSpeed || 0;
            let targetSurfaceDist = nearestTargetSurfaceDistance(model, player);

            // speed as a multiplier of this player's current max step size (~1.0 at
            // full speed), matching the semantics of the rl action's speed_multiplier
            let maxStep = player.getSpeed() * (config.TICK_FAST_INTERVAL / (1000 / 60));
            let speedMultiplier = maxStep > 0 ? Math.min(2, speed / maxStep) : 0;

            let row = {
                ts       : Date.now(),
                tick     : self.tick,
                session  : self.sessionId,
                player_id: player.id,

                self: {
                    x: state[0], y: state[1], z: state[2], r: state[3],
                    vx: state[4], vy: state[5], vz: state[6], speed: state[7],
                    boost_available: !!state[29],
                },

                nearest_target: {
                    dx: state[8], dy: state[9], dz: state[10], r: state[11],
                    vx: state[12], vy: state[13], vz: state[14], speed: state[15],
                    intercept_dx: state[16], intercept_dy: state[17], intercept_dz: state[18],
                    can_eat: !!state[19],
                    they_can_eat: !!state[20],
                    size_ratio: state[21],
                    wall_dist_x: state[25], wall_dist_y: state[26], wall_dist_z: state[27],
                    cornered: !!state[28],
                    in_drain_range: targetSurfaceDist !== null
                        && targetSurfaceDist < config.DRAIN_MAX_DISTANCE,
                },

                nearest_threat: {
                    dx: state[34], dy: state[35], dz: state[36], r: state[37],
                },

                walls: {
                    self_x: state[22], self_y: state[23], self_z: state[24],
                },

                food: {
                    dx: state[30], dy: state[31], dz: state[32], cluster_value: state[33],
                },

                action: {
                    dx: vel.x, dy: vel.y, dz: vel.z,
                    speed: speedMultiplier,
                    boosted: !!(player.abilities && player.abilities.speed_boost
                        && player.abilities.speed_boost.isActive()),
                    drained: false, // real drain events aren't attributed per-player upstream yet
                },
            };

            self.stream.write(JSON.stringify(row) + '\n');
        });
    };

    // surface-to-surface distance to the nearest other human, or null if none
    function nearestTargetSurfaceDistance(model, player) {
        let pos = player.sphere.position;
        let best = null;

        model.players.forEach(function(other) {
            if (other.id === player.id || other.type !== Zorbio.PlayerTypes.PLAYER) return;
            let d = pos.distanceTo(other.sphere.position) - player.sphere.scale - other.sphere.scale;
            if (best === null || d < best) best = d;
        });

        return best;
    }

    // Marks the moment a logged human died, and by what (another PLAYER vs a
    // BOT), so training can later exclude or down-weight the careless
    // seconds that led up to it. Only logged for human targets - bot deaths
    // aren't training data.
    self.logDeath = function sessionLoggerLogDeath(targetPlayer, attackingPlayer) {
        if (targetPlayer.type !== Zorbio.PlayerTypes.PLAYER) return;

        self.stream.write(JSON.stringify({
            ts       : Date.now(),
            tick     : self.tick,
            session  : self.sessionId,
            event    : 'death',
            player_id: targetPlayer.id,
            killed_by: attackingPlayer.type,
            killer_id: attackingPlayer.id,
        }) + '\n');
    };

    self.close = function sessionLoggerClose() {
        self.stream.end();
    };
};

module.exports = SessionLogger;
