/**
 * Builds the RL state vector shared by:
 *   - server/SessionLogger.js (labels for behavioral cloning from human play)
 *   - server/Bot.js 'rl' movement pattern (live inference input)
 * Keeping this in one place means the format training data was captured in
 * always matches what the sidecar sees at inference time.
 *
 * Per ZORBIO_PHASE2_ADDENDUM.md - expanded from the original 13-float vector
 * to add velocity, a 1s intercept prediction, wall/cornering awareness,
 * boost availability, a food cluster estimate, and nearest-threat info.
 *
 * Vector layout (38 floats). NOTE: the "human_*" fields are named for their
 * historical origin (SessionLogger recording a human's nearest opponent) but
 * as of 2026-08-28 track the nearest EATABLE OPPONENT OF ANY TYPE (bot or
 * human) - previously this block was gated to PlayerTypes.PLAYER only, which
 * meant bots had zero positional/velocity/intercept info about other bots
 * and could never learn to hunt them. See [[rl-phase-progress]].
 * [ bot_x, bot_y, bot_z, bot_radius,
 *   bot_vel_dx, bot_vel_dy, bot_vel_dz, bot_speed,
 *   human_dx, human_dy, human_dz, human_radius,
 *   human_vel_dx, human_vel_dy, human_vel_dz, human_speed,
 *   human_intercept_dx, human_intercept_dy, human_intercept_dz,
 *   can_eat_human, human_can_eat_us, size_ratio,
 *   bot_wall_dist_x, bot_wall_dist_y, bot_wall_dist_z,
 *   human_wall_dist_x, human_wall_dist_y, human_wall_dist_z, human_is_cornered,
 *   boost_available,
 *   food_dx, food_dy, food_dz, food_cluster_value,
 *   threat_dx, threat_dy, threat_dz, threat_radius ]
 */
let Zorbio = require('./zorbio.js');
let config = require('./config.js');

let RLState = {};

RLState.STATE_DIM  = 38;
RLState.ACTION_DIM = 6; // [dx, dy, dz, speed_multiplier, use_boost, use_drain]

// 1 second lookahead for intercept prediction
let INTERCEPT_TICKS_AHEAD = 1000 / config.TICK_FAST_INTERVAL;

// radius within which food is considered part of the same "cluster"
let FOOD_CLUSTER_RADIUS   = 250;
let FOOD_CLUSTER_NORM_MAX = 15; // nearby-food count that maps to a cluster_value of 1

// previous tick's position per player id, used to derive velocity
let prevPositions = new Map();

/**
 * Call right after teleporting a player (e.g. Bot.js's stuck-detection
 * reset) so the next updateVelocities() call computes velocity from the
 * NEW position, not a straight line from wherever it was before the
 * teleport. Without this, a stuck bot's one-tick jump across the map reads
 * as a real, huge velocity/speed - bot_speed and human_speed are otherwise
 * bounded to roughly [0, 2] (see PLAYER_GET_SPEED), so a teleport spike
 * stood out sharply as an unbounded outlier in the state vector, the same
 * class of problem sizeRatio had (see RLState.build).
 */
RLState.resetVelocity = function rlStateResetVelocity(playerId, position) {
    prevPositions.set(playerId, { x: position.x, y: position.y, z: position.z });
};

/**
 * Must be called once per tick (before building any player's state) so
 * velocity/speed are available on every player via getVelocity()/getSpeed2().
 */
RLState.updateVelocities = function rlStateUpdateVelocities(model) {
    let seen = new Set();

    model.players.forEach(function updateOne(player) {
        seen.add(player.id);

        let pos  = player.sphere.position;
        let prev = prevPositions.get(player.id);

        if (prev) {
            let dx    = pos.x - prev.x;
            let dy    = pos.y - prev.y;
            let dz    = pos.z - prev.z;
            let speed = Math.sqrt(dx * dx + dy * dy + dz * dz);

            player._rlVelocity = speed > 1e-6
                ? { x: dx / speed, y: dy / speed, z: dz / speed }
                : { x: 0, y: 0, z: 0 };
            player._rlSpeed = speed;
        }
        else {
            player._rlVelocity = { x: 0, y: 0, z: 0 };
            player._rlSpeed = 0;
        }

        prevPositions.set(player.id, { x: pos.x, y: pos.y, z: pos.z });
    });

    // prune players that disconnected/were removed
    prevPositions.forEach(function(_, id) {
        if (!seen.has(id)) prevPositions.delete(id);
    });
};

/**
 * model.food is a flat typed array [x, y, z, x, y, z, ...]
 * @returns {{x:number,y:number,z:number}|null}
 */
RLState.findNearestFood = function rlStateFindNearestFood(model, position) {
    let food = model.food;
    if (!food || food.length === 0) return null;

    let bestDistSq = Infinity;
    let best = null;

    for (let i = 0; i < food.length; i += 3) {
        let dx = food[i] - position.x;
        let dy = food[i + 1] - position.y;
        let dz = food[i + 2] - position.z;
        let distSq = dx * dx + dy * dy + dz * dz;

        if (distSq < bestDistSq) {
            bestDistSq = distSq;
            best = { x: food[i], y: food[i + 1], z: food[i + 2] };
        }
    }

    return best;
};

/**
 * Rough estimate of how much food is clustered near a position, in [0, 1].
 */
RLState.foodClusterValue = function rlStateFoodClusterValue(model, position) {
    let food = model.food;
    if (!food || food.length === 0) return 0;

    let radiusSq = FOOD_CLUSTER_RADIUS * FOOD_CLUSTER_RADIUS;
    let count = 0;

    for (let i = 0; i < food.length; i += 3) {
        let dx = food[i] - position.x;
        let dy = food[i + 1] - position.y;
        let dz = food[i + 2] - position.z;
        if ((dx * dx + dy * dy + dz * dz) <= radiusSq) count++;
    }

    return Math.min(1, count / FOOD_CLUSTER_NORM_MAX);
};

/**
 * Distance from a coordinate to the nearest wall of the world cube, both
 * raw (world units) and normalized to [0, 1] (0 = at the wall, 1 = center).
 */
function wallDistances(position) {
    let half = config.WORLD_SIZE / 2;
    let raw = {
        x: half - Math.abs(position.x),
        y: half - Math.abs(position.y),
        z: half - Math.abs(position.z),
    };
    return {
        raw,
        normalized: { x: raw.x / half, y: raw.y / half, z: raw.z / half },
    };
}

/**
 * @param {Object} model  the game model (has .players and .food)
 * @param {Object} player the player/bot to build the state vector from
 * @returns {number[]} length RLState.STATE_DIM
 */
RLState.build = function rlStateBuild(model, player) {
    let pos    = player.sphere.position;
    let radius = player.sphere.scale;
    let half   = config.WORLD_SIZE / 2;
    let vel    = player._rlVelocity || { x: 0, y: 0, z: 0 };
    let speed  = player._rlSpeed || 0;

    let nearestHuman  = null;
    let nearestHumanDistSq = Infinity;
    let nearestThreat = null;
    let nearestThreatDistSq = Infinity;

    model.players.forEach(function scanOthers(other) {
        if (other.id === player.id || other.type === Zorbio.PlayerTypes.SPECTATOR) return;

        let dx = other.sphere.position.x - pos.x;
        let dy = other.sphere.position.y - pos.y;
        let dz = other.sphere.position.z - pos.z;
        let distSq = dx * dx + dy * dy + dz * dz;

        // nearest eatable opponent, bot OR human - this is the hunting target.
        // (was gated on PlayerTypes.PLAYER only, so bots could never perceive
        // other bots as prey - see [[rl-phase-progress]])
        if (radius > other.sphere.scale * 1.1 && distSq < nearestHumanDistSq) {
            nearestHumanDistSq = distSq;
            nearestHuman = other;
        }

        // any player/bot big enough to eat us is a threat
        if (other.sphere.scale > radius * 1.1 && distSq < nearestThreatDistSq) {
            nearestThreatDistSq = distSq;
            nearestThreat = other;
        }
    });

    let nearestFood       = RLState.findNearestFood(model, pos);
    let foodClusterValue  = RLState.foodClusterValue(model, pos);
    let botWalls          = wallDistances(pos);
    let boostAvailable    = player.abilities && player.abilities.speed_boost
        ? (player.abilities.speed_boost.isReady() ? 1 : 0) : 0;

    let humanDx = 0, humanDy = 0, humanDz = 0, humanRadius = 0;
    let humanVelDx = 0, humanVelDy = 0, humanVelDz = 0, humanSpeed = 0;
    let interceptDx = 0, interceptDy = 0, interceptDz = 0;
    let canEatHuman = 0, humanCanEatUs = 0, sizeRatio = 0;
    let humanWallX = 1, humanWallY = 1, humanWallZ = 1, humanIsCornered = 0;

    if (nearestHuman) {
        let hPos    = nearestHuman.sphere.position;
        let hRadius = nearestHuman.sphere.scale;
        let hVel    = nearestHuman._rlVelocity || { x: 0, y: 0, z: 0 };
        let hSpeed  = nearestHuman._rlSpeed || 0;

        humanDx = (hPos.x - pos.x) / config.WORLD_SIZE;
        humanDy = (hPos.y - pos.y) / config.WORLD_SIZE;
        humanDz = (hPos.z - pos.z) / config.WORLD_SIZE;
        humanRadius = hRadius / config.MAX_PLAYER_RADIUS;

        humanVelDx = hVel.x;
        humanVelDy = hVel.y;
        humanVelDz = hVel.z;
        humanSpeed = hSpeed / config.MAX_PLAYER_SPEED;

        let interceptPoint = {
            x: hPos.x + hVel.x * hSpeed * INTERCEPT_TICKS_AHEAD,
            y: hPos.y + hVel.y * hSpeed * INTERCEPT_TICKS_AHEAD,
            z: hPos.z + hVel.z * hSpeed * INTERCEPT_TICKS_AHEAD,
        };
        interceptPoint.x = Math.max(-half, Math.min(half, interceptPoint.x));
        interceptPoint.y = Math.max(-half, Math.min(half, interceptPoint.y));
        interceptPoint.z = Math.max(-half, Math.min(half, interceptPoint.z));

        interceptDx = (interceptPoint.x - pos.x) / config.WORLD_SIZE;
        interceptDy = (interceptPoint.y - pos.y) / config.WORLD_SIZE;
        interceptDz = (interceptPoint.z - pos.z) / config.WORLD_SIZE;

        canEatHuman   = radius > hRadius * 1.1 ? 1 : 0;
        humanCanEatUs = hRadius > radius * 1.1 ? 1 : 0;
        // Every other feature in this vector is roughly [-1, 1] or [0, 1] -
        // this raw ratio was not, and since the nearest-opponent slot now
        // always tracks a smaller prey (see the 2026-08-28 fix above), it's
        // structurally >= 1.1 whenever populated and can spike into the
        // teens (a maxed bot next to freshly-spawned prey). That one
        // unbounded input dominated the trunk's weighted sums enough to
        // saturate the mean_head's tanh regardless of every other feature -
        // this is what was causing bots to output a near-constant direction
        // vector (the "same corner" behavior) irrespective of context.
        // Bounded to [0, 1] the same way as the other ratio-like features.
        sizeRatio     = hRadius > 0 ? Math.min(radius / hRadius, 5) / 5 : 0;

        let hWalls = wallDistances(hPos);
        humanWallX = hWalls.normalized.x;
        humanWallY = hWalls.normalized.y;
        humanWallZ = hWalls.normalized.z;
        humanIsCornered = Math.min(hWalls.raw.x, hWalls.raw.y, hWalls.raw.z) < 200 ? 1 : 0;
    }

    let threatDx = 0, threatDy = 0, threatDz = 0, threatRadius = 0;
    if (nearestThreat) {
        threatDx = (nearestThreat.sphere.position.x - pos.x) / config.WORLD_SIZE;
        threatDy = (nearestThreat.sphere.position.y - pos.y) / config.WORLD_SIZE;
        threatDz = (nearestThreat.sphere.position.z - pos.z) / config.WORLD_SIZE;
        threatRadius = nearestThreat.sphere.scale / config.MAX_PLAYER_RADIUS;
    }

    return [
        pos.x / half, pos.y / half, pos.z / half, radius / config.MAX_PLAYER_RADIUS,
        vel.x, vel.y, vel.z, speed / config.MAX_PLAYER_SPEED,

        humanDx, humanDy, humanDz, humanRadius,
        humanVelDx, humanVelDy, humanVelDz, humanSpeed,
        interceptDx, interceptDy, interceptDz,

        canEatHuman, humanCanEatUs, sizeRatio,

        botWalls.normalized.x, botWalls.normalized.y, botWalls.normalized.z,
        humanWallX, humanWallY, humanWallZ, humanIsCornered,

        boostAvailable,

        nearestFood ? (nearestFood.x - pos.x) / config.WORLD_SIZE : 0,
        nearestFood ? (nearestFood.y - pos.y) / config.WORLD_SIZE : 0,
        nearestFood ? (nearestFood.z - pos.z) / config.WORLD_SIZE : 0,
        foodClusterValue,

        threatDx, threatDy, threatDz, threatRadius,
    ];
};

module.exports = RLState;
