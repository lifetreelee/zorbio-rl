let config = require('../common/config.js');
let Zorbio = require('../common/zorbio.js');
let UTIL = require('../common/util.js');
let RLState = require('../common/RLState.js');
let RLBridge = require('./RLBridge.js');
let _ = require('lodash');
let datasets = require('datasets');
let THREE = require('three');

// Unique per episode, not per bot - a bot's episode restarts (e.g. on the
// step cap) get a fresh id, so the RL sidecar can tell trajectories apart
// even when they belong to the same long-lived bot instance.
let nextRLEpisodeId = 1;

let Bot = function(scale, model, movementPattern, curvePoints) {
    //  Scope
    let self = this;
    self.model = model;

    self.movementPattern = movementPattern || 'curve';  // default movement pattern is curve

    // Array of skin names with duplicates to balance which one will be picked
    // I like line trail type bots because I think they look prettier, default, and neptune have line trails
    let skins = [
        'default',
        'default',
        'default',
        'default',
        'neptune',
        'earth',
        'venus',
        'jupiter',
        'boing',
        'mars',
    ];

    // initialized bot properties
    self.colorCode = UTIL.getRandomIntInclusive(0, config.COLORS.length - 1);
    self.skin_name = skins[UTIL.getRandomIntInclusive(0, skins.length - 1)];
    self.id = Zorbio.IdGenerator.get_next_id();
    self.name = 'AI ' + _.sample(Bot.prototype.names);
    self.scale = scale || UTIL.getRandomIntInclusive(config.INITIAL_PLAYER_RADIUS, config.MAX_PLAYER_RADIUS);

    // curve properties
    self.curvePoints = curvePoints;
    self.nextCurvePoint = 1;


    let position = self.model.getSafeSpawnPosition(10);  // initial spawn position

    if (self.movementPattern === 'curve') {
        if (!Array.isArray(curvePoints) || curvePoints.length === 0) {
            // Generate a default curve points if none is provided
            self.curvePoints = UTIL.randomWanderPath(10, 1.2, 300);
        }

        position = self.curvePoints[0];
    }


    // Create the player model
    self.player = new Zorbio.Player(
        self.id,
        self.name,
        self.colorCode,
        Zorbio.PlayerTypes.BOT,
        position,
        self.scale,
        null,
        self.skin_name
    );

    // Bots historically never paid the boost mass-shrink cost that
    // ServerPlayer.js wires up for real human connections (bots are built
    // from the plain Zorbio.Player class, which has no such listener) - this
    // meant boosting was pure upside for a bot: extra speed, zero size cost.
    // At MAX_PLAYER_RADIUS a maxed-out bot has nowhere to go but to shrink
    // back down the STATIONARY_RADIUS speed curve, so without this it had NO
    // mechanism to ever regain mobility once capped - a one-way ratchet into
    // population-wide stalemate. Mirrors ServerPlayer.js's penalty exactly.
    self.player.abilities.speed_boost.on('update', function botBoostShrinkPenalty() {
        // The shrink amount is an absolute radius delta, but growth reward is
        // normalized by (current scale) - the same absolute cost that's a
        // minor trim on a maxed-out bot is a catastrophic *relative* hit on
        // a small one (a bot near INITIAL_PLAYER_RADIUS can lose 10-20% of
        // its size in one tick). That was blowing up episode rewards to
        // -300+ and destabilizing PPO. Small bots have no strategic need to
        // shed size anyway - only bots big enough to actually be stuck on
        // the STATIONARY_RADIUS speed curve need this trade-off available.
        if (self.player.sphere.scale < config.RL_BOT_SIZE_THRESHOLD) return;

        let active_duration = self.player.abilities.speed_boost.active_duration / 1000;
        let shrink_amount = config.ABILITY_SPEED_BOOST_PENALTY + (Math.pow(active_duration, 2) * 0.005);
        self.player.sphere.growExpected(-shrink_amount);
    });

    // --- RL episode bookkeeping (Phase 1 of the real-RL scope) ---
    // An "episode" is one continuous stretch of the 'rl' movement pattern
    // actually driving this bot. It starts lazily the first time moveRL()
    // runs (covers both a bot spawned straight into 'rl' and one live-swapped
    // from 'hunt' by BotController.setRlEnabled) and ends either when the bot
    // is actually eaten (terminated) or removed/capped for any other reason
    // (truncated). That distinction matters once this feeds a PPO update -
    // truncation should bootstrap the value estimate, termination shouldn't -
    // so it's tracked from day one even before anything consumes it.
    self.rlEpisode = null;

    self.startRLEpisode = function botStartRLEpisode() {
        self.rlEpisode = {
            id              : nextRLEpisodeId++,
            prevScale       : self.player.sphere.scale,
            // playerCaptures is a lifetime counter on the player object
            // (see zorbio.js), never reset between episodes - tracking it
            // here the same way as prevScale lets the per-tick reward see
            // only *this episode's* new kills, not the bot's whole history
            prevCaptures    : self.player.playerCaptures,
            step            : 0,
            cumulativeReward: 0,
            lastReward      : 0,
            // sampled once per episode, not per tick - an opponent's skill
            // shouldn't flicker mid-episode. 0 = live training policy, 1..N =
            // a frozen pool slot (see zorbio_agent.py's OpponentPool - it
            // resolves what a slot currently holds, this side only samples
            // the number). Only policy_id 0 trajectories get logged for
            // training on the Python side - this is what keeps frozen
            // opponents from contaminating the training data.
            policyId: Math.random() < config.RL_POOL_SAMPLE_RATE
                ? UTIL.getRandomIntInclusive(1, config.RL_POOL_SIZE)
                : 0,
        };

        // a stale appliedActionId from a previous episode would carry no
        // meaning under this new episode_id (Python keys pending actions by
        // (episode_id, request_id), so it'd just fail to match anything) -
        // reset explicitly anyway so nothing here depends on that being true
        self.appliedActionId = 0;
    };

    // terminated=true  -> the bot was actually eaten (real death)
    // terminated=false -> episode ended some other way (admin removal, reset,
    //                      or hit the step cap) - not a policy failure
    //
    // step_cap and stuck are handled differently on the wire on purpose: the
    // bot keeps living in both cases, so its regular per-tick moveRL()
    // message already carries this episode's closing done=true (see below)
    // and still needs a real action back to move this tick. Real removals
    // (death, admin) have no next tick to piggyback on, so this sends a
    // standalone closing message whose response only exists to keep the
    // shared connection's FIFO queue aligned for every other bot - the
    // action itself is discarded.
    self.endRLEpisode = function botEndRLEpisode(terminated, reason) {
        if (!self.rlEpisode) return;

        // the death penalty only ever applies here, on a real termination -
        // truncated episodes get exactly the reward they earned, nothing more
        let deathPenalty = terminated ? config.RL_DEATH_PENALTY : 0;
        let totalReward  = self.rlEpisode.cumulativeReward + deathPenalty;
        let avgReward    = totalReward / Math.max(1, self.rlEpisode.step);

        console.log('[RL episode] bot', self.id, 'ep', self.rlEpisode.id, 'ended after', self.rlEpisode.step,
            'steps -', terminated ? 'terminated' : 'truncated', '(' + reason + ')',
            '- reward', totalReward.toFixed(3), '(avg', avgReward.toFixed(4) + '/step)');

        if (reason !== 'step_cap' && reason !== 'stuck') {
            let episodeId   = self.rlEpisode.id;
            let finalReward = self.rlEpisode.lastReward + deathPenalty;
            let state       = RLState.build(self.model, self.player);
            let policyId    = self.rlEpisode.policyId;

            // requestId doesn't matter here - whatever action the sidecar
            // computes for this closing message is discarded (see comment
            // above), so it never needs to be referenced by a future
            // causedByActionId. What DOES matter is causedByActionId itself:
            // it closes out the transition for whichever action was actually
            // applied last, attaching the terminal reward/done/terminated to
            // it instead of leaving it dangling unpaired.
            RLBridge.requestAction(episodeId, finalReward, true, terminated, policyId, state,
                0, self.appliedActionId || 0,
                function rlEpisodeCloseAck() {
                    // intentionally discarded - see comment above
                });
        }

        self.rlEpisode = null;
    };

    // Used only by moveRL()'s boost reward shaping - a lightweight "is there
    // something worth boosting toward" check, not tied to the RL state
    // vector (which only tracks the nearest *human*, not bots, as a named
    // target - a gap that predates this and is out of scope here).
    self.hasNearbyEatableTarget = function botHasNearbyEatableTarget(range) {
        let myPos    = self.player.sphere.position;
        let myRadius = self.player.sphere.scale;

        for (let i = 0; i < self.model.players.length; i++) {
            let other = self.model.players[i];
            if (other.id === self.id || other.type === Zorbio.PlayerTypes.SPECTATOR) continue;

            if (myRadius > other.sphere.scale * 1.1 && myPos.distanceTo(other.sphere.position) < range) {
                return true;
            }
        }

        return false;
    };

    self.movementPaterns = {

        // hold still
        hold: function moveHold() {
            let sphere = self.player.sphere;
            self.player.sphere.pushRecentPosition({
                position: sphere.position,
                radius  : sphere.scale,
                time    : Date.now(),
            });
        },

        // chase a target actor, fleeing instead if it has grown too big to eat
        chase: function moveChase() {
            if (!self.chasePosition ||
                !(self.chasePosition instanceof THREE.Vector3)) {
                return;  // no valid chase target set
            }

            // Decrement chase duration
            self.chaseTime -= config.TICK_FAST_INTERVAL;
            if (self.chaseTime <= 0) {
                self.setChaseTarget();
            }

            let myRadius     = self.player.sphere.scale;
            let targetRadius = self.chasePlayer && self.chasePlayer.sphere
                ? self.chasePlayer.sphere.scale : 0;
            let myPos        = self.player.sphere.position;

            if (targetRadius > myRadius * 0.9) {
                // Target has grown to be a threat - flee in the opposite direction
                let fleeDir = myPos.clone().sub(self.chasePosition).normalize();
                let fleeTarget = myPos.clone().add(fleeDir.multiplyScalar(500));
                self.moveTowardPoint(fleeTarget, 1.2);
            }
            else {
                // We can still eat them - pursue at near full speed
                self.moveTowardPoint(self.chasePosition.clone(), 0.9);
            }
        },

        // actively hunt the nearest player we're big enough to eat, otherwise wander a curve
        hunt: function moveHunt() {
            let myPos    = self.player.sphere.position;
            let myRadius = self.player.sphere.scale;
            let bestTarget = null;
            let bestScore  = -Infinity;

            self.model.players.forEach(function scorePlayer(player) {
                if (player.id === self.id || player.type === Zorbio.PlayerTypes.SPECTATOR) return;

                let theirRadius = player.sphere.scale;
                let dist        = myPos.distanceTo(player.sphere.position);

                // Only consider players we can meaningfully eat
                if (myRadius > theirRadius * 1.1) {
                    // Prefer close, small targets
                    let score = (myRadius - theirRadius) / (dist + 1);
                    if (score > bestScore) {
                        bestScore  = score;
                        bestTarget = player.sphere.position;
                    }
                }
            });

            if (bestTarget) {
                self.moveTowardPoint(bestTarget.clone(), 1.0);
            }
            else {
                // No valid prey nearby - keep growing by wandering the curve
                self.movementPaterns.curve();
            }
        },

        // move to a random points
        randomPoint: function moveRandomPoint() {
            let sphere = self.player.sphere;

            if (!self.moveToPoint) {
                self.moveToPoint = UTIL.randomWorldPosition();
            }

            let dist = sphere.position.distanceTo(self.moveToPoint);

            if (dist < 5) {
                // reached point, generate a new one
                self.moveToPoint = UTIL.randomWorldPosition();
            }

            self.moveTowardPoint(self.moveToPoint.clone());
        },

        // Query the Python RL sidecar for an action; falls back to hunt()
        // while disconnected or before the first action has come back
        rl: function moveRL() {
            if (!self.rlEpisode) self.startRLEpisode();

            self.rlEpisode.step++;

            // reward for the transition into *this* tick, i.e. the consequence
            // of whatever action was taken last tick - normalized growth minus
            // a small constant tax so standing still is never the safe choice,
            // plus a flat bonus per new kill this tick. The kill bonus matters
            // most once a bot hits MAX_PLAYER_RADIUS: scale stops changing
            // entirely at that point, so the growth term alone would give a
            // maxed-out bot zero reward for continuing to hunt successfully -
            // this keeps kills a real, learnable signal at any size.
            //
            // Boost bonus: bots pay no mass cost to boost (unlike human
            // connections - see ServerPlayer.js), so nothing already
            // discourages using it, but useBoost is a rarely-sampled binary
            // action with no reason yet for PPO to discover it's free
            // upside. This nudges exploration toward it specifically when
            // there's something worth chasing, rather than rewarding boost
            // unconditionally (which would just teach "spam boost", not
            // "boost to close distance on prey").
            let scale       = self.player.sphere.scale;
            let captures    = self.player.playerCaptures;
            let killBonus   = config.RL_KILL_REWARD * (captures - self.rlEpisode.prevCaptures);
            let boostBonus  = (self.player.abilities.speed_boost.isActive()
                && self.hasNearbyEatableTarget(config.RL_BOOST_REWARD_RANGE))
                ? config.RL_BOOST_REWARD : 0;
            let reward      = config.RL_GROWTH_REWARD_SCALE * (scale - self.rlEpisode.prevScale) / self.rlEpisode.prevScale
                - config.RL_HUNGER_TAX
                + killBonus
                + boostBonus;

            self.rlEpisode.prevScale    = scale;
            self.rlEpisode.prevCaptures = captures;
            self.rlEpisode.lastReward = reward;
            self.rlEpisode.cumulativeReward += reward;

            // Stuck detection: ending the RL episode's *bookkeeping* alone
            // changes nothing about the bot's actual position - whatever's
            // pinning a bot in place (wall clamp, a corner, or just being
            // near MAX_PLAYER_RADIUS where top speed is naturally tiny -
            // see STATIONARY_RADIUS) carries over unchanged into the next
            // "episode" if only the bookkeeping resets. This forces a real
            // reposition alongside the episode boundary instead, which a
            // step cap alone can never provide.
            //
            // Deliberately NOT gated on wall proximity - a maxed-out bot can
            // be stuck anywhere, not just at the boundary, and a stochastic
            // policy jitters with real per-tick speed even while cornered,
            // so instantaneous speed can't tell "stuck" from "moving" either.
            // Instead this checks NET displacement over a rolling window:
            // every RL_STUCK_TICKS_LIMIT ticks, re-anchor and require at
            // least RL_STUCK_NET_DIST of real progress since the last
            // anchor. Even at max size, committing to one direction covers
            // ~170 world units in that window (see the config comment) - so
            // this only catches genuine stalls, not "big and slow but still
            // trying".
            let pos = self.player.sphere.position;

            if (!self.rlEpisode.stuckAnchor) self.rlEpisode.stuckAnchor = pos.clone();
            self.rlEpisode.stuckTicks = (self.rlEpisode.stuckTicks || 0) + 1;

            let stuck = false;
            if (self.rlEpisode.stuckTicks >= config.RL_STUCK_TICKS_LIMIT) {
                stuck = pos.distanceTo(self.rlEpisode.stuckAnchor) < config.RL_STUCK_NET_DIST;
                // start a fresh window regardless of the outcome
                self.rlEpisode.stuckAnchor = pos.clone();
                self.rlEpisode.stuckTicks = 0;
            }

            // capture before a possible endRLEpisode() call below clears
            // self.rlEpisode, since this tick's message still belongs to the
            // episode that's about to close, not the one that starts after it
            let episodeId = self.rlEpisode.id;
            let policyId  = self.rlEpisode.policyId;
            let requestId = self.rlEpisode.step;
            let capped    = self.rlEpisode.step >= config.RL_EPISODE_MAX_STEPS;
            let done      = stuck || capped;

            if (stuck) {
                // truncated, not terminated - being cornered isn't a policy
                // failure the way getting eaten is
                self.endRLEpisode(false, 'stuck');
                pos.copy(self.model.getSafeSpawnPosition(10));
                RLState.resetVelocity(self.player.id, pos);
                // whatever direction produced the stall was chosen against
                // the OLD position - reusing it against the freshly
                // teleported position makes no sense, fall back to hunt()
                // for this tick instead (same fallback moveRL already uses
                // before any action has ever arrived)
                self.lastRLAction = null;
            }
            else if (capped) {
                // logs + sends no separate wire message (see endRLEpisode) -
                // this tick's normal request below carries done=true instead,
                // since the bot keeps living and still needs an action
                self.endRLEpisode(false, 'step_cap');
            }

            let state = RLState.build(self.model, self.player);

            // causedByActionId tells the sidecar which earlier action
            // actually earned `reward` - see RLBridge.js's v3 protocol note.
            // appliedActionId is only updated below, at the point an action
            // is actually applied for movement, not just received - so it
            // always reflects "whatever action produced the world state this
            // reward was computed from", however many ticks that took.
            RLBridge.requestAction(episodeId, reward, done, false, policyId, state,
                requestId, self.appliedActionId || 0,
                function rlActionReceived(err, action) {
                    if (!err && action) {
                        self.lastRLAction = action;
                        self.lastRLActionId = requestId;
                    }
                });

            if (self.lastRLAction) {
                let [dx, dy, dz, speedMultiplier, useBoost, useDrain] = self.lastRLAction;
                let dir = new THREE.Vector3(dx, dy, dz);

                // snapshot which action this tick's movement is actually
                // applying - see the requestAction call above
                self.appliedActionId = self.lastRLActionId;

                if (useBoost && self.player.abilities.speed_boost.isReady()) {
                    self.player.abilities.speed_boost.activate();
                }

                if (dir.lengthSq() > 1e-6) {
                    let myPos = self.player.sphere.position;
                    let moveSpeedMultiplier = Math.max(0.5, Math.min(1.5, speedMultiplier || 1));

                    if (useDrain) {
                        // Hold near drain range instead of closing for a capture - passive
                        // draining already happens automatically while in range (Drain.js)
                        moveSpeedMultiplier *= 0.5;
                    }

                    let target = myPos.clone().add(dir.normalize().multiplyScalar(50));
                    self.moveTowardPoint(target, moveSpeedMultiplier);
                    return;
                }
            }

            self.movementPaterns.hunt();
        },

        // Move along curve path
        curve: function moveCurve() {
            let sphere = self.player.sphere;
            let nextPoint = self.curvePoints[self.nextCurvePoint];

            // Check the distance
            if (sphere.position.distanceTo(nextPoint) < 5) {
                // reached the point, go to next
                self.nextCurvePoint++;


                // See if we've reached the end of the points and reset
                if (self.nextCurvePoint === self.curvePoints.length) {
                    self.nextCurvePoint = 1;
                }

                nextPoint = self.curvePoints[self.nextCurvePoint];
            }

            self.moveTowardPoint(nextPoint.clone());
        },
    };

    self.moveTowardPoint = function botMoveTowardPoint(point, speedMultiplier) {
        let sphere = self.player.sphere;
        speedMultiplier = speedMultiplier || 1;

        point.sub(sphere.position);
        point.normalize();

        // convert speed multiplier from fps to tick fast
        let speed = self.player.getSpeed() * (config.TICK_FAST_INTERVAL / (1000 / 60));
        speed *= speedMultiplier;

        point.multiplyScalar(speed);

        sphere.position.add(point);

        // bots have no wall-clamp like the human player does (see
        // PlayerController's adjustVelocityWallHit) - without this, a bot
        // fleeing a threat near the edge (chase()'s flee target is 500 units
        // out with no bounds check) can walk straight out of the world and
        // never come back, since nothing ever pulls it back in
        let half = config.WORLD_SIZE / 2;
        sphere.position.x = Math.max(-half, Math.min(half, sphere.position.x));
        sphere.position.y = Math.max(-half, Math.min(half, sphere.position.y));
        sphere.position.z = Math.max(-half, Math.min(half, sphere.position.z));

        sphere.pushRecentPosition({ position: sphere.position, radius: sphere.scale, time: Date.now() });
    };

    self.setChaseTarget = function botChaseTarget(playerId) {
        if (self.movementPattern !== 'chase') return;

        let targetPlayer;

        // Initialize
        self.chasePosition = new THREE.Vector3();

        // Set a timeout to pick a new chase target
        self.chaseTime = UTIL.getRandomIntInclusive(config.BOT_CHASE_TIME_MIN, config.BOT_CHASE_TIME_MAX);

        if (playerId) {
            // look up specific player
            targetPlayer = self.model.getPlayerById(playerId);
        }
        else {
            // pick a random other player that is not this bot or a spectator to chase
            targetPlayer = _.sample(_.filter(self.model.players,
                (p) => p.id !== self.id && p.type !== Zorbio.PlayerTypes.SPECTATOR));
        }

        if (targetPlayer &&
            targetPlayer.sphere &&
            targetPlayer.sphere.position instanceof THREE.Vector3 &&
            targetPlayer.id !== self.id) {
            // Valid target player set to chase
            self.chasePlayer = targetPlayer;
            self.chasePosition = self.chasePlayer.sphere.position;  // for quick lookup

            console.log('Bot', self.id, 'set to chase player id', targetPlayer.id, 'for', self.chaseTime / 1000, 'seconds');
        }
        else {
            console.log('Bot', self.id, 'no chase target available');
        }
    };

    self.setCurvePoints = function botSetCurvePoints(curvePoints) {
        self.curvePoints = curvePoints;
    };

    // Initialize chase target
    self.setChaseTarget();

    self.move = self.movementPaterns[self.movementPattern];
};

Bot.prototype.names = datasets['male-first-names-en'].concat(datasets['female-first-names-en']);

module.exports = Bot;

