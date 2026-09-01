# Zorbio Phase 2 Addendum — Expanded State Vector & Behavioral Nuances

## Context

Phase 1 complete and tested — bots are measurably harder. Phase 2 (data logging +
Python sidecar) is in progress. This addendum must be applied BEFORE the data logging
schema is finalized, because retrofitting the state vector later means reprocessing
all accumulated session data.

---

## Critical: Expand the State Vector

The original handoff state vector (13 floats) is insufficient to capture the tactical
behaviors that make this game interesting. Expand it before finalizing the JSONL schema.

### Original (insufficient):
```
bot_x, bot_y, bot_z, bot_radius,
nearest_human_dx, dy, dz, nearest_human_radius,
nearest_food_dx, dy, dz,
can_eat_nearest_human, human_can_eat_us
```

### Expanded state vector (target: ~35 floats):

```python
state = [
    # Self
    bot_x, bot_y, bot_z,              # normalized to [-1,1] by WORLD_SIZE/2
    bot_radius,                        # normalized by MAX_PLAYER_RADIUS (150)
    bot_vel_dx, bot_vel_dy, bot_vel_dz, # velocity direction (derived: pos_now - pos_prev)
    bot_speed,                         # scalar speed this tick

    # Nearest human (the primary target)
    human_dx, human_dy, human_dz,     # delta to nearest human, normalized
    human_radius,                      # normalized
    human_vel_dx, human_vel_dy, human_vel_dz,  # human's velocity direction
    human_speed,                       # human's scalar speed
    human_intercept_dx, dy, dz,       # projected intercept point (1 second ahead)
                                       # = human_pos + human_vel * (1000/TICK_FAST_INTERVAL)

    # Size relationships
    can_eat_human,                     # binary: bot_radius > human_radius * 1.1
    human_can_eat_us,                  # binary: human_radius > bot_radius * 1.1
    size_ratio,                        # bot_radius / human_radius (continuous)

    # Wall proximity (cornering awareness)
    bot_wall_dist_x,                   # distance to nearest X wall, normalized
    bot_wall_dist_y,                   # distance to nearest Y wall, normalized
    bot_wall_dist_z,                   # distance to nearest Z wall, normalized
    human_wall_dist_x,                 # human's distance to nearest X wall
    human_wall_dist_y,
    human_wall_dist_z,
    human_is_cornered,                 # binary: any human wall dist < 200

    # Boost state
    boost_available,                   # binary (if boost cooldown system exists)
    # Note: boost costs radius, gains speed temporarily
    # Agent must learn: only boost when (radius - boost_cost) still > human_radius * 1.1

    # Nearest food cluster
    food_dx, food_dy, food_dz,        # delta to nearest food, normalized
    food_cluster_value,               # estimated growth available nearby

    # Threat awareness
    nearest_larger_bot_dx, dy, dz,    # nearest bot that can eat us
    nearest_larger_bot_radius,
]
# Total: ~38 floats
```

### Deriving velocity:
Track previous position per entity every tick. Velocity = (pos_now - pos_prev) normalized.
Store `player.sphere.prevPosition` — update it in the tick loop before computing state.

### Deriving intercept point:
```javascript
// 1-second lookahead
let ticksAhead = 1000 / config.TICK_FAST_INTERVAL; // = 20 ticks
let interceptPoint = humanPos.clone().add(
    humanVelocity.clone().multiplyScalar(ticksAhead)
);
// Clamp to world bounds
interceptPoint.clamp(
    new THREE.Vector3(-WORLD_SIZE/2, -WORLD_SIZE/2, -WORLD_SIZE/2),
    new THREE.Vector3( WORLD_SIZE/2,  WORLD_SIZE/2,  WORLD_SIZE/2)
);
```

---

## Behavioral Nuances to Encode

These are the tactics Michael uses that the agent must learn. They inform what state
features matter most.

### 1. Boost mechanics
- Boost: temporary speed multiplier, costs radius (shrinks you)
- Agent must learn the tradeoff: boost only when remaining radius after cost still
  exceeds target * 1.1 (still able to eat them)
- State features needed: current radius, target radius, boost cost estimate
- Reward shaping: successful capture after boost = positive, boost that drops you
  below eating threshold = negative

### 2. Cross-map intercept (the key skill)
- Expert play: don't chase target's current position, predict where they're going
  and get there first
- Requires: human velocity vector + intercept point in state (added above)
- The agent should learn to aim at `interceptPoint` not `humanPosition`
- This is the single most important behavioral upgrade over naive chase bots

### 3. Cornering
- Drive target toward walls to limit their escape vectors
- State features needed: human wall distances + human_is_cornered flag (added above)
- Reward shaping: captures that occur near walls (human_is_cornered=1) should get
  bonus reward — they represent successful cornering strategy

### 4. Evasion when smaller
- Already partially in Phase 1 chase logic, but the RL agent should learn this
  more fluidly from Michael's own evasion patterns
- Key: don't just flee in opposite direction, flee toward food to grow while evading

### 5. Drain beam usage (Drain.js)
- Large player can drain a smaller one at range (up to DRAIN_MAX_DISTANCE = 300)
- Agent should learn: when within drain range but not capture range, use drain
- Adds a medium-range attack vector that pure position-chasing misses
- Add to state: `human_in_drain_range` binary (dist < 300)
- Add to action vector: `use_drain` binary output

---

## Revised Action Vector

```python
action = [
    dx, dy, dz,          # movement direction (normalized, Tanh output)
    speed_multiplier,    # 0.5 to 1.5 (scale Tanh output to this range)
    use_boost,           # binary (threshold Tanh output at 0)
    use_drain,           # binary (threshold Tanh output at 0)
]
# Total: 6 outputs
```

---

## JSONL Session Log Schema

Each line = one tick of game state from Michael's perspective as the human player.
Log Michael's state, not bot state — we're modeling *his* behavior to clone it.

```json
{
  "ts": 1724630400050,
  "tick": 1842,
  "session": "2026-08-25T19:40:00",

  "self": {
    "x": 0.234, "y": -0.112, "z": 0.445,
    "r": 0.623,
    "vx": 0.01, "vy": 0.002, "vz": -0.008,
    "speed": 0.012
  },

  "nearest_target": {
    "dx": -0.3, "dy": 0.1, "dz": 0.05,
    "r": 0.4,
    "vx": 0.005, "vy": -0.001, "vz": 0.002,
    "speed": 0.008,
    "intercept_dx": -0.28, "intercept_dy": 0.098, "intercept_dz": 0.054,
    "can_eat": true,
    "they_can_eat": false,
    "size_ratio": 1.56,
    "wall_dist_x": 0.8, "wall_dist_y": 0.6, "wall_dist_z": 0.3,
    "cornered": false,
    "in_drain_range": true
  },

  "nearest_threat": {
    "dx": 0.5, "dy": 0.2, "dz": -0.1,
    "r": 0.9
  },

  "walls": {
    "self_x": 0.3, "self_y": 0.7, "self_z": 0.5
  },

  "food": {
    "dx": 0.05, "dy": -0.02, "dz": 0.1,
    "cluster_value": 0.3
  },

  "action": {
    "dx": -0.3, "dy": 0.1, "dz": 0.05,
    "speed": 1.0,
    "boosted": false,
    "drained": false
  }
}
```

All spatial values normalized. Action is derived from Michael's actual input/movement
that tick — this is the label for supervised learning.

---

## Where to Hook the Logger in the Codebase

**Best location:** `AppServer.js` in the fast tick loop (every 50ms).

Look for the function that calls `botController.update()` — it's in the same tick
that player positions update. Add logger call there:

```javascript
// In the fast tick handler:
sessionLogger.logTick(model, humanPlayer);
```

Build `SessionLogger.js` in the server directory:
- Constructor takes output path
- `logTick(model, humanPlayer)` computes full state vector, appends JSON line
- Buffers writes (don't fsync every 50ms — buffer 100 ticks, flush periodically)
- New file per session, named by timestamp
- Output: `~/zorbio/training_data/session_YYYY-MM-DDTHH-MM-SS.jsonl`

---

## Python Sidecar Notes

- Use `torch` + `stable-baselines3` or pure PyTorch for flexibility
- Phase 2a: behavioral cloning (supervised) on JSONL data
- Phase 2b: self-play refinement (PPO or similar) after cloning establishes baseline
- Keep model small: 3-4 linear layers, 64-128 hidden units — must run inference in
  <10ms to stay within the 50ms tick budget on CPU
- Socket: Unix domain socket preferred over TCP for local IPC latency
- Fallback: if Python sidecar unavailable, bot falls back to `hunt` movement pattern

---

## Notes on "Playing Against Yourself"

The goal isn't a generically optimal AI. It's an AI that:
1. Mirrors Michael's own approach vectors and tendencies
2. Knows his preferred engagement angles (cross-map intercept)
3. Exploits his patterns — if he always approaches from below, the ghost learns to
   expect that and preemptively cut it off
4. Creates a feedback loop: Michael evolves, retrains, ghost evolves

This is only possible with Michael's own gameplay data as the foundation.
Generic game AI training would produce a different (possibly stronger but less
interesting) opponent.
