# Zorbio Bot AI Upgrade — Claude Code Handoff

## Context

Michael has a self-hosted instance of **Zorbio** (MIT-licensed 3D multiplayer WebGL game,
originally zorb.io) running locally on Ubuntu Linux. The game is running directly via
`npm run start-dev` (not Docker, though Docker is available). He dominates the existing
bots trivially — 2600 points, ate every bot. The bots need to be smarter, and longer
term we want an RL training loop that learns from his gameplay.

**Repo location on his machine:** `~/zorbio/zorbio/`
**GitHub source (MIT):** https://github.com/ScriptaGames/zorbio
**Game runs at:** `http://localhost:8080` (dev mode, HTTP, no TLS)

---

## System

- **OS:** Ubuntu Linux (Ubuntu Pro)
- **Machine:** Dell Optiplex (lee-lab-1991), likely 7020 with reasonable RAM
- **Node:** v22.23.1
- **npm:** 10.9.8
- **Docker:** Available but not required — he's running npm directly
- **Python:** Available (assume standard Ubuntu install, verify version)
- **GPU:** Dedicated card present (used for WebGL client rendering only)

---

## Repo Structure (relevant files)

```
~/zorbio/zorbio/
├── common/
│   ├── config.js          # All tunable game constants — BOT settings here
│   ├── CurvePaths.js      # Geometric path generators (trefoil, cinquefoil, granny, randomWander)
│   ├── CurveExtras.js     # THREE.js curve math
│   ├── util.js            # Helpers including randomWanderPath()
│   └── zorbio.js          # Core game model (Player, Sphere, IdGenerator etc)
├── server/
│   ├── server.js          # Entry point
│   ├── AppServer.js       # Game loop, tick logic, spawn/remove bots
│   ├── BotController.js   # Manages bot lifecycle, spawn cycles, update loop
│   ├── Bot.js             # Individual bot brain — movement patterns live here
│   ├── ServerPlayer.js    # Server-side player representation
│   └── Drain.js           # Drain beam mechanic
└── client/                # WebGL frontend (THREE.js) — don't touch for this task
```

---

## Current Bot Architecture (what exists)

### Bot.js — movement patterns:
- **`curve`** — follows pre-computed geometric path (trefoil/cinquefoil/granny knot),
  zero world awareness, ignores all players and food
- **`chase`** — moves toward a target player's position at 0.5x speed, no size
  awareness (will chase players larger than itself — suicidal), switches target
  every 20-45 seconds randomly
- **`randomPoint`** — moves to random world positions, effectively useless
- **`hold`** — stationary

### BotController.js — spawn logic:
- Maintains up to `config.MAX_BOTS` (currently 20) bots
- Always ensures at least 1 chase bot exists
- Spawn scale follows a curve: `(30 / (cycle - 0.75)) + 3` — early bots are large,
  later bots smaller
- `update()` called every tick, calls `bot.move()` for each bot

### Key config values (common/config.js):
```javascript
config.MAX_BOTS           = 20;
config.MAX_BOT_RADIUS     = 100;    // bots cap at 100, player max is 150
config.BOT_CHASE_TIME_MIN = 20000;  // ms
config.BOT_CHASE_TIME_MAX = 45000;  // ms
config.TICK_FAST_INTERVAL = 50;     // ms — bot move() called this often
config.TICK_SLOW_INTERVAL = 200;    // ms — general server updates
config.WORLD_SIZE         = 2000;   // cube side length
config.INITIAL_PLAYER_RADIUS = 5;
config.MAX_PLAYER_RADIUS     = 150;
config.MAX_PLAYER_SPEED      = 2;
// Speed formula: s - ((r * s) / STATIONARY_RADIUS)
// i.e. larger = slower
```

### State available per tick in Bot.js:
```javascript
self.player.sphere.position    // THREE.Vector3 — bot position
self.player.sphere.scale       // number — bot radius
self.model.players             // array of all players (bots + humans)
self.chasePlayer               // currently targeted player
// Each player has: player.sphere.position, player.sphere.scale, player.type
// Zorbio.PlayerTypes.BOT vs Zorbio.PlayerTypes.HUMAN
```

---

## Phase 1 Task — Smart Bot Behavior (no ML, pure logic)

**Goal:** Make bots actually threatening to a skilled human player.

### 1a. Config changes (common/config.js):
```javascript
config.MAX_BOT_RADIUS     = 140;   // closer to player max of 150
config.BOT_CHASE_TIME_MIN = 45000;
config.BOT_CHASE_TIME_MAX = 90000;
```

### 1b. Rewrite chase movement in Bot.js:

Replace the `chase` movement pattern with size-aware logic:

```javascript
chase: function moveChase() {
    if (!self.chasePosition || !(self.chasePosition instanceof THREE.Vector3)) return;

    self.chaseTime -= config.TICK_FAST_INTERVAL;
    if (self.chaseTime <= 0) self.setChaseTarget();

    let myRadius     = self.player.sphere.scale;
    let targetRadius = self.chasePlayer && self.chasePlayer.sphere
        ? self.chasePlayer.sphere.scale : 0;
    let myPos        = self.player.sphere.position;

    if (targetRadius > myRadius * 0.9) {
        // Target is same size or bigger — flee
        let fleeDir = myPos.clone().sub(self.chasePosition).normalize();
        let fleeTarget = myPos.clone().add(fleeDir.multiplyScalar(500));
        self.moveTowardPoint(fleeTarget, 1.2);
    } else {
        // We can eat them — pursue at near full speed
        self.moveTowardPoint(self.chasePosition.clone(), 0.9);
    }
},
```

### 1c. Add a new `hunt` movement pattern to Bot.js:

This is the key upgrade — a bot that actively hunts food AND smaller players:

```javascript
hunt: function moveHunt() {
    let myPos    = self.player.sphere.position;
    let myRadius = self.player.sphere.scale;
    let bestTarget = null;
    let bestScore  = -Infinity;

    // Score all other players
    self.model.players.forEach(function(player) {
        if (player.id === self.id) return;
        let theirRadius = player.sphere.scale;
        let dist = myPos.distanceTo(player.sphere.position);

        // Only consider players we can eat (we must be meaningfully bigger)
        if (myRadius > theirRadius * 1.1) {
            // Score: prefer close + small targets
            let score = (myRadius - theirRadius) / (dist + 1);
            if (score > bestScore) {
                bestScore  = score;
                bestTarget = player.sphere.position;
            }
        }
    });

    if (bestTarget) {
        self.moveTowardPoint(bestTarget.clone(), 1.0);
    } else {
        // No valid prey — revert to curve behavior to keep growing
        self.movementPaterns.curve();
    }
},
```

### 1d. Update BotController.js spawnBot() to use hunt:

Replace the spawn logic so larger bots use `hunt` instead of `curve`:

```javascript
self.spawnBot = function botSpawnBot() {
    self.setNextSpawnCycle();
    let scale = self.getNextSpawnScale();
    let bot;

    if (!self.hasChaserBot() && scale < 10) {
        bot = new Bot(scale, self.model, 'chase');
        console.log('Spawning chaser bot');
    } else if (scale > 50) {
        // Large bots actively hunt
        bot = new Bot(scale, self.model, 'hunt', self.curvePaths.getRandomCurve());
        console.log('Spawning hunter bot');
    } else {
        bot = new Bot(scale, self.model, config.BOT_DEFAULT_MOVEMENT, self.curvePaths.getRandomCurve());
    }

    self.bots.push(bot);
    self.model.players.push(bot.player);
    self.model.addActor(bot.player.sphere);
    console.log('Spawned bot: ', bot.name, bot.player.id, bot.scale, self.currentSpawnCycle);
    return bot;
};
```

**After Phase 1:** Restart with `npm run start-dev` and test. Bots should now flee
when outmatched and hunt when dominant.

---

## Phase 2 Task — RL Training Loop (Python sidecar)

**Goal:** Train a policy network on Michael's gameplay data, deploy it as a bot
movement controller.

### Architecture:

```
Node.js game server
    │
    │  UDP or Unix socket (low latency)
    │  state vector → action vector
    │
Python sidecar process
    ├── During training: logs Michael's game state + actions to dataset
    ├── Behavioral cloning: trains policy on Michael's data
    └── Inference: serves trained policy to replace bot movement
```

### State vector per tick (what to send from Node to Python):
```
[
  bot_x, bot_y, bot_z,           // normalized to [-1, 1] by WORLD_SIZE/2
  bot_radius,                     // normalized by MAX_PLAYER_RADIUS
  nearest_human_dx, dy, dz,      // delta to nearest human, normalized
  nearest_human_radius,           // normalized
  nearest_food_dx, dy, dz,       // delta to nearest food cluster
  can_eat_nearest_human,          // binary: 1 if bot_radius > human_radius * 1.1
  human_can_eat_us,               // binary: 1 if human_radius > bot_radius * 1.1
]
```
Total: ~13 floats per tick.

### Action vector (what Python returns to Node):
```
[dx, dy, dz, speed_multiplier]   // direction normalized + speed scalar 0.5-1.5
```

### Python sidecar skeleton to build:

```python
# zorbio_agent.py
import socket, struct, numpy as np
import torch, torch.nn as nn

STATE_DIM  = 13
ACTION_DIM = 4

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_DIM),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

# Socket server: receive state, return action
# Training loop: behavioral cloning from recorded Michael gameplay
# Self-play refinement after initial cloning
```

### Data collection from Michael's sessions:
- Add a logging hook in `ServerPlayer.js` or `AppServer.js`
- Every tick, log: timestamp, player position, player radius, nearby entities,
  player velocity (derived from position delta)
- Output: JSONL file `~/zorbio/training_data/session_YYYY-MM-DD.jsonl`
- Michael plays normally, data accumulates, train offline

### Node side socket client (to add to Bot.js for RL bots):
```javascript
// rl: movement pattern that queries Python sidecar
rl: function moveRL() {
    // Build state vector, send over unix socket, get action back
    // Apply action as moveTowardPoint() call
    // Falls back to hunt() if socket unavailable
}
```

---

## Suggested Work Order for Claude Code

1. Read `Bot.js`, `BotController.js`, `common/config.js` in full
2. Apply Phase 1 changes (config + chase rewrite + hunt pattern + spawn logic)
3. Restart server, verify in logs that hunter bots are spawning
4. Check `AppServer.js` for where player capture events fire — that's the reward signal
5. Design the data logging schema for Michael's sessions
6. Build Python sidecar skeleton with socket server + PolicyNet
7. Add Node socket client stub to Bot.js as `rl` movement pattern
8. Wire data collection into AppServer.js tick loop
9. Test round-trip: Node sends state → Python returns random action → bot moves

---

## Notes / Constraints

- Michael runs on Ubuntu Pro, comfortable with CLI, custom Linux environments
- He uses voice-to-text so file/command names matter more than prose
- Legacy hardware consciousness: keep Python model small (shallow MLP, not transformer)
- No need for GPU on server side — inference on CPU is fine at this tick rate (50ms)
- The game is purely for personal use, private server, no external players
- MIT license — full freedom to modify anything
- Michael is genuinely skilled at these games — the RL opponent needs to be trained
  against his actual play patterns to be meaningful, not generic game AI
- Eventual goal: bots that mirror his own style back at him, creating a ghost-of-self
  opponent that exploits his tendencies
