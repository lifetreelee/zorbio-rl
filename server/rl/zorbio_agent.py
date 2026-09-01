#!/usr/bin/env python3
"""
Zorbio RL sidecar.

Serves bot movement decisions over a Unix domain socket to the Node game
server (see ../RLBridge.js). Also doubles as an offline behavioral-cloning
trainer/evaluator over logged human gameplay (see AppServer.js session
logging / SessionLogger.js).

Wire protocol v3 per request (see RLBridge.js for the authoritative spec -
keep PROTOCOL_VERSION/STATE_DIM/ACTION_DIM here in sync with there):
    request:  1 uint8   -- protocol version
              1 uint32  -- episode_id (unique per bot per episode)
              1 float32 -- prev_reward (reward for the transition into this tick)
              1 uint8   -- done (1 on an episode's final message)
              1 uint8   -- terminated (only meaningful if done: 1=death, 0=truncated)
              1 uint8   -- policy_id (0=live policy; nonzero=frozen opponent-pool slot)
              1 uint32  -- request_id (monotonic per episode - identifies the
                           action this request is asking for)
              1 uint32  -- caused_by_action_id (request_id of whichever earlier
                           action actually earned prev_reward - see below)
              38 float32, little-endian (152 bytes) -- state vector
    response:  6 float32, little-endian  (24 bytes) -- [dx, dy, dz, speed_multiplier,
                                                          use_boost, use_drain]
              (unchanged by v3 - everything else only ever flows client -> server)

v3 note: request/response round trips are asynchronous (Bot.js fires a
request then immediately moves using whatever action a *previous* response
already delivered), and reward-producing captures resolve on a separate,
slower game tick than the one that fires these requests - so the number of
ticks between "action applied" and "reward observed" isn't a fixed constant.
v2 assumed adjacency (row i's reward paired with row i-1's action), which is
wrong under both of those. request_id/caused_by_action_id let this process
pair reward with the exact action that earned it instead - see
handle_connection's pending_actions dict.

Phase 5 note: this process is inference-only. Every live-policy (policy_id=0)
transition gets appended to ROLLOUT_BUFFER_PATH as it completes, and
CHECKPOINT_PATH is polled every CHECKPOINT_POLL_SECONDS for a hot-swappable
update - but nothing in THIS process trains anything; --ppo in ppo_train.py
(a separate process, run manually or via --ppo-loop) is what consumes the
buffer and writes new checkpoints. Run --buffer-stats to sanity check what's
landed in the buffer between rounds.

Usage:
    python3 zorbio_agent.py                    # run the socket server (inference)
    python3 zorbio_agent.py --train PATTERN     # behavioral-clone from JSONL logs
    python3 zorbio_agent.py --eval PATTERN --checkpoint PATH  # score an existing checkpoint

See ppo_train.py for the PPO training loop (Phase 5b).
"""
import argparse
import glob
import json
import os
import random
import shutil
import socket
import struct
import sys
import threading
import time

PROTOCOL_VERSION = 3
STATE_DIM  = 38
ACTION_DIM = 6
SOCKET_PATH = os.environ.get('ZORBIO_RL_SOCKET', '/tmp/zorbio_rl.sock')
RL_DIR = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(RL_DIR, 'policy.pt')
CHECKPOINT_BACKUP_DIR = os.path.join(RL_DIR, 'checkpoints')
TRAINING_LOG_PATH = os.path.join(RL_DIR, 'training_log.jsonl')
EVAL_SESSION_PATH = os.path.join(RL_DIR, 'eval_session.json')

# Phase 7 (bookkeeping only - see zorbio_agent.py's --pin-baseline/--report):
# the BC-era eval (score against a pinned held-out human session) stops
# meaning anything once there's no human to imitate. Its replacement is a
# pinned baseline checkpoint plus fixed-seed current-vs-baseline match
# outcomes. Two real prerequisites for actually *running* that comparison -
# a seedable world RNG, and a match-orchestration mechanism (spawn exactly
# two bots, play to a decision, record it) - don't exist yet and aren't
# built here on purpose. What's here is just the pinning/recording/reporting
# so that whichever future phase builds the orchestrator has a format to
# write into and a baseline to compare against, rather than needing to
# invent both the mechanism and the bookkeeping format in one pass.
BASELINE_CHECKPOINT_MARKER = os.path.join(RL_DIR, 'baseline_checkpoint.json')
EVAL_MATCHES_PATH = os.path.join(RL_DIR, 'eval_matches.jsonl')

# Phase 5 of the real-RL scope: this process only ever does inference now -
# it appends every live-policy step to this file instead of training on
# anything itself. A separate process (not built yet - that's Phase 5b, the
# actual PPO loop) will consume this file and periodically write a new
# CHECKPOINT_PATH, which this process picks up via CHECKPOINT_POLL_SECONDS.
# Keeping training out of this process is the whole point: a PPO backward
# pass competing with this process's socket-serving threads for the GIL
# would make bots visibly stutter exactly when training is busiest.
ROLLOUT_BUFFER_PATH = os.path.join(RL_DIR, 'rollout_buffer.jsonl')
CHECKPOINT_POLL_SECONDS = 5

# Phase 5b: ppo_train.py's run-history log, parallel to TRAINING_LOG_PATH's
# BC run history above - kept as a separate file/format since PPO metrics
# (policy/value loss, clip fraction, transitions consumed) don't map onto
# the BC metrics shape (weighted_mse, direction_cos, ...).
PPO_TRAINING_LOG_PATH = os.path.join(RL_DIR, 'ppo_training_log.jsonl')

# Phase 6: self-play opponent pool. Backed by the same checkpoints/ directory
# backup_checkpoint() already writes to on every training run - no separate
# pool storage needed. Node samples a plain integer slot id (1..RL_POOL_SIZE
# in common/config.js) at episode start; this process resolves what (if
# anything) currently occupies that slot, so Node never needs to know real
# checkpoint filenames.
POOL_MAX_SIZE = 8

# Careless deaths teach the wrong lesson - a human's last few seconds before
# getting caught by another PLAYER are exactly the play we don't want to
# imitate. Bot kills aren't excluded (they're mostly just the bot being
# strong, not the human being careless) - only human-vs-human deaths are.
DEATH_WINDOW_SECONDS = 3.0

# version, episode_id, prev_reward, done, terminated, policy_id, request_id, caused_by_action_id
HEADER_STRUCT = struct.Struct('<BIfBBBII')
STATE_STRUCT  = struct.Struct('<%df' % STATE_DIM)
ACTION_STRUCT = struct.Struct('<%df' % ACTION_DIM)

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0  # exp(1.0) ~= 2.7 - plenty for a [-1,1]-ish action range without exploding

if HAVE_TORCH:
    class PolicyNet(nn.Module):
        """Small MLP - CPU inference at 20Hz per bot is trivial, no need for
        anything bigger given the legacy hardware this runs on.

        Phase 4 of the real-RL scope: the old network was a direct regression
        head (state in, one deterministic action out), fine for behavioral
        cloning but wrong for PPO, which needs a *distribution* to sample from
        during training. This splits the 6 action channels into:
          - 4 continuous channels (dx, dy, dz, speed_multiplier) as an
            independent Gaussian - mean is state-dependent, std is a single
            learned state-independent parameter (a common simplification for
            small continuous-control problems like this one).
          - 2 binary channels (use_boost, use_drain) as independent Bernoulli
            logits. Only boost is ever trained/used (see TorchPolicy.act) -
            drain has no real labels yet, same as before this change.

        Resolved simplification (was flagged here through Phase 4): the
        Gaussian's mean is tanh-squashed as part of computing the
        *location parameter* of Normal(mean, std) - it is not applied to a
        sample afterward. log_prob is always taken against that same
        Normal(mean, std) for whatever value is passed in (the mean itself
        for deterministic serving, or a real sample for PPO exploration), so
        it's exact either way with no tanh-Jacobian correction needed. What
        would need that correction is squashing *after* sampling, which
        nothing here does.

        Phase 5b addition: value_head turns this into an actor-critic net -
        PPO's advantage estimation (GAE) needs a state-value estimate
        alongside the action distribution, so PolicyNet now outputs both.
        This is a checkpoint-format change like Phase 4's was; the same
        load_policy() try/except handles it the same way (serve untrained
        rather than crash on an incompatible old checkpoint).

        Phase 5b fix (2026-08-29): value_head originally shared a single
        trunk with mean_head. value_loss is ~1e4x policy_loss in magnitude
        (discounted returns land in the tens, policy loss doesn't), so at
        vf_coef=0.5 the value objective owned the shared trunk outright and
        dragged its output magnitude up over the course of a run to
        represent ever-larger returns - saturating mean_head's tanh as
        collateral damage. This recurred three times (once pre-fix, twice
        post weight-decay-only fixes that didn't address the actual
        mechanism). Splitting into separate trunks severs that coupling
        structurally: value_trunk can grow however large the value
        objective demands without touching policy_trunk / mean_head at all.
        Paired with a 20x reward-constant reduction (see config.js) that
        shrinks the value target magnitude at the source.
        """
        CONT_DIM = 4  # dx, dy, dz, speed_multiplier
        BIN_DIM  = 2  # use_boost, use_drain

        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(STATE_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.value_trunk = nn.Sequential(
                nn.Linear(STATE_DIM, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            self.mean_head   = nn.Linear(64, self.CONT_DIM)
            self.log_std     = nn.Parameter(torch.zeros(self.CONT_DIM))
            self.binary_head = nn.Linear(64, self.BIN_DIM)
            self.value_head  = nn.Linear(64, 1)

        def forward(self, x):
            h = self.trunk(x)
            mean = torch.tanh(self.mean_head(h))
            log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
            binary_logits = self.binary_head(h)
            value = self.value_head(self.value_trunk(x))
            return mean, log_std.expand_as(mean), binary_logits, value


class RandomPolicy:
    """Fallback used when torch isn't installed, or no checkpoint exists yet.
    Produces a plausible-looking (but untrained) action so the Node<->Python
    round trip can be exercised end to end before any real model exists."""
    def act(self, state, deterministic=True, return_extras=False):
        action = [
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            1.0,   # speed_multiplier
            0.0,   # use_boost
            0.0,   # use_drain
        ]
        if return_extras:
            # no log_prob/value under a model-free random policy - callers
            # (the rollout logger) must treat a None log_prob as unusable
            # for PPO, not as a zero
            return action, None, None
        return action


class TorchPolicy:
    def __init__(self, net):
        self.net = net
        self.net.eval()

    def act(self, state, deterministic=True, return_extras=False):
        """deterministic=True (default, used for live bot serving and eval)
        returns the distribution's mean/argmax - stable, repeatable behavior.
        deterministic=False samples instead - used for the live policy
        (policy_id=0) during serving, since PPO needs real exploration to
        have anything to learn from; opponent-pool bots stay deterministic
        for stable, repeatable behavior.

        return_extras=True additionally returns (log_prob, value): the
        sampled/mean action's log-probability under this net's current
        distribution, and this state's value estimate - both needed to log a
        usable PPO training row. Only meaningful for policy_id=0 traffic;
        never requested for opponent-pool actions."""
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, log_std, binary_logits, value = self.net(x)
            std = log_std.exp()
            cont_dist = torch.distributions.Normal(mean, std)
            bin_dist  = torch.distributions.Bernoulli(logits=binary_logits[:, 0:1])

            if deterministic:
                cont_sample  = mean
                boost_sample = (binary_logits[:, 0:1] > 0).float()
            else:
                cont_sample  = cont_dist.sample()
                boost_sample = bin_dist.sample()

            dx, dy, dz, speed_raw = cont_sample.squeeze(0).tolist()
            boost = boost_sample.item()

            log_prob = None
            if return_extras:
                # against the exact (unclamped) sample actually taken - see
                # the resolved-simplification note on PolicyNet.forward
                logp = cont_dist.log_prob(cont_sample).sum(dim=1) + bin_dist.log_prob(boost_sample).sum(dim=1)
                log_prob = logp.item()

        # remap tanh-range speed to a usable multiplier - keep in sync with
        # how targets are encoded in row_to_target() below, and with
        # ppo_train.py's action_to_raw() inverse
        speed = 1.0 + speed_raw * 0.5      # -> 0.5 .. 1.5
        # drain isn't trained on real labels yet (SessionLogger always logs
        # drained=false) - never let an untrained channel drive behavior
        drain = 0.0
        action = [dx, dy, dz, speed, boost, drain]

        if return_extras:
            return action, log_prob, value.item()
        return action


def load_policy():
    if not HAVE_TORCH:
        print('zorbio_agent: torch not installed, serving RandomPolicy '
              '(pip install torch to enable real inference/training)')
        return RandomPolicy()

    net = PolicyNet()

    if os.path.exists(CHECKPOINT_PATH):
        try:
            net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
            print('zorbio_agent: loaded checkpoint from', CHECKPOINT_PATH)
        except RuntimeError as e:
            # expected once, right after the Phase 4 architecture change - old
            # checkpoints were a plain regression head, not this distribution
            # split, so their state_dict keys don't match. Serve untrained
            # rather than crash; retrain to get a compatible checkpoint.
            print('zorbio_agent: checkpoint at', CHECKPOINT_PATH,
                  'is incompatible with the current network architecture - '
                  'serving an untrained PolicyNet instead. Retrain to fix.')
            print('  (%s)' % e)
    else:
        print('zorbio_agent: no checkpoint found at', CHECKPOINT_PATH,
              '- serving an untrained PolicyNet (random weights)')

    return TorchPolicy(net)


def append_transition(episode_id, state, action, log_prob, reward, next_state, done, terminated):
    """Append one COMPLETE (s, a, log_prob, r, s', done, terminated) transition
    to the rollout buffer for ppo_train.py to consume. Caller
    (handle_connection) only ever calls this for policy_id==0 (the live
    training policy) - frozen opponent-pool bots (Phase 6) are environment,
    not students, and must never leak into training data.

    Unlike the v2 scheme this replaced, there's no row-adjacency to
    reconstruct: handle_connection's pending_actions dict already paired
    reward with the exact action that earned it (via request_id/
    caused_by_action_id - see RLBridge.js's v3 protocol note), so what's
    written here is already a ready-to-train transition, one per row.

    Opened fresh on every call (not held open) so ppo_train.py can safely
    rotate this file out from under this process (rename it away, this just
    recreates it on the next write) without any coordination.
    """
    row = {
        'ts'        : time.time(),
        'episode_id': episode_id,
        'state'     : state,
        'action'    : action,
        'log_prob'  : log_prob,
        'reward'    : reward,
        'next_state': next_state,
        'done'      : bool(done),
        'terminated': bool(terminated),
    }
    try:
        with open(ROLLOUT_BUFFER_PATH, 'a') as f:
            f.write(json.dumps(row) + '\n')
    except OSError as e:
        print('zorbio_agent: failed to append transition (dropped):', e)


class PolicyHolder:
    """Holds the currently-serving policy behind a single attribute so it can
    be hot-swapped by one atomic reference assignment (safe under the GIL,
    no lock needed) without ever mutating a policy object that's mid-use on
    another connection's serving thread."""
    def __init__(self, policy):
        self.policy = policy
        self._mtime = self._checkpoint_mtime()

    @staticmethod
    def _checkpoint_mtime():
        try:
            return os.path.getmtime(CHECKPOINT_PATH)
        except OSError:
            return None

    def maybe_reload(self):
        mtime = self._checkpoint_mtime()
        if mtime is None or mtime == self._mtime:
            return

        try:
            new_policy = load_policy()
        except Exception as e:
            print('zorbio_agent: failed to reload checkpoint, keeping current policy:', e)
            return

        self.policy = new_policy  # atomic swap - see class docstring
        self._mtime = mtime
        print('zorbio_agent: hot-reloaded checkpoint (mtime changed)')


class OpponentPool:
    """Frozen past-checkpoint opponents for self-play, so live-policy bots
    aren't only ever training against a mirror of themselves updating out
    from under them mid-episode. Slots are numbered 1..N by recency (1 =
    newest) purely for the wire - see policyId in Bot.js."""
    def __init__(self, max_size=POOL_MAX_SIZE):
        self.max_size = max_size
        self.slots = {}    # slot_id (1..N) -> TorchPolicy
        self._paths = []   # source paths for the current slots, for change detection
        self.refresh()

    def refresh(self):
        if not HAVE_TORCH or not os.path.isdir(CHECKPOINT_BACKUP_DIR):
            return

        candidate_paths = sorted(
            glob.glob(os.path.join(CHECKPOINT_BACKUP_DIR, '*.pt')),
            key=os.path.getmtime, reverse=True
        )[:self.max_size]

        # This runs every CHECKPOINT_POLL_SECONDS (5s) in a background
        # thread sharing the GIL with every live bot's connection handler.
        # checkpoints/ only actually gets a new file every ~5 minutes (one
        # PPO round), so on nearly every poll candidate_paths is identical
        # to last time - without this check, every single poll was doing a
        # full glob + 8x fresh PolicyNet construction + torch.load from disk
        # regardless, which got a lot more expensive once checkpoints/
        # accumulated hundreds of files. Real candidate for the periodic
        # multi-second hangs RLBridge.js's watchdog has been recovering
        # from (2026-08-29) - this fixes the likely cause instead of just
        # relying on the watchdog to paper over it every time.
        if candidate_paths == self._paths:
            return

        slots = {}
        loaded_paths = []
        for i, path in enumerate(candidate_paths):
            try:
                slots[i + 1] = TorchPolicy(load_checkpoint(path))
                loaded_paths.append(path)
            except RuntimeError:
                # architecture-incompatible checkpoint (e.g. pre-Phase-4,
                # still sitting in checkpoints/ from before that change) -
                # just leave this slot unfilled rather than crash the pool
                continue

        if loaded_paths != self._paths:
            print('zorbio_agent: opponent pool refreshed - %d usable checkpoint(s): %s' % (
                len(slots), ', '.join(os.path.basename(p) for p in loaded_paths) or '(none)'))

        self.slots = slots
        self._paths = loaded_paths

    def get(self, slot_id):
        """Policy for slot_id, or None if that slot isn't currently filled
        (pool smaller than the sampling range, or emptied since the bot's
        episode started) - caller should fall back to the live policy."""
        return self.slots.get(slot_id)


def checkpoint_poll_loop(holder, pool, interval=CHECKPOINT_POLL_SECONDS):
    while True:
        time.sleep(interval)
        holder.maybe_reload()
        pool.refresh()


def recv_exact(conn, n):
    """Read exactly n bytes from a stream socket, or return None on EOF."""
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def handle_connection(conn, holder, pool):
    print('zorbio_agent: bot connected')

    # (episode_id, request_id) -> {'state', 'action', 'log_prob'} for
    # live-policy (policy_id==0) actions awaiting the reward that will
    # complete their transition - see the v3 protocol note on
    # caused_by_action_id. Scoped to this connection so it's discarded
    # wholesale on disconnect rather than grown across reconnects.
    #
    # Known bounded leak: an episode that dies within its first tick (no
    # action ever applied before death) leaves its one stored action
    # unpopped, since the closing message's caused_by_action_id is 0. Rare
    # and tiny (one state vector's worth) - not worth pruning machinery for.
    pending_actions = {}

    try:
        while True:
            raw_header = recv_exact(conn, HEADER_STRUCT.size)
            if raw_header is None:
                break

            (version, episode_id, prev_reward, done, terminated, policy_id,
             request_id, caused_by_action_id) = HEADER_STRUCT.unpack(raw_header)

            if version != PROTOCOL_VERSION:
                print('zorbio_agent: protocol version mismatch (got %d, expected %d) - '
                      'closing connection rather than risk misparsing the stream'
                      % (version, PROTOCOL_VERSION))
                break

            raw_state = recv_exact(conn, STATE_STRUCT.size)
            if raw_state is None:
                break

            state = list(STATE_STRUCT.unpack(raw_state))

            if done:
                print('zorbio_agent: episode %d closed - %s - final reward %.3f'
                      % (episode_id, 'terminated' if terminated else 'truncated', prev_reward))

            # complete whichever earlier action earned prev_reward, if any -
            # caused_by_action_id==0 means none applied yet this episode
            if policy_id == 0 and caused_by_action_id:
                pending = pending_actions.pop((episode_id, caused_by_action_id), None)
                if pending is not None and pending['log_prob'] is not None:
                    append_transition(episode_id, pending['state'], pending['action'], pending['log_prob'],
                                       prev_reward, state, done, terminated)

            if policy_id == 0:
                policy = holder.policy
                action, log_prob, _value = policy.act(state, deterministic=False, return_extras=True)
                # a closing message's action is discarded by Bot.js and has
                # no future tick to pair it with - don't bother remembering it
                if not done:
                    pending_actions[(episode_id, request_id)] = {
                        'state': state, 'action': action, 'log_prob': log_prob,
                    }
            else:
                policy = pool.get(policy_id) or holder.policy  # empty slot -> fall back to live
                action = policy.act(state, deterministic=True)

            conn.sendall(ACTION_STRUCT.pack(*action))
    finally:
        conn.close()
        print('zorbio_agent: bot disconnected')


def serve(policy):
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(8)
    print('zorbio_agent: listening on', SOCKET_PATH)

    holder = PolicyHolder(policy)
    pool = OpponentPool()
    threading.Thread(target=checkpoint_poll_loop, args=(holder, pool), daemon=True).start()

    try:
        while True:
            conn, _ = server.accept()
            # No socket-level recv timeout here on purpose: this connection
            # is legitimately idle for long stretches (e.g. whenever no bot
            # is currently above RL_BOT_SIZE_THRESHOLD, which can be minutes
            # after a population reset), and a recv() timeout can't tell that
            # apart from a real hang - it fired every ~15s during ordinary
            # idle periods when tried (2026-08-28), for no benefit: if
            # RLBridge.js's watchdog ever destroys this socket because a
            # request genuinely went unanswered, recv() below unblocks via a
            # normal EOF with no timeout needed. See RLBridge.js for the
            # actual (request-scoped, not idle-scoped) hang detection.
            # each connection gets its own thread so one stuck/slow bot can't
            # block others from connecting or getting served
            threading.Thread(target=handle_connection, args=(conn, holder, pool), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)


def row_to_state(row):
    """Rebuild the flat 38-float state vector from a logged JSONL row. Field
    order here must match common/RLState.js exactly - see SessionLogger.js
    for how each field was populated from that same vector."""
    s, nt, nth, w, f = row['self'], row['nearest_target'], row['nearest_threat'], row['walls'], row['food']
    return [
        s['x'], s['y'], s['z'], s['r'],
        s['vx'], s['vy'], s['vz'], s['speed'],
        nt['dx'], nt['dy'], nt['dz'], nt['r'],
        nt['vx'], nt['vy'], nt['vz'], nt['speed'],
        nt['intercept_dx'], nt['intercept_dy'], nt['intercept_dz'],
        1.0 if nt['can_eat'] else 0.0,
        1.0 if nt['they_can_eat'] else 0.0,
        nt['size_ratio'],
        w['self_x'], w['self_y'], w['self_z'],
        nt['wall_dist_x'], nt['wall_dist_y'], nt['wall_dist_z'],
        1.0 if nt['cornered'] else 0.0,
        1.0 if s.get('boost_available') else 0.0,
        f['dx'], f['dy'], f['dz'], f['cluster_value'],
        nth['dx'], nth['dy'], nth['dz'], nth['r'],
    ]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def row_to_target(row):
    """Build the network's *training* target - not the same units as the raw
    logged action. The net's output head is Tanh (bounded to [-1, 1] per
    channel), so every channel here is encoded into that same range, using
    the exact inverse of the remap TorchPolicy.act() applies at inference:

      - dx/dy/dz: raw velocity components can have magnitude way over 1
        (real speeds are usually >1 unit/tick), but only *direction* is ever
        used downstream (Bot.js normalizes it) - so train on the unit
        direction vector, not the raw magnitude, or the loss just saturates.
      - speed:  multiplier ~0.5..1.5 -> inverse of `1.0 + t*0.5`
      - boost:  encoded as -1/+1 (not 0/1) so the decision boundary used at
        inference (`boost > 0`) sits exactly between the two classes instead
        of at the edge of a 0..1 range dominated by the majority class.
      - drain:  no real labels exist yet (SessionLogger always logs
        drained=false) - target is a constant that carries no signal, so
        it's excluded from the loss via row_weight() below rather than
        trained on.
    """
    a = row['action']

    dx, dy, dz = a['dx'], a['dy'], a['dz']
    norm = (dx * dx + dy * dy + dz * dz) ** 0.5
    if norm > 1e-6:
        dx, dy, dz = dx / norm, dy / norm, dz / norm
    else:
        dx = dy = dz = 0.0

    speed_t = _clamp((a['speed'] - 1.0) / 0.5, -1.0, 1.0)
    boost_t = 1.0 if a.get('boosted') else -1.0

    return [dx, dy, dz, speed_t, boost_t, 0.0]


# per-channel loss weight: drain (index 5) is untrained/unlabeled, exclude it
LOSS_WEIGHT = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]


def load_dataset(pattern, death_window=DEATH_WINDOW_SECONDS, verbose=True):
    """Read logged session JSONL files into (state, target, meta) triples for
    behavioral cloning, excluding the seconds immediately before a human
    died to another human PLAYER (see SessionLogger.logDeath / capturePlayer
    in AppServer.js). Bot kills are left in - those aren't "careless play",
    the bot just won.
    """
    paths = sorted(glob.glob(pattern))
    states, targets, meta = [], [], []
    total_rows = 0
    excluded_rows = 0

    for path in paths:
        session = os.path.basename(path)
        ticks = []
        # (player_id, death_ts) for every death caused by another PLAYER
        pvp_deaths = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                if row.get('event') == 'death':
                    if row.get('killed_by') == 'PLAYER':
                        pvp_deaths.append((row['player_id'], row['ts']))
                    continue

                ticks.append(row)

        window_ms = death_window * 1000.0
        for row in ticks:
            total_rows += 1
            pid, ts = row.get('player_id'), row['ts']

            excluded = any(
                pid == death_pid and 0 <= (death_ts - ts) <= window_ms
                for death_pid, death_ts in pvp_deaths
            )
            if excluded:
                excluded_rows += 1
                continue

            states.append(row_to_state(row))
            targets.append(row_to_target(row))
            meta.append({'session': session, 'ts': ts, 'raw_action': row['action']})

    if verbose:
        print('zorbio_agent: loaded %d rows from %d session file(s), excluded %d '
              '(%.1f%%) as pre-death carelessness' % (
                  total_rows, len(paths), excluded_rows,
                  100.0 * excluded_rows / total_rows if total_rows else 0.0))

    return states, targets, meta


def get_eval_session(sessions):
    """Pick (and permanently remember) one session file as the fixed
    validation benchmark, instead of always holding out "whatever's most
    recent". A moving val set makes run-to-run comparison meaningless - run 5
    scoring better than run 3 might just mean run 5's held-out session was
    easier. Pinning one session forever means every checkpoint from now on is
    judged against the exact same held-out gameplay, so the trend in
    training_log.jsonl is actually trustworthy.
    """
    pinned = None
    if os.path.exists(EVAL_SESSION_PATH):
        with open(EVAL_SESSION_PATH) as f:
            pinned = json.load(f).get('session')

    if pinned and pinned in sessions:
        return pinned

    if pinned:
        print('zorbio_agent: WARNING - pinned eval session %s is no longer in the '
              'dataset (deleted/moved?), picking a new one' % pinned)

    # pin the earliest session with a healthy number of rows as the permanent
    # benchmark - "earliest" just so it's excluded from training as of now
    # and everything after it (including today's play) goes into training
    chosen = sessions[0]
    with open(EVAL_SESSION_PATH, 'w') as f:
        json.dump({'session': chosen, 'pinned_at': time.time()}, f)
    print('zorbio_agent: pinned %s as the permanent held-out eval session' % chosen)
    return chosen


def split_by_session(states, targets, meta):
    """Session-based train/val split - never a random row split. Adjacent
    rows are logged at 20Hz and are near-duplicates, so a random split leaks
    and makes validation loss meaningless. Held-out data must be a whole
    session the model never saw any part of - and it's always the *same*
    session run to run (see get_eval_session) so results are comparable over
    time.

    Falls back to a time-based tail split within a single session when only
    one session file is available, with a caveat printed - that's weaker
    (still no diversity of play), but better than no validation at all.
    """
    sessions = sorted(set(m['session'] for m in meta))

    if len(sessions) >= 2:
        val_session = get_eval_session(sessions)
        train_idx = [i for i, m in enumerate(meta) if m['session'] != val_session]
        val_idx   = [i for i, m in enumerate(meta) if m['session'] == val_session]
        print('zorbio_agent: validating on pinned session %s (%d rows), '
              'training on the other %d session(s) (%d rows)' % (
                  val_session, len(val_idx), len(sessions) - 1, len(train_idx)))
    else:
        print('zorbio_agent: WARNING - only one session file matched, falling back to '
              'a time-based 80/20 split within it. Validation here only tells you '
              'about held-out *time*, not held-out play style - treat results as weak '
              'until there are 2+ sessions to hold a whole one out.')
        order = sorted(range(len(meta)), key=lambda i: meta[i]['ts'])
        split_at = int(len(order) * 0.8)
        train_idx, val_idx = order[:split_at], order[split_at:]

    def gather(idx):
        return ([states[i] for i in idx], [targets[i] for i in idx], [meta[i] for i in idx])

    return gather(train_idx), gather(val_idx)


def evaluate(net, states, targets, meta):
    """Score a policy against held-out data with per-channel metrics instead
    of one aggregate MSE (which is dominated by whichever channel has the
    largest scale and hides everything else)."""
    if not states:
        return None

    with torch.no_grad():
        x = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)

        # deterministic mean/logit reconstructed into the old plain-6-vector
        # shape so the rest of this metric code (written for the pre-Phase-4
        # regression head) doesn't need to change
        mean, _log_std, binary_logits, _value = net(x)
        pred = torch.cat([mean, torch.tanh(binary_logits)], dim=1)

        weight = torch.tensor(LOSS_WEIGHT)
        weighted_mse = (((pred - y) ** 2) * weight).sum() / weight.sum()

        pred_dir = pred[:, 0:3]
        true_dir = y[:, 0:3]
        pred_norm = pred_dir.norm(dim=1, keepdim=True).clamp_min(1e-6)
        true_norm = true_dir.norm(dim=1, keepdim=True).clamp_min(1e-6)
        cos_sim = ((pred_dir / pred_norm) * (true_dir / true_norm)).sum(dim=1)
        # rows where the human wasn't moving have no real direction target -
        # cosine similarity against a zero vector is meaningless, skip them
        moving = true_dir.norm(dim=1) > 1e-3
        mean_cos_sim = cos_sim[moving].mean().item() if moving.any() else float('nan')

        # speed MAE in real multiplier units (undo the tanh remap)
        pred_speed = 1.0 + pred[:, 3] * 0.5
        true_speed = 1.0 + y[:, 3] * 0.5
        speed_mae = (pred_speed - true_speed).abs().mean().item()

        boost_acc = ((pred[:, 4] > 0) == (y[:, 4] > 0)).float().mean().item()

    return {
        'n'             : len(states),
        'weighted_mse'  : weighted_mse.item(),
        'direction_cos' : mean_cos_sim,
        'speed_mae'     : speed_mae,
        'boost_accuracy': boost_acc,
    }


def format_metrics(m):
    if m is None:
        return '(no data)'
    return ('n=%d  weighted_mse=%.4f  direction_cos_sim=%.3f  speed_mae=%.3f  boost_acc=%.3f'
            % (m['n'], m['weighted_mse'], m['direction_cos'], m['speed_mae'], m['boost_accuracy']))


def compute_bc_loss(net, x, y, weight):
    """Behavioral-cloning loss for the Phase 4 stochastic head: Gaussian NLL
    on the continuous channels (dx, dy, dz, speed) against the BC target,
    BCE on the boost channel against its sign. Same per-channel weighting
    (and same drain exclusion) as the old plain-MSE loss, just against a
    distribution instead of a point estimate now that the network is one."""
    mean, log_std, binary_logits, _value = net(x)
    std = log_std.exp()

    cont_target = y[:, 0:4]
    cont_nll = -torch.distributions.Normal(mean, std).log_prob(cont_target)  # [N,4]

    boost_target01 = (y[:, 4:5] + 1.0) / 2.0  # row_to_target encodes boost as -1/+1
    boost_bce = nn.functional.binary_cross_entropy_with_logits(
        binary_logits[:, 0:1], boost_target01, reduction='none')

    drain_term = torch.zeros_like(boost_bce)  # no real labels - stays excluded via weight

    per_channel = torch.cat([cont_nll, boost_bce, drain_term], dim=1)
    return (per_channel * weight).sum(dim=1).mean() / weight.sum()


def pin_baseline(checkpoint_path):
    """Pin a checkpoint as the fixed comparison point for future eval matches
    (Phase 7). Validates it actually loads under the current architecture
    first - pinning something incompatible would silently make every future
    win-rate comparison meaningless, same failure shape as an incompatible
    checkpoint anywhere else in this file, just worse since nothing else
    would catch it later."""
    checkpoint_path = os.path.abspath(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        print('zorbio_agent: no checkpoint at', checkpoint_path, file=sys.stderr)
        sys.exit(1)

    try:
        load_checkpoint(checkpoint_path)
    except RuntimeError as e:
        print('zorbio_agent: refusing to pin', checkpoint_path,
              '- incompatible with the current network architecture:', e, file=sys.stderr)
        sys.exit(1)

    with open(BASELINE_CHECKPOINT_MARKER, 'w') as f:
        json.dump({'path': checkpoint_path, 'pinned_at': time.time()}, f)

    print('zorbio_agent: pinned baseline checkpoint:', checkpoint_path)


def get_baseline():
    """Returns the pinned baseline checkpoint's path, or None (with an
    explanatory print) if nothing's pinned yet or the pinned file no longer
    loads under the current architecture."""
    if not os.path.exists(BASELINE_CHECKPOINT_MARKER):
        return None

    with open(BASELINE_CHECKPOINT_MARKER) as f:
        path = json.load(f)['path']

    if not os.path.exists(path):
        print('zorbio_agent: pinned baseline', path, 'no longer exists on disk')
        return None

    try:
        load_checkpoint(path)
    except RuntimeError:
        print('zorbio_agent: pinned baseline', path,
              'is incompatible with the current network architecture - re-pin a fresh one')
        return None

    return path


def record_eval_match(current_checkpoint, baseline_checkpoint, seed, winner,
                       current_final_scale, baseline_final_scale, duration_ticks):
    """Append one completed eval match's outcome. Not called from anywhere
    yet - this is the format the not-yet-built match orchestrator writes
    into. winner is 'current', 'baseline', or 'draw'."""
    row = {
        'ts'                  : time.time(),
        'current_checkpoint'  : current_checkpoint,
        'baseline_checkpoint' : baseline_checkpoint,
        'seed'                : seed,
        'winner'              : winner,
        'current_final_scale' : current_final_scale,
        'baseline_final_scale': baseline_final_scale,
        'duration_ticks'      : duration_ticks,
    }
    with open(EVAL_MATCHES_PATH, 'a') as f:
        f.write(json.dumps(row) + '\n')


def eval_win_rate(n=20):
    """Win rate for 'current' over the most recent n recorded eval matches
    (see record_eval_match). Returns None if there's nothing recorded yet."""
    if not os.path.exists(EVAL_MATCHES_PATH):
        return None

    with open(EVAL_MATCHES_PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        return None

    rows = rows[-n:]
    wins  = sum(1 for r in rows if r['winner'] == 'current')
    draws = sum(1 for r in rows if r['winner'] == 'draw')

    return {
        'n'       : len(rows),
        'win_rate': wins / len(rows),
        'draws'   : draws,
    }


def backup_checkpoint():
    """Copy the current checkpoint aside before overwriting it, so a new
    training run can always be compared back against what was actually
    running before - not just re-derived training-loss numbers."""
    if not os.path.exists(CHECKPOINT_PATH):
        return None

    os.makedirs(CHECKPOINT_BACKUP_DIR, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    backup_path = os.path.join(CHECKPOINT_BACKUP_DIR, 'policy_%s.pt' % stamp)
    shutil.copy2(CHECKPOINT_PATH, backup_path)
    print('zorbio_agent: backed up current checkpoint to', backup_path)
    return backup_path


def load_checkpoint(path):
    net = PolicyNet()
    net.load_state_dict(torch.load(path, map_location='cpu'))
    net.eval()
    return net


def train(pattern, epochs=20, batch_size=256, lr=1e-3, eval_every=5):
    if not HAVE_TORCH:
        print('zorbio_agent: torch is required for training '
              '(pip install torch numpy)', file=sys.stderr)
        sys.exit(1)

    states, targets, meta = load_dataset(pattern)
    if not states:
        print('zorbio_agent: no training examples found matching', pattern,
              file=sys.stderr)
        sys.exit(1)

    (train_states, train_targets, train_meta), (val_states, val_targets, val_meta) = \
        split_by_session(states, targets, meta)

    # score whatever is currently deployed on the same held-out data before
    # touching it, so "improved or degraded" is a real before/after
    baseline_metrics = None
    if os.path.exists(CHECKPOINT_PATH):
        try:
            baseline_net = load_checkpoint(CHECKPOINT_PATH)
            baseline_metrics = evaluate(baseline_net, val_states, val_targets, val_meta)
            print('zorbio_agent: baseline (currently deployed) checkpoint on held-out data:')
            print('  ', format_metrics(baseline_metrics))
        except RuntimeError:
            print('zorbio_agent: existing checkpoint predates the current network '
                  'architecture - skipping before/after baseline comparison this run')

    backup_path = backup_checkpoint()

    x = torch.tensor(train_states, dtype=torch.float32)
    y = torch.tensor(train_targets, dtype=torch.float32)
    weight = torch.tensor(LOSS_WEIGHT)
    n = x.shape[0]

    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    print('zorbio_agent: training on %d rows (batch_size=%d, %d epochs = %d steps)' % (
        n, batch_size, epochs, epochs * max(1, n // batch_size)))

    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        steps = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            bx, by = x[idx], y[idx]

            opt.zero_grad()
            loss = compute_bc_loss(net, bx, by, weight)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            steps += 1

        msg = 'epoch %d/%d - train loss %.5f' % (epoch + 1, epochs, epoch_loss / steps)

        # periodically check val loss *during* training - train loss alone can't
        # tell you if you're overfitting, only train-vs-val divergence can. If val
        # loss stops following train loss down (or starts climbing), that's the
        # signal to stop adding epochs, not a lower train-loss number.
        if eval_every and ((epoch + 1) % eval_every == 0 or epoch + 1 == epochs) and val_states:
            net.eval()
            val_metrics = evaluate(net, val_states, val_targets, val_meta)
            msg += '   val weighted_mse=%.4f  direction_cos=%.3f' % (
                val_metrics['weighted_mse'], val_metrics['direction_cos'])
            net.train()

        print(msg)

    net.eval()
    new_metrics = evaluate(net, val_states, val_targets, val_meta)
    print('zorbio_agent: new checkpoint on held-out data:')
    print('  ', format_metrics(new_metrics))

    if baseline_metrics and new_metrics:
        delta = new_metrics['weighted_mse'] - baseline_metrics['weighted_mse']
        direction = 'improved' if delta < 0 else 'degraded' if delta > 0 else 'unchanged'
        print('zorbio_agent: weighted_mse %s by %.4f (%.1f%% of baseline)' % (
            direction, abs(delta),
            100.0 * abs(delta) / baseline_metrics['weighted_mse'] if baseline_metrics['weighted_mse'] else 0.0))
        if len(set(m['session'] for m in val_meta)) < 1 or baseline_metrics['n'] < 500:
            print('zorbio_agent: NOTE - small/limited validation set, treat this delta as noisy '
                  'until there is more held-out play to evaluate against.')

    torch.save(net.state_dict(), CHECKPOINT_PATH)
    print('zorbio_agent: saved checkpoint to', CHECKPOINT_PATH)

    with open(TRAINING_LOG_PATH, 'a') as f:
        f.write(json.dumps({
            'ts'               : time.time(),
            'pattern'          : pattern,
            'train_rows'       : len(train_states),
            'val_rows'         : len(val_states),
            'epochs'           : epochs,
            'batch_size'       : batch_size,
            'baseline_checkpoint': backup_path,
            'baseline_metrics' : baseline_metrics,
            'new_metrics'      : new_metrics,
        }) + '\n')
    print('zorbio_agent: appended run summary to', TRAINING_LOG_PATH)


def _print_eval_section():
    baseline = get_baseline()
    print('zorbio_agent: Phase 7 eval status')
    if baseline:
        print('  baseline checkpoint:', baseline)
    else:
        print('  no baseline checkpoint pinned yet (see --pin-baseline)')

    win_rate = eval_win_rate()
    if win_rate:
        print('  win rate vs baseline: %.1f%% over last %d match(es) (%d draws)' % (
            win_rate['win_rate'] * 100, win_rate['n'], win_rate['draws']))
    else:
        print('  no eval matches recorded yet (match orchestration not built)')
    print()


def _print_ppo_section():
    print('zorbio_agent: Phase 5b PPO run history')
    if not os.path.exists(PPO_TRAINING_LOG_PATH):
        print('  no PPO rounds logged yet (run ppo_train.py)')
        print()
        return

    with open(PPO_TRAINING_LOG_PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print('%-20s %10s %9s %11s %10s %9s' % (
        'when', 'transns', 'episodes', 'policy_loss', 'value_loss', 'entropy'))
    for run in rows[-10:]:
        when = time.strftime('%m-%d %H:%M', time.localtime(run['ts']))
        print('%-20s %10d %9d %11.4f %10.4f %9.4f' % (
            when, run['n_transitions'], run['n_episodes'],
            run['policy_loss'], run['value_loss'], run['entropy']))
    print()


def report():
    """Print the improve/degrade trend across every training run logged in
    training_log.jsonl. Since split_by_session now pins a single fixed eval
    session (get_eval_session), every row's new_metrics is scored against the
    same held-out gameplay, so this trend is directly comparable run to run -
    unlike the first two runs on record, which each held out whatever the
    most-recent session happened to be at the time (flagged below).

    Also prints the Phase 7 baseline/win-rate section (pinned checkpoint,
    recent eval match win rate) if there's anything to show - this is the
    metric meant to eventually *replace* the BC trend above once RL training
    is actually running, not sit alongside it forever; it's additive for now
    only because there's no real eval data yet to make that replacement."""
    _print_eval_section()
    _print_ppo_section()

    if not os.path.exists(TRAINING_LOG_PATH):
        print('zorbio_agent: no training runs logged yet at', TRAINING_LOG_PATH)
        return

    pinned = None
    if os.path.exists(EVAL_SESSION_PATH):
        with open(EVAL_SESSION_PATH) as f:
            pinned = json.load(f).get('session')

    print('%-20s %8s %8s %10s %8s %9s %8s' % (
        'when', 'train_n', 'val_n', 'weighted_mse', 'delta', 'dir_cos', 'speed_mae'))

    prev_mse = None
    with open(TRAINING_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            run = json.loads(line)
            nm = run.get('new_metrics')
            when = time.strftime('%m-%d %H:%M', time.localtime(run['ts']))

            if not nm:
                print('%-20s (no metrics recorded)' % when)
                continue

            delta = '' if prev_mse is None else '%+.2f' % (nm['weighted_mse'] - prev_mse)
            print('%-20s %8d %8d %10.4f %8s %9.3f %8.3f' % (
                when, run['train_rows'], run['val_rows'], nm['weighted_mse'],
                delta, nm['direction_cos'], nm['speed_mae']))
            prev_mse = nm['weighted_mse']

    if pinned:
        print('\nfixed eval session:', pinned)
    print('NOTE: runs before the eval session was pinned each held out a different '
          '"most recent" session, so early weighted_mse deltas above may reflect a '
          'harder/easier val set, not just a better/worse model.')


def buffer_stats():
    """Phase 5 sanity check - summarize what's actually landed in the rollout
    buffer, without needing the (not yet built) PPO consumer to exist. Reads
    only, never truncates or moves the file."""
    if not os.path.exists(ROLLOUT_BUFFER_PATH):
        print('zorbio_agent: no rollout buffer at', ROLLOUT_BUFFER_PATH, '(nothing served yet)')
        return

    episodes = {}
    n_rows = 0
    n_done = 0
    n_terminated = 0

    with open(ROLLOUT_BUFFER_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            eid = row['episode_id']
            episodes.setdefault(eid, {'steps': 0, 'reward_sum': 0.0})
            episodes[eid]['steps'] += 1
            episodes[eid]['reward_sum'] += row['reward']
            if row['done']:
                n_done += 1
            if row['terminated']:
                n_terminated += 1

    print('zorbio_agent: rollout buffer at', ROLLOUT_BUFFER_PATH)
    print('  %d rows across %d episodes (%d closed: %d terminated, %d truncated)' % (
        n_rows, len(episodes), n_done, n_terminated, n_done - n_terminated))

    if episodes:
        lengths = [e['steps'] for e in episodes.values()]
        rewards = [e['reward_sum'] for e in episodes.values()]
        print('  episode length: min=%d max=%d avg=%.1f' % (min(lengths), max(lengths), sum(lengths) / len(lengths)))
        print('  episode reward: min=%.3f max=%.3f avg=%.3f' % (min(rewards), max(rewards), sum(rewards) / len(rewards)))


def eval_checkpoint(pattern, checkpoint_path):
    if not HAVE_TORCH:
        print('zorbio_agent: torch is required for eval', file=sys.stderr)
        sys.exit(1)

    states, targets, meta = load_dataset(pattern)
    if not states:
        print('zorbio_agent: no examples found matching', pattern, file=sys.stderr)
        sys.exit(1)

    net = load_checkpoint(checkpoint_path)
    metrics = evaluate(net, states, targets, meta)
    print('zorbio_agent: %s on %s' % (checkpoint_path, pattern))
    print('  ', format_metrics(metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', metavar='JSONL_GLOB',
                         help='behavioral-clone a policy from logged session JSONL files '
                              'instead of running the inference server')
    parser.add_argument('--eval', metavar='JSONL_GLOB',
                         help='score an existing checkpoint (--checkpoint) against logged '
                              'session JSONL files without training')
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH,
                         help='checkpoint path for --eval (default: %s)' % CHECKPOINT_PATH)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval-every', type=int, default=5,
                         help='print val metrics every N epochs during --train (0 to disable)')
    parser.add_argument('--report', action='store_true',
                         help='print the improve/degrade trend across all logged training runs')
    parser.add_argument('--buffer-stats', action='store_true',
                         help='summarize the Phase 5 rollout buffer (episodes/rewards) without training on it')
    parser.add_argument('--pin-baseline', metavar='CHECKPOINT_PATH',
                         help='pin a checkpoint as the fixed Phase 7 comparison point for future eval matches')
    args = parser.parse_args()

    if args.report:
        report()
    elif args.buffer_stats:
        buffer_stats()
    elif args.pin_baseline:
        pin_baseline(args.pin_baseline)
    elif args.train:
        train(args.train, args.epochs, args.batch_size, args.lr, args.eval_every)
    elif args.eval:
        eval_checkpoint(args.eval, args.checkpoint)
    else:
        serve(load_policy())
