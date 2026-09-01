#!/usr/bin/env python3
"""
Zorbio PPO trainer (Phase 5b of the real-RL scope).

Run as its own process, separate from zorbio_agent.py's inference server -
on purpose, so a PPO backward pass never competes with socket-serving
threads for the GIL and makes bots stutter mid-tick (see zorbio_agent.py's
ROLLOUT_BUFFER_PATH comment). It consumes zorbio_agent.py's rollout buffer
(complete (s, a, log_prob, r, s', done, terminated) transitions - see
append_transition there), runs a clipped-surrogate PPO update, and writes a
new checkpoint that the inference server picks up on its next
CHECKPOINT_POLL_SECONDS poll. No Node-side changes are needed for a new
checkpoint to take effect.

Usage:
    python3 ppo_train.py                 # one round, then exit
    python3 ppo_train.py --loop          # repeat forever, sleeping --interval
                                          # seconds between rounds
    python3 ppo_train.py --min-transitions 500   # lower the bar for testing
"""
import argparse
import json
import os
import shutil
import sys
import time

import zorbio_agent as za

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


CONSUMING_PATH = za.ROLLOUT_BUFFER_PATH + '.consuming'


def _restore_consuming_file():
    """Put rows from a leftover .consuming file back at the front of the
    live rollout buffer. Runs at the start of every round (not just after a
    bail-out) so a round that crashed or was killed between the rotate and
    the restore doesn't silently strand data forever - see the module
    docstring on why this file exists as a separate process at all, which is
    exactly the kind of boundary where a mid-flight kill is possible.

    The live inference process may have appended fresh rows to a
    newly-recreated buffer file while this was down - merges rather than
    overwrites, older (.consuming) rows first."""
    if not os.path.exists(CONSUMING_PATH):
        return

    print('zorbio_agent: found leftover', CONSUMING_PATH, '- restoring into the live buffer')
    tmp_path = za.ROLLOUT_BUFFER_PATH + '.tmp'
    with open(tmp_path, 'wb') as out:
        with open(CONSUMING_PATH, 'rb') as src:
            shutil.copyfileobj(src, out)
        if os.path.exists(za.ROLLOUT_BUFFER_PATH):
            with open(za.ROLLOUT_BUFFER_PATH, 'rb') as cur:
                shutil.copyfileobj(cur, out)
    os.replace(tmp_path, za.ROLLOUT_BUFFER_PATH)
    os.remove(CONSUMING_PATH)


def load_rollout_episodes():
    """Rotate the rollout buffer out from under the live inference process
    (rename, not delete - see _restore_consuming_file) and group its rows by
    episode_id, preserving each episode's own write order. Returns
    (episodes, consumed) where episodes is {episode_id: [row, ...]} and
    consumed is True if anything was actually rotated (False when there was
    no buffer file at all, e.g. no bots have run yet).

    Caller is responsible for either committing (delete CONSUMING_PATH once
    training succeeds) or restoring (_restore_consuming_file) - this
    function only rotates and reads, never deletes, so a bail-out never
    loses data.
    """
    _restore_consuming_file()

    if not os.path.exists(za.ROLLOUT_BUFFER_PATH):
        return {}, False

    os.rename(za.ROLLOUT_BUFFER_PATH, CONSUMING_PATH)

    episodes = {}
    with open(CONSUMING_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Only the last line of the file can ever be torn this way -
                # writes are append-only, so a mid-write kill (process kill,
                # power loss) can only truncate whatever line was in flight
                # at that moment. Skip it rather than losing every other row
                # already safely on disk.
                print('zorbio_agent: skipping truncated trailing line in', CONSUMING_PATH)
                continue
            episodes.setdefault(row['episode_id'], []).append(row)

    return episodes, True


def action_to_raw(action):
    """Invert TorchPolicy.act()'s speed remap to recover the raw Gaussian
    sample a stored action came from - dx/dy/dz pass through unchanged (only
    speed was ever remapped), boost is already 0/1 matching the Bernoulli
    sample space. Needed to recompute log_prob under the current net during
    the PPO update."""
    dx, dy, dz, speed, boost, _drain = action
    return [dx, dy, dz, (speed - 1.0) / 0.5], boost


def compute_gae_for_episode(values, next_values, rewards, dones, terminateds, gamma, lam):
    """Standard GAE-lambda backward recursion over one episode's transitions,
    in order. `values`/`next_values` are this net's current V(s)/V(s') for
    each transition; bootstrapping from next_value only happens when the
    episode didn't actually terminate (truncation, e.g. the step cap, still
    has a future worth estimating - real death doesn't)."""
    n = len(rewards)
    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0

    for t in reversed(range(n)):
        bootstrap = 0.0 if terminateds[t] else next_values[t]
        delta = rewards[t] + gamma * bootstrap - values[t]
        continue_mask = 0.0 if dones[t] else 1.0
        gae = delta + gamma * lam * continue_mask * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    return advantages, returns


def ppo_update(net, opt, transitions, advantages, returns, clip_ratio, epochs,
                minibatch_size, vf_coef, entropy_coef, max_grad_norm):
    states = torch.tensor([t['state'] for t in transitions], dtype=torch.float32)

    raw_cont, raw_boost = [], []
    for t in transitions:
        cont, boost = action_to_raw(t['action'])
        raw_cont.append(cont)
        raw_boost.append(boost)
    cont_actions  = torch.tensor(raw_cont, dtype=torch.float32)
    boost_actions = torch.tensor(raw_boost, dtype=torch.float32).unsqueeze(1)
    old_log_probs = torch.tensor([t['log_prob'] for t in transitions], dtype=torch.float32)

    advantages_t = torch.tensor(advantages, dtype=torch.float32)
    returns_t    = torch.tensor(returns, dtype=torch.float32)
    # standard PPO variance reduction - a single very lucky/unlucky episode
    # otherwise dominates the gradient
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    n = states.shape[0]
    stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'clip_frac': 0.0, 'steps': 0}

    for _epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]

            mean, log_std, binary_logits, value = net(states[idx])
            std = log_std.exp()
            cont_dist = torch.distributions.Normal(mean, std)
            bin_dist  = torch.distributions.Bernoulli(logits=binary_logits[:, 0:1])

            new_log_prob = (cont_dist.log_prob(cont_actions[idx]).sum(dim=1)
                             + bin_dist.log_prob(boost_actions[idx]).sum(dim=1))

            ratio = (new_log_prob - old_log_probs[idx]).exp()
            surr1 = ratio * advantages_t[idx]
            surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(value.squeeze(1), returns_t[idx])
            entropy = (cont_dist.entropy().sum(dim=1) + bin_dist.entropy().sum(dim=1)).mean()

            loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()

            with torch.no_grad():
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss']  += value_loss.item()
                stats['entropy']     += entropy.item()
                stats['clip_frac']   += ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
                stats['steps'] += 1

    for k in ('policy_loss', 'value_loss', 'entropy', 'clip_frac'):
        stats[k] /= max(1, stats['steps'])
    return stats


def train_ppo(gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, epochs=4, minibatch_size=256,
              lr=3e-4, vf_coef=0.5, entropy_coef=0.001, max_grad_norm=0.5, min_transitions=2000,
              weight_decay=1e-4):
    if not HAVE_TORCH:
        print('zorbio_agent: torch is required for PPO training '
              '(pip install torch)', file=sys.stderr)
        sys.exit(1)

    episodes, consumed = load_rollout_episodes()
    if not consumed:
        print('zorbio_agent: no rollout buffer yet - nothing to train on (start the '
              'inference server and enable RL bots first)')
        return

    # build per-episode transition lists directly from the already-complete
    # rows append_transition wrote - no adjacency reconstruction needed
    transitions_by_episode = [rows for rows in episodes.values() if rows]
    total = sum(len(rows) for rows in transitions_by_episode)

    if total < min_transitions:
        print('zorbio_agent: only %d transitions collected (need >= %d) - not enough '
              'fresh data for a PPO round yet, restoring buffer for next time' % (
                  total, min_transitions))
        _restore_consuming_file()
        return

    net = za.PolicyNet()
    if os.path.exists(za.CHECKPOINT_PATH):
        try:
            net.load_state_dict(torch.load(za.CHECKPOINT_PATH, map_location='cpu'))
            print('zorbio_agent: PPO starting from checkpoint at', za.CHECKPOINT_PATH)
        except RuntimeError as e:
            print('zorbio_agent: checkpoint at', za.CHECKPOINT_PATH,
                  'is incompatible with the current network architecture - '
                  'starting PPO from a fresh untrained net instead. (%s)' % e)
    else:
        print('zorbio_agent: no checkpoint found - starting PPO from a fresh untrained net')

    # AdamW (decoupled weight decay), not plain Adam - max_grad_norm clipping
    # alone bounds how much any single update can move the weights, but does
    # nothing about a small, consistent drift compounding over tens of
    # thousands of updates. That's what caused the tanh-saturation bug
    # (2026-08-28/29, see memory: rl-phase-progress): even after bounding the
    # one unbounded input feature that triggered it, the trunk's output norm
    # was still climbing back toward saturation within ~6 hours on
    # unregularized weights. Weight decay gives the optimizer an actual
    # pull-toward-zero counterforce instead of just a per-step speed limit.
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    all_advantages, all_returns, flat_transitions = [], [], []
    for rows in transitions_by_episode:
        states      = [r['state'] for r in rows]
        next_states = [r['next_state'] for r in rows]
        with torch.no_grad():
            _, _, _, values      = net(torch.tensor(states, dtype=torch.float32))
            _, _, _, next_values = net(torch.tensor(next_states, dtype=torch.float32))

        adv, ret = compute_gae_for_episode(
            values.squeeze(1).tolist(), next_values.squeeze(1).tolist(),
            [r['reward'] for r in rows], [r['done'] for r in rows], [r['terminated'] for r in rows],
            gamma, gae_lambda)

        all_advantages.extend(adv)
        all_returns.extend(ret)
        flat_transitions.extend(rows)

    print('zorbio_agent: PPO round on %d transitions from %d episodes '
          '(epochs=%d, minibatch=%d, lr=%g)' % (
              total, len(transitions_by_episode), epochs, minibatch_size, lr))

    stats = ppo_update(net, opt, flat_transitions, all_advantages, all_returns,
                        clip_ratio, epochs, minibatch_size, vf_coef, entropy_coef, max_grad_norm)

    print('zorbio_agent: policy_loss=%.4f value_loss=%.4f entropy=%.4f clip_frac=%.3f' % (
        stats['policy_loss'], stats['value_loss'], stats['entropy'], stats['clip_frac']))
    print('zorbio_agent: NOTE - a fresh value_head starts random-initialized, so the first '
          'round or two is mostly the critic catching up, not real policy improvement; '
          'don\'t read early policy_loss/clip_frac as signal.')

    za.backup_checkpoint()
    torch.save(net.state_dict(), za.CHECKPOINT_PATH)
    print('zorbio_agent: saved checkpoint to', za.CHECKPOINT_PATH,
          '- inference server will hot-reload it within', za.CHECKPOINT_POLL_SECONDS, 'seconds')

    # only now that the checkpoint is safely written do we discard the
    # rotated-out data - anything short-circuited above (torch missing,
    # not enough data) restored it instead
    os.remove(CONSUMING_PATH)

    with open(za.PPO_TRAINING_LOG_PATH, 'a') as f:
        f.write(json.dumps({
            'ts'            : time.time(),
            'n_transitions' : total,
            'n_episodes'    : len(transitions_by_episode),
            'gamma'         : gamma,
            'gae_lambda'    : gae_lambda,
            'clip_ratio'    : clip_ratio,
            'epochs'        : epochs,
            'policy_loss'   : stats['policy_loss'],
            'value_loss'    : stats['value_loss'],
            'entropy'       : stats['entropy'],
            'clip_frac'     : stats['clip_frac'],
        }) + '\n')
    print('zorbio_agent: appended run summary to', za.PPO_TRAINING_LOG_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loop', action='store_true',
                         help='run rounds forever, sleeping --interval seconds between each')
    parser.add_argument('--interval', type=float, default=300.0,
                         help='seconds to sleep between rounds in --loop mode (default: 300)')
    parser.add_argument('--min-transitions', type=int, default=2000,
                         help='minimum fresh transitions required to run a round (default: 2000)')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-ratio', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=4, help='PPO update epochs per round')
    parser.add_argument('--minibatch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.001)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                         help='AdamW weight decay - counters slow unbounded weight/trunk-output '
                              'growth over many updates that per-step grad-norm clipping alone '
                              'does not (see the tanh-saturation bug, memory: rl-phase-progress)')
    args = parser.parse_args()

    kwargs = dict(gamma=args.gamma, gae_lambda=args.gae_lambda, clip_ratio=args.clip_ratio,
                  epochs=args.epochs, minibatch_size=args.minibatch_size, lr=args.lr,
                  vf_coef=args.vf_coef, entropy_coef=args.entropy_coef,
                  max_grad_norm=args.max_grad_norm, min_transitions=args.min_transitions,
                  weight_decay=args.weight_decay)

    if args.loop:
        print('zorbio_agent: PPO loop starting - one round every', args.interval, 'seconds')
        while True:
            train_ppo(**kwargs)
            time.sleep(args.interval)
    else:
        train_ppo(**kwargs)
