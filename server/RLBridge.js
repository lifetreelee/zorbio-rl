/**
 * RLBridge maintains a single persistent Unix socket connection to the Python
 * RL sidecar (server/rl/zorbio_agent.py). Bots using the 'rl' movement pattern
 * share this one connection and request action vectors for their current
 * state vector.
 *
 * Wire protocol v3 (see server/rl/zorbio_agent.py for the authoritative spec -
 * keep PROTOCOL_VERSION/STATE_DIM/ACTION_DIM in sync with those):
 *   request:  1  uint8   -- protocol version
 *             1  uint32  -- episode_id (unique per bot per episode)
 *             1  float32 -- prev_reward (reward for the transition into this tick)
 *             1  uint8   -- done (1 on an episode's final message)
 *             1  uint8   -- terminated (only meaningful if done: 1=real death, 0=truncated)
 *             1  uint8   -- policy_id (0=live training policy; nonzero = a
 *                           frozen self-play opponent-pool slot)
 *             1  uint32  -- request_id (monotonic per episode, identifies the
 *                           action this request is asking for)
 *             1  uint32  -- caused_by_action_id (request_id of whichever
 *                           earlier action is actually responsible for
 *                           prev_reward - see Bot.js's appliedActionId. 0 =
 *                           no action applied yet this episode)
 *             RLState.STATE_DIM floats -- state vector
 *   response: RLState.ACTION_DIM floats -- [dx, dy, dz, speedMultiplier, useBoost, useDrain]
 *             (unchanged by v3 - still just "here's an action", everything
 *             else only ever flows Node -> Python)
 *
 * v3 note: v2 paired each buffer row's reward with the *previous* row's
 * action by plain adjacency, which assumed a fixed one-tick request/response
 * lag. That assumption doesn't hold - responses arrive asynchronously (see
 * requestAction below), and reward-producing captures are resolved on a
 * separate, slower game tick than the one that fires these requests (see
 * AppServer.js's slow-tick checkBotFoodCaptures), so the real lag varies.
 * request_id/caused_by_action_id let the sidecar pair reward with the exact
 * action that earned it regardless of how many ticks that took, instead of
 * guessing a fixed offset.
 *
 * Requests are answered strictly in order (FIFO) since it's one ordered
 * stream socket, so responses are matched to the pending queue by position.
 * This matters more than it used to: an episode's final message (done=true,
 * sent out-of-band by Bot.js when a bot dies or is removed, not as part of
 * its normal per-tick request) still gets a reply that must be consumed even
 * though it's thrown away - skip that and every bot sharing this connection
 * silently starts receiving the wrong action from that point on.
 */
let net     = require('net');
let RLState = require('../common/RLState.js');

let PROTOCOL_VERSION = 3;
let STATE_FLOATS  = RLState.STATE_DIM;
let ACTION_FLOATS = RLState.ACTION_DIM;
let STATE_BYTES   = STATE_FLOATS * 4;
let ACTION_BYTES  = ACTION_FLOATS * 4;
// version, episode_id, prev_reward, done, terminated, policy_id, request_id, caused_by_action_id
let HEADER_BYTES  = 1 + 4 + 4 + 1 + 1 + 1 + 4 + 4;
let REQUEST_BYTES = HEADER_BYTES + STATE_BYTES;

let SOCKET_PATH = process.env.ZORBIO_RL_SOCKET || '/tmp/zorbio_rl.sock';
let RECONNECT_DELAY = 2000; // ms

// Responses are matched to pending callbacks purely by FIFO arrival order on
// one shared stream socket (see the class docstring) - if the sidecar ever
// stops responding to a single request for ANY reason (a hang inside model
// inference, a stuck lock, a wedged thread after a prior abrupt disconnect -
// this has happened twice with no error logged on either side and no known
// root cause), every bot sharing this connection silently stalls forever,
// since nothing ever un-jams the queue on its own. This has cost anywhere
// from 20 minutes to 10 hours of dead training time. WATCHDOG_INTERVAL_MS
// periodically checks how long the oldest pending request has been waiting;
// past REQUEST_TIMEOUT_MS, the socket is forcibly destroyed and reconnected
// - Python's accept loop runs each connection in its own thread (see
// zorbio_agent.py's serve()), so a stuck handler thread doesn't block a
// fresh connection from being accepted right behind it.
let REQUEST_TIMEOUT_MS   = 5000;
let WATCHDOG_INTERVAL_MS = 1000;

let RLBridge = function() {
    let self = this;

    self.connected    = false;
    self.socket       = null;
    self.pending      = [];            // FIFO queue of pending callbacks
    self.recvBuffer   = Buffer.alloc(0);

    self.connect = function rlBridgeConnect() {
        self.socket = net.createConnection(SOCKET_PATH);

        self.socket.on('connect', function rlBridgeOnConnect() {
            self.connected = true;
            console.log('RLBridge: connected to Python sidecar at', SOCKET_PATH);
        });

        self.socket.on('data', function rlBridgeOnData(chunk) {
            self.recvBuffer = Buffer.concat([self.recvBuffer, chunk]);

            while (self.recvBuffer.length >= ACTION_BYTES) {
                let actionBuf = self.recvBuffer.slice(0, ACTION_BYTES);
                self.recvBuffer = self.recvBuffer.slice(ACTION_BYTES);

                let action = [];
                for (let i = 0; i < ACTION_FLOATS; i++) {
                    action.push(actionBuf.readFloatLE(i * 4));
                }

                let waiter = self.pending.shift();
                if (waiter) waiter.callback(null, action);
            }
        });

        self.socket.on('error', function rlBridgeOnError(err) {
            if (self.connected) {
                console.log('RLBridge: socket error:', err.message);
            }
        });

        self.socket.on('close', function rlBridgeOnClose() {
            self.connected = false;
            self.recvBuffer = Buffer.alloc(0);

            // fail out anything still waiting for a response
            self.pending.forEach(function(waiter) {
                waiter.callback(new Error('RLBridge disconnected'));
            });
            self.pending = [];

            setTimeout(self.connect, RECONNECT_DELAY);
        });
    };

    // See the WATCHDOG_INTERVAL_MS comment above - this is what actually
    // recovers from a wedged sidecar instead of hanging until something else
    // (previously: the 20-minute population reset, by coincidence) happens
    // to clear the queue.
    setInterval(function rlBridgeWatchdog() {
        if (!self.connected || self.pending.length === 0) return;

        let oldest = self.pending[0];
        if (Date.now() - oldest.sentAt < REQUEST_TIMEOUT_MS) return;

        console.log('RLBridge: oldest pending request has waited', Date.now() - oldest.sentAt,
            'ms with no response (', self.pending.length, 'requests queued) - assuming the sidecar',
            'is wedged, forcing a reconnect');
        self.socket.destroy(); // triggers the 'close' handler above: fails out self.pending, reconnects
    }, WATCHDOG_INTERVAL_MS);

    /**
     * Send one RL step to the sidecar and get an action vector back
     * asynchronously via callback(err, [dx, dy, dz, speedMultiplier, useBoost, useDrain]).
     *
     * @param {number}  episodeId  unique per bot per episode (see Bot.js)
     * @param {number}  prevReward reward for the transition into this tick
     * @param {boolean} done       true only on an episode's final message
     * @param {boolean} terminated only meaningful when done: real death vs. truncation
     * @param {number}  policyId   0 = live training policy (reserved for Phase 6's pool)
     * @param {number[]} stateArray
     * @param {number}  requestId  monotonic per episode - identifies the action being requested
     * @param {number}  causedByActionId  requestId of the action actually responsible
     *                  for prevReward (0 = none applied yet this episode)
     *
     * Returns true if the request was sent, false immediately if there's no
     * live connection (caller should fall back to non-RL movement).
     */
    self.requestAction = function rlBridgeRequestAction(episodeId, prevReward, done, terminated, policyId,
                                                          stateArray, requestId, causedByActionId, callback) {
        if (!self.connected || !self.socket) return false;

        let buf = Buffer.alloc(REQUEST_BYTES);
        let offset = 0;

        buf.writeUInt8(PROTOCOL_VERSION, offset); offset += 1;
        buf.writeUInt32LE(episodeId >>> 0, offset); offset += 4;
        buf.writeFloatLE(prevReward || 0, offset); offset += 4;
        buf.writeUInt8(done ? 1 : 0, offset); offset += 1;
        buf.writeUInt8(terminated ? 1 : 0, offset); offset += 1;
        buf.writeUInt8(policyId || 0, offset); offset += 1;
        buf.writeUInt32LE(requestId >>> 0, offset); offset += 4;
        buf.writeUInt32LE(causedByActionId >>> 0, offset); offset += 4;

        for (let i = 0; i < STATE_FLOATS; i++) {
            buf.writeFloatLE(stateArray[i] || 0, offset);
            offset += 4;
        }

        self.pending.push({ callback: callback, sentAt: Date.now() });
        self.socket.write(buf);
        return true;
    };

    self.connect();
};

// Singleton - all rl bots share one connection to the sidecar
module.exports = new RLBridge();
