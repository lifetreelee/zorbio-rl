/**
 * Zorbio Game Client
 */

// ESLint global declarations: https://eslint.org/docs/rules/no-undef
/*
global config:true,
 ZOR:true,
 UTIL:true,
 gameStart:true
*/

const NODEJS_CLIENT = typeof module !== 'undefined' && module.exports;

if (NODEJS_CLIENT) {
    global._ = require('lodash');
    global.UTIL = require('./util.js');
    global.config = require('./config.js');
    global.WebSocket = require('ws');
    ZOR.Schemas = require('./schemas');
}

/**
 * Zorbio game client can be used in a browser or headless
 * @param {Object} handler
 * @constructor
 */
ZOR.ZORClient = function ZORClient(handler) {
    this.z_zorPingDuration = -1;
    this.z_actorUpdateGap = 0;
    this.z_handler = handler;
    this.z_rapidSendBuffer = new ArrayBuffer(20);
    this.z_rapidSendView = new Float32Array(this.z_rapidSendBuffer);
};

/**
 * Connect to the game server based on config
 * @param {string} uri
 */
ZOR.ZORClient.prototype.z_connectToServer = function ZORConnectToServer(uri) {
    let self = this;

    console.log('Client attempting to connect to ws server: ', uri);

    this.z_ws = new WebSocket( uri );
    this.z_ws.binaryType = 'arraybuffer';

    this.z_ws.onopen = function wsOpen() {
        console.log('WebSocket connection established and ready.');
        self.z_setupSocket(self.z_ws);
    };
};

/**
 * Send message to enter the game.  This is the function that sends all the initial player
 * Meta data, like name, skin, etc.
 * @param {Object} meta
 */
ZOR.ZORClient.prototype.z_sendEnterGame = function ZORSendEnterGame(meta) {
    if (this.z_ws.readyState === WebSocket.OPEN) {
        this.z_ws.send(JSON.stringify({
            op   : 'enter_game',
            type : meta.playerType,
            name : meta.playerName,
            color: meta.color,
            skin : meta.skin,
            key  : meta.key,
        }));

        console.log('Send enter game');
    }
};

/**
 * Sets up all the message and event handlers for the socket.
 * @param {Object} ws
 */
ZOR.ZORClient.prototype.z_setupSocket = function ZORSetupSocket(ws) {
    let self = this;

    ws.onmessage = function wsMessage(msg) {
        if (typeof msg.data === 'string') {
            let message = parseJson(msg.data);

            switch (message.op) {
                case 'game_setup':
                    handle_msg_game_setup();
                    break;
                case 'zor_pong':
                    handle_msg_zor_pong();
                    break;
                case 'player_join':
                    handle_msg_player_join(message);
                    break;
                case 'captured_player':
                    handle_msg_captured_player(message);
                    break;
                case 'player_died':
                    handle_msg_player_died(message);
                    break;
                case 'kick':
                    handle_msg_kick(message);
                    break;
                case 'remove_player':
                    handle_msg_remove_player(message);
                    break;
                case 'speeding_warning':
                    handle_msg_speeding_warning();
                    break;
                case 'speed_boost_res':
                    handle_msg_speed_boost_res(message);
                    break;
                case 'speed_boost_stop':
                    handle_msg_speed_boost_stop();
                    break;
                case 'admin_status':
                    handle_msg_admin_status(message);
                    break;
                case 'admin_debug_stats':
                    handle_msg_admin_debug_stats(message);
                    break;
                case 'player_type_changed':
                    handle_msg_player_type_changed(message);
                    break;
            }
        }
        else {
            let op = UTIL.readFirstByte(msg.data);

            switch (op) {
                case ZOR.Schemas.ops.INIT_GAME:
                    handle_msg_init_game(ZOR.Schemas.initGameSchema.decode(msg.data));
                    break;
                case ZOR.Schemas.ops.WELCOME:
                    handle_msg_welcome(ZOR.Schemas.welcomeSchema.decode(msg.data));
                    break;
                case ZOR.Schemas.ops.ACTOR_UPDATES:
                    handle_msg_actor_updates(ZOR.Schemas.actorUpdatesSchema.decode(msg.data));
                    break;
                case ZOR.Schemas.ops.TICK_SLOW:
                    handle_msg_server_tick_slow(ZOR.Schemas.tickSlowSchema.decode(msg.data));
                    break;
                case ZOR.Schemas.ops.YOU_DIED:
                    handle_msg_you_died(ZOR.Schemas.youDied.decode(msg.data));
                    break;
                case ZOR.Schemas.ops.LEADERBOARDS_UPDATE:
                    handle_msg_leaderboard_update(ZOR.Schemas.leaderboardUpdateSchema.decode(msg.data));
                    break;
                default: {
                    // // see if this is a player fast update
                    const msgView = new Float32Array(msg.data);
                    if (msgView[0] === ZOR.Schemas.ops.CLIENT_POSITION_RAPID) {
                        handle_msg_client_position_rapid(msgView);
                    }
                    else {
                        console.error('Error: Unknown binary op code: ', op);
                    }
                }
            }
        }
    };

    ws.onclose = function wsClose(e) {
        if (e.code !== config.CLOSE_NO_RESTART) {
            self.z_handleNetworkTermination();
        }
        console.log('Connection closed:', e.code, e.reason);
    };

    ws.onerror = function wsError(e) {
        console.error('Websocket error occured', e);
    };

    /**
     * Parses json
     * @param {string} msg
     * @returns {Object}
     */
    function parseJson(msg) {
        // put in own function so we can see how long this takes in the profiler
        return JSON.parse(msg);
    }

    /**
     * Handles the init game message from the server
     * @param {Object} msg
     */
    function handle_msg_init_game(msg) {
        // iterate over actors and create THREE objects that don't serialize over websockets
        msg.model.actors.forEach(function eachActor(actor) {
            UTIL.toVector3(actor, 'position');
            UTIL.toVector3(actor, 'velocity');
        });

        self.z_NB_SRVID = msg.NB_SRVID;  // Linode nodebalancer node id that handled this socket connection

        self.z_handler.z_handle_init_game(msg.model);
    }

    /**
     * Handles the welcome message from the server
     * @param {Object} msg
     */
    function handle_msg_welcome(msg) {
        console.log('Welcome: ', msg.player.name);

        self.z_playerModel = self.z_handler.z_handle_welcome(msg);

        ws.send(JSON.stringify({ op: 'player_ready' }));
    }

    /**
     * Handles the game setup message from the server signaling that the game is ready to start
     */
    function handle_msg_game_setup() {
        console.log('Game setup');
        self.z_handler.z_handle_game_setup();
        self.z_setIntervalMethods();
    }

    /**
     * Handle the player join message from the server
     * @param {Object} msg
     */
    function handle_msg_player_join(msg) {
        let newPlayer = msg.player;

        if (self.z_playerModel && (newPlayer.id === self.z_playerModel.id)) {
            return; // ignore own join message
        }

        // Initialize THREE objects
        UTIL.toVector3(newPlayer.sphere, 'position');
        UTIL.toVector3(newPlayer.sphere, 'velocity');

        self.z_handler.z_handle_player_join(newPlayer);
    }

    /**
     * Handle the admin status message from the server (sent on welcome for admins, and
     * after any admin_set_* command is processed).
     * @param {Object} msg
     */
    function handle_msg_admin_status(msg) {
        self.z_handler.z_handle_admin_status(msg);
    }

    /**
     * Handle the admin debug stats snapshot from the server (sent in response
     * to z_sendAdminDebugStatsRequest).
     * @param {Object} msg
     */
    function handle_msg_admin_debug_stats(msg) {
        self.z_handler.z_handle_admin_debug_stats(msg);
    }

    /**
     * Handle a player switching between PLAYER and SPECTATOR mid-session
     * (see z_sendSwitchPlayerType) - could be this client or any other
     * connected player.
     * @param {Object} msg
     */
    function handle_msg_player_type_changed(msg) {
        self.z_handler.z_handle_player_type_changed(msg);
    }

    /**
     * Handle the pong response from the server
     */
    function handle_msg_zor_pong() {
        self.z_zorPingDuration = Date.now() - self.z_zorPingStart;

        self.z_handler.z_handle_pong(self.z_zorPingDuration);
    }

    /**
     * Handle the client position rapid from the server.  This is a small binary message that just contains
     * the player_id, position, and scale.  That can be sent every frame.  Currently disabled.
     * @param {Float32Array} messageView
     */
    function handle_msg_client_position_rapid(messageView) {
        self.z_handler.z_handle_client_position_rapid(messageView);
    }

    /**
     * Handles the actor update message
     * @param {Object} msg
     */
    function handle_msg_actor_updates(msg) {
        if (self.z_playerModel) {
            // Record gap since last actor update was received
            let nowTime = Date.now();
            self.z_actorUpdateGap = nowTime - self.z_playerModel.au_receive_metric.last_time;
            self.z_playerModel.au_receive_metric.last_time = nowTime;
        }

        self.z_handler.z_handle_actor_updates(msg.actors);
    }

    /**
     * Handles the captured player message
     * @param {Object} msg
     */
    function handle_msg_captured_player(msg) {
        self.z_handler.z_handle_captured_player(msg.targetPlayerId);
    }

    /**
     * Handles the you died message
     * @param {Object} msg
     */
    function handle_msg_you_died(msg) {
        self.z_handler.z_handle_you_died(msg);
        self.z_clearIntervalMethods();
    }

    /**
     * Handles the leaderboard update message
     * @param {Object} msg
     */
    function handle_msg_leaderboard_update(msg) {
        self.z_handler.z_handle_leaderboard_update(msg);
    }

    /**
     * Handle the player died message.  This means other player died not current player.
     * @param {Object} msg
     */
    function handle_msg_player_died(msg) {
        let attackingPlayerId = msg.attackingPlayerId;
        let targetPlayerId = msg.targetPlayerId;

        if (!self.z_playerModel || (attackingPlayerId !== self.z_playerModel.id)) {
            // someone else killed another player
            self.z_handler.z_handle_player_died(targetPlayerId);
        }
    }

    /**
     * Handle server tick slow
     * @param {Object} msg
     */
    function handle_msg_server_tick_slow(msg) {
        if (!gameStart) return;

        self.z_handler.z_handle_server_tick(msg.tick_data);
    }

    /**
     * Handle kick message
     * @param {Object} msg
     */
    function handle_msg_kick(msg) {
        self.z_handler.z_handle_kick(msg.reason);
    }

    /**
     * Handle remove player message
     * @param {Object} msg
     */
    function handle_msg_remove_player(msg) {
        self.z_handler.z_handle_remove_player(msg.playerId);
    }

    /**
     * Handle speeding warning message
     */
    function handle_msg_speeding_warning() {
        console.log('WARNING! You are speeding!');
    }

    /**
     * Handle speed boost response message from the server.  Signaling that a speed boost request is valid and should
     * start.
     * @param {Object} msg
     */
    function handle_msg_speed_boost_res(msg) {
        self.z_handler.handle_speed_boost_res(msg.is_valid);
    }

    /**
     * Handle speed boost stop message. Server says stop speed boosting now!
     */
    function handle_msg_speed_boost_stop() {
        console.log('Received speed boost STOP');
        self.z_handler.z_handle_speed_boost_stop();
        self.z_sendSpeedBoostStop();
    }
};

ZOR.ZORClient.prototype.z_handleNetworkTermination = function ZORHandleNetworkTermination() {
    this.z_clearIntervalMethods();
    this.z_handler.z_handleNetworkTermination();
};

ZOR.ZORClient.prototype.z_sendRespawn = function ZORSendRespawn() {
    gameStart = false;
    this.z_ws.send(JSON.stringify({ op: 'respawn' }));
};

/**
 * Intentionally leave the game. Closes the socket with the app-reserved
 * CLOSE_NO_RESTART code so the client's own onclose handler doesn't treat
 * this like an unexpected network drop (see z_setupSocket's ws.onclose).
 */
ZOR.ZORClient.prototype.z_sendLeaveGame = function ZORSendLeaveGame() {
    if (this.z_ws && this.z_ws.readyState === WebSocket.OPEN) {
        this.z_ws.close(config.CLOSE_NO_RESTART, 'player left');
    }
};

ZOR.ZORClient.prototype.z_sendPing = function sendPing() {
    this.z_zorPingStart = Date.now();

    let fps = this.z_handler.z_handle_send_ping();

    this.z_ws.send(JSON.stringify({ op: 'zor_ping', lastPing: this.z_zorPingDuration, fps: fps }));
};


ZOR.ZORClient.prototype.z_setIntervalMethods = function ZORSetIntervalMethods() {
    let self = this;

    // start sending heartbeat
    self.z_interval_id_heartbeat = setInterval(function sendPing() {
        self.z_sendPing();
    }, config.HEARTBEAT_PULSE_INTERVAL);
};


ZOR.ZORClient.prototype.z_sendPlayerUpdate = function ZORSendPlayerUpdate(playerSphere, food_capture_queue) {
    // save metrics
    let nowTime = Date.now();
    let gap = nowTime - this.z_playerModel.pp_send_metric.last_time;
    this.z_playerModel.pp_send_metric.last_time = nowTime;
    let bufferedAmount = this.z_ws.bufferedAmount;

    // Send oldest position and most recent 4 positions
    let playerUpdateMessage = {
        0                 : ZOR.Schemas.ops.PLAYER_UPDATE,
        player_id         : this.z_playerModel.id,
        sphere_id         : playerSphere.id,
        pp_gap            : gap,
        au_gap            : this.z_actorUpdateGap,
        buffered_mount    : bufferedAmount,
        latest_position   : playerSphere.recentPositions[playerSphere.recentPositions.length - 1],
        prev_position_1   : playerSphere.recentPositions[playerSphere.recentPositions.length - 2],
        prev_position_2   : playerSphere.recentPositions[playerSphere.recentPositions.length - 3],
        prev_position_3   : playerSphere.recentPositions[playerSphere.recentPositions.length - 4],
        oldest_position   : playerSphere.recentPositions[0],
        food_capture_queue: food_capture_queue,
    };

    // Send player update data
    this.z_ws.send(ZOR.Schemas.playerUdateSchema.encode(playerUpdateMessage));
};

ZOR.ZORClient.prototype.z_sendClientPositionRapid = function ZORSendClientPositionRapid(actor_id, position) {
    if (!config.ENABLE_RAPID_UPDATES) return;

    // first byte op code
    this.z_rapidSendView[0] = ZOR.Schemas.ops.CLIENT_POSITION_RAPID;

    // actor id
    this.z_rapidSendView[1] = actor_id;

    // position
    this.z_rapidSendView[2] = position.x;
    this.z_rapidSendView[3] = position.y;
    this.z_rapidSendView[4] = position.z;

    this.z_ws.send(this.z_rapidSendBuffer);
};

ZOR.ZORClient.prototype.z_sendSpeedBoostStart = function ZORSendSpeedBoostStart() {
    this.z_ws.send(JSON.stringify({ op: 'speed_boost_start' }));
};

ZOR.ZORClient.prototype.z_sendSpeedBoostStop = function ZORSendSpeedBoostStop() {
    this.z_ws.send(JSON.stringify({ op: 'speed_boost_stop' }));
};

/**
 * Request the leaderboards from the server.
 */
ZOR.ZORClient.prototype.z_sendLeaderboardsRequest = function ZORSendLeaderboardsRequest() {
    let msg = {
        0: ZOR.Schemas.ops.LEADERBOARDS_REQUEST,
    };

    this.z_ws.send(ZOR.Schemas.leaderboardRequestSchema.encode(msg));
};

ZOR.ZORClient.prototype.z_clearIntervalMethods = function ZORClearIntervalMethods() {
    clearInterval(this.z_interval_id_heartbeat);
};

/**
 * Admin-only: toggle bot spawning on/off. Ignored server-side for non-admins.
 * @param {boolean} enabled
 */
ZOR.ZORClient.prototype.z_sendAdminSetBotsEnabled = function ZORSendAdminSetBotsEnabled(enabled) {
    this.z_ws.send(JSON.stringify({ op: 'admin_set_bots_enabled', value: !!enabled }));
};

/**
 * Admin-only: toggle whether large bots use the trained RL movement pattern.
 * Ignored server-side for non-admins.
 * @param {boolean} enabled
 */
ZOR.ZORClient.prototype.z_sendAdminSetRlEnabled = function ZORSendAdminSetRlEnabled(enabled) {
    this.z_ws.send(JSON.stringify({ op: 'admin_set_rl_enabled', value: !!enabled }));
};

/**
 * Admin-only: pause/unpause bot movement and all captures/drains server-wide.
 * Ignored server-side for non-admins.
 * @param {boolean} enabled
 */
ZOR.ZORClient.prototype.z_sendAdminSetPaused = function ZORSendAdminSetPaused(enabled) {
    this.z_ws.send(JSON.stringify({ op: 'admin_set_paused', value: !!enabled }));
};

/**
 * Admin-only: set the desired bot population. Ignored server-side for non-admins.
 * @param {number} value
 */
ZOR.ZORClient.prototype.z_sendAdminSetMaxBots = function ZORSendAdminSetMaxBots(value) {
    this.z_ws.send(JSON.stringify({ op: 'admin_set_max_bots', value: value }));
};

/**
 * Admin-only: set the size threshold above which bots use 'hunt'/'rl' instead
 * of a random curve. Ignored server-side for non-admins.
 * @param {number} value
 */
ZOR.ZORClient.prototype.z_sendAdminSetRlThreshold = function ZORSendAdminSetRlThreshold(value) {
    this.z_ws.send(JSON.stringify({ op: 'admin_set_rl_threshold', value: value }));
};

/**
 * Admin-only: request a fresh snapshot of every player/bot's food and player
 * capture counts. Ignored server-side for non-admins.
 */
ZOR.ZORClient.prototype.z_sendAdminDebugStatsRequest = function ZORSendAdminDebugStatsRequest() {
    this.z_ws.send(JSON.stringify({ op: 'admin_debug_stats_request' }));
};

/**
 * Admin-only: clear and respawn every bot at the current world settings.
 * No restart needed. Ignored server-side for non-admins.
 */
ZOR.ZORClient.prototype.z_sendAdminResetBots = function ZORSendAdminResetBots() {
    this.z_ws.send(JSON.stringify({ op: 'admin_reset_bots' }));
};

/**
 * Admin-only: toggle whether bots can capture each other, not just humans.
 * Ignored server-side for non-admins.
 * @param {boolean} enabled
 */
ZOR.ZORClient.prototype.z_sendAdminSetBotsCanEatBots = function ZORSendAdminSetBotsCanEatBots(enabled) {
    this.z_ws.send(JSON.stringify({ op: 'admin_set_bots_can_eat_bots', value: !!enabled }));
};

/**
 * Switches this connection between PLAYER and SPECTATOR without a
 * disconnect/rejoin. Available to anyone.
 * @param {string} newType ZOR.PlayerTypes.PLAYER or ZOR.PlayerTypes.SPECTATOR
 */
ZOR.ZORClient.prototype.z_sendSwitchPlayerType = function ZORSendSwitchPlayerType(newType) {
    this.z_ws.send(JSON.stringify({ op: 'switch_player_type', value: newType }));
};

/**
 * Admin-only: cap the size of newly-spawned bots. Ignored server-side for non-admins.
 * @param {number} value
 */
ZOR.ZORClient.prototype.z_sendAdminSetBotMaxSpawnScale = function ZORSendAdminSetBotMaxSpawnScale(value) {
    this.z_ws.send(JSON.stringify({ op: 'admin_set_bot_max_spawn_scale', value: value }));
};

/**
 * Admin-only: apply a new world size / food density. This rewrites
 * common/config.js on disk and restarts the server process (via
 * supervisor's file watch) to apply it - every connected client, including
 * this one, will see its socket close and auto-reload a moment later.
 * Ignored server-side for non-admins.
 * @param {number} worldSize
 * @param {number} foodDensity
 */
ZOR.ZORClient.prototype.z_sendAdminApplyWorldSettings = function ZORSendAdminApplyWorldSettings(worldSize, foodDensity) {
    this.z_ws.send(JSON.stringify({
        op          : 'admin_apply_world_settings',
        world_size  : worldSize,
        food_density: foodDensity,
    }));
};

if (NODEJS_CLIENT) module.exports = ZOR.ZORClient;
