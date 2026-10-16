"use strict";
// JSON-lines driver around the pinned Showdown simulator (dist/sim) for the W5 CF0
// counterfactual-feasibility spike. One request per stdin line, one response per line.
//
// Requests : {"cmd": ..., "id": ..., ...}   Responses: {"ok": bool, "events": [...], ...}
// Events are what the real server would see, in emission order:
//   {"t":"update","omni":[lines],"p1":[lines],"p2":[lines]}   (|split| extraction identical
//        to sim/battle.ts extractChannelMessages; p1/p2 are the per-player channels)
//   {"t":"sideupdate","side":"p1","data":"|request|{...}"}    (that side only)
//   {"t":"end","data":{...}}
// Nothing here patches third_party/: only public exports of the pinned build are used.
const path = require("path");
const readline = require("readline");

const SIM = process.env.SHOWDOWN_SIM_DIR ||
  path.resolve(__dirname, "../../../third_party/metamon/server/pokemon-showdown/dist/sim");
const { Battle, extractChannelMessages } = require(path.join(SIM, "battle"));

const battles = new Map(); // id -> { battle, out: [] }
const snapshots = new Map(); // key -> JSON string
let snapshotCounter = 0;

function makeSend(slot) {
  return (type, data) => {
    if (Array.isArray(data)) data = data.join("\n");
    if (type === "update") {
      const channels = extractChannelMessages(data, [-1, 1, 2]);
      slot.out.push({ t: "update", omni: channels[-1], p1: channels[1], p2: channels[2] });
    } else if (type === "sideupdate") {
      const nl = data.indexOf("\n");
      slot.out.push({ t: "sideupdate", side: data.slice(0, nl), data: data.slice(nl + 1) });
    } else if (type === "end") {
      slot.out.push({ t: "end", data: JSON.parse(data) });
    }
  };
}

function drain(slot) {
  const events = slot.out;
  slot.out = [];
  return events;
}

function get(id) {
  const slot = battles.get(id);
  if (!slot) throw new Error(`unknown battle id ${id}`);
  return slot;
}

// Normalise the only wall-clock-dependent protocol line (|t:|<unix seconds>).
function normalisedState(battle) {
  const state = JSON.parse(JSON.stringify(battle.toJSON()));
  state.log = state.log.map((line) => (line.startsWith("|t:|") ? "|t:|" : line));
  return state;
}

const handlers = {
  create(req) {
    const slot = { battle: null, out: [] };
    const battle = new Battle({
      formatid: req.formatid || "gen9ou",
      seed: req.seed,
      rated: "",
      send: makeSend(slot),
    });
    slot.battle = battle;
    battles.set(req.id, slot);
    // Same writes as the server (`>start`, `>player p1`, `>player p2`), each followed by
    // the flush BattleStream._write performs, so update chunking matches the server's.
    battle.sendUpdates();
    battle.setPlayer("p1", { name: req.p1.name, avatar: "170", team: req.p1.team });
    battle.sendUpdates();
    battle.setPlayer("p2", { name: req.p2.name, avatar: "2", team: req.p2.team });
    battle.sendUpdates();
    return { events: drain(slot) };
  },
  choose(req) {
    const slot = get(req.id);
    const accepted = slot.battle.choose(req.side, req.choice);
    // BattleStream._write flushes after every written line.
    slot.battle.sendUpdates();
    return { accepted, events: drain(slot) };
  },
  snapshot(req) {
    const slot = get(req.id);
    if (slot.out.length) throw new Error("snapshot with undelivered output");
    const key = `snap${++snapshotCounter}`;
    snapshots.set(key, JSON.stringify(slot.battle.toJSON()));
    return { key };
  },
  restore(req) {
    const json = snapshots.get(req.key);
    if (json === undefined) throw new Error(`unknown snapshot ${req.key}`);
    if (battles.has(req.id)) throw new Error(`battle id ${req.id} exists`);
    const slot = { battle: null, out: [] };
    // Fresh parse every time: restoring never aliases or mutates the stored snapshot.
    slot.battle = Battle.fromJSON(json);
    slot.battle.restart(makeSend(slot));
    battles.set(req.id, slot);
    return { events: [] };
  },
  // Replace the PRNG state without resetRNG (which writes a public log line).
  set_prng(req) {
    get(req.id).battle.prng.setSeed(req.seed);
    return {};
  },
  get_prng(req) {
    return { seed: get(req.id).battle.prng.getSeed() };
  },
  state(req) {
    return { state: normalisedState(get(req.id).battle) };
  },
  info(req) {
    const b = get(req.id).battle;
    return {
      turn: b.turn,
      ended: b.ended,
      winner: b.winner,
      requestState: b.requestState,
      requests: b.sides.map((s) => (s.activeRequest ? JSON.parse(JSON.stringify(s.activeRequest)) : null)),
    };
  },
  destroy(req) {
    battles.delete(req.id);
    return {};
  },
  drop_snapshot(req) {
    snapshots.delete(req.key);
    return {};
  },
};

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  if (!line.trim()) return;
  let response;
  try {
    const req = JSON.parse(line);
    const handler = handlers[req.cmd];
    if (!handler) throw new Error(`unknown cmd ${req.cmd}`);
    response = { ok: true, ...handler(req) };
  } catch (err) {
    response = { ok: false, error: String(err && err.stack ? err.stack : err) };
  }
  process.stdout.write(JSON.stringify(response) + "\n");
});
rl.on("close", () => process.exit(0));
