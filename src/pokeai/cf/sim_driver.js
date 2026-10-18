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
const { Pokemon } = require(path.join(SIM, "pokemon"));
const { TeamValidator, Dex, toID, Teams } = require(path.join(SIM, "index"));

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


// ---------------------------------------------------------------- CF1 belief support

// Public HP display exactly as Pokemon.getHealth().shared (HP Percentage Mod).
function publicPct(p) {
  if (!p.hp) return 0;
  let pct = Math.ceil((100 * p.hp) / p.maxhp);
  if (pct === 100 && p.hp < p.maxhp) pct = 99;
  return pct;
}

// Smallest-change inverse of publicPct for a new max HP.
function hpForPct(pct, maxhp) {
  if (pct === 0) return 0;
  if (pct >= 100) return maxhp;
  return Math.max(1, Math.floor((pct * maxhp) / 100));
}

function moveId(battle, name) {
  const id = battle.dex.moves.get(name).id;
  return id.startsWith("hiddenpower") ? "hiddenpower" : id;
}

function p2Roster(battle) {
  return battle.sides[1].pokemon.map((p, index) => ({
    index,
    name: p.name,
    species: Dex.species.get(p.baseSpecies.name.replace(/-Tera$/, "")).name,
    current_species: p.species.name,
    level: p.level,
    gender: p.gender,
    shiny: !!p.set.shiny,
    legal_abilities: Object.values(
      Dex.species.get(p.baseSpecies.name.replace(/-Tera$/, "")).abilities),
    is_active: p.isActive,
    fainted: !p.hp,
  }));
}

function applyWorld(battle, world) {
  const side = battle.sides[1];
  const hp_maps = [];
  for (const m of world.mons) {
    const p = side.pokemon[m.index];
    const oldMax = p.maxhp;
    const pct = publicPct(p);
    const set = {
      name: p.name, species: p.baseSpecies.name, gender: p.gender || "N",
      level: p.level, shiny: !!p.set.shiny, item: m.item || "", ability: m.ability,
      moves: m.moves, nature: m.nature, evs: m.evs, ivs: m.ivs,
      teraType: m.tera_type, happiness: p.set.happiness, pokeball: p.set.pokeball,
    };
    const fresh = new Pokemon(set, side);
    p.set = fresh.set;
    p.ppUps = fresh.ppUps;
    p.hpType = fresh.hpType; p.hpPower = fresh.hpPower;
    p.baseHpType = fresh.baseHpType; p.baseHpPower = fresh.baseHpPower;
    p.baseStoredStats = fresh.baseStoredStats;
    p.baseMaxhp = fresh.baseMaxhp;
    if (!p.transformed) {
      const stats = p.species.id === p.baseSpecies.id
        ? fresh.storedStats : battle.spreadModify(p.species.baseStats, p.set);
      for (const k of Object.keys(p.storedStats)) p.storedStats[k] = stats[k];
    }
    p.maxhp = fresh.baseMaxhp;
    p.hp = hpForPct(pct, p.maxhp);
    const oldBase = p.baseMoveSlots, oldCur = p.moveSlots;
    const ids = m.moves.map((n) => moveId(battle, n));
    const pick = (list, id, i) => {
      const hit = list.find((x) => x.id === id);
      return hit ? { ...hit } : { ...fresh.baseMoveSlots[i] };
    };
    p.baseMoveSlots = ids.map((id, i) => pick(oldBase, id, i));
    p.moveSlots = ids.map((id, i) => pick(oldCur, id, i));
    if (!m.keep_item) {
      p.item = toID(m.item);
      p.itemState = battle.initEffectState({ id: p.item, target: p });
    }
    if (!m.keep_ability) {
      p.baseAbility = toID(m.ability);
      p.ability = p.baseAbility;
      p.abilityState = battle.initEffectState({ id: p.ability, target: p });
    }
    if (!m.keep_tera) p.teraType = m.tera_type;
    if (p.isActive) p.updateSpeed();
    hp_maps.push({ index: m.index, old_max: oldMax, new_max: p.maxhp, pct, hp: p.hp });
  }
  const reqs = battle.getRequests(battle.requestState);
  side.activeRequest = reqs[1];
  return { request: JSON.stringify(reqs[1]), hp_maps };
}

function publicView(battle) {
  const sideView = (side) => ({
    name: side.name,
    conditions: Object.keys(side.sideConditions).sort(),
    pokemon: side.pokemon.map((p) => ({
      name: p.name, species: p.species.name, base_species: p.baseSpecies.name,
      level: p.level, gender: p.gender, shiny: !!p.set.shiny,
      hp_public: p.getHealth().shared, fainted: !p.hp, status: p.status,
      boosts: p.boosts, volatiles: Object.keys(p.volatiles).sort(),
      is_active: p.isActive, position: p.position, terastallized: p.terastallized || "",
      types: p.getTypes ? p.getTypes() : p.types,
    })),
  });
  return {
    turn: battle.turn,
    request_state: battle.requestState,
    field: {
      weather: battle.field.weather, terrain: battle.field.terrain,
      pseudo: Object.keys(battle.field.pseudoWeather).sort(),
    },
    p1: sideView(battle.sides[0]), p2: sideView(battle.sides[1]),
  };
}

function p2Full(battle) {
  return battle.sides[1].pokemon.map((p, index) => ({
    index, name: p.name, species: p.baseSpecies.name, item: p.item, ability: p.ability,
    base_ability: p.baseAbility, moves: p.baseMoveSlots.map((x) => x.id),
    current_moves: p.moveSlots.map((x) => x.id),
    pp: p.moveSlots.map((x) => x.pp), nature: p.set.nature, evs: p.set.evs,
    ivs: p.set.ivs, tera_type: p.teraType, terastallized: p.terastallized || "",
    hp: p.hp, maxhp: p.maxhp, stats: p.storedStats,
    set: { name: p.name, species: p.baseSpecies.name, gender: p.gender || "N",
      level: p.level, shiny: !!p.set.shiny, item: p.set.item, ability: p.set.ability,
      moves: p.set.moves, nature: p.set.nature, evs: p.set.evs, ivs: p.set.ivs,
      teraType: p.set.teraType },
  }));
}

Object.assign(handlers, {
  roster(req) { return { roster: p2Roster(get(req.id).battle) }; },
  public_log(req) {
    const b = get(req.id).battle;
    return { lines: extractChannelMessages(b.log.join("\n"), [1])[1] };
  },
  stats() {
    const m = process.memoryUsage();
    return { battles: battles.size, snapshots: snapshots.size, rss: m.rss, heap: m.heapUsed };
  },
  public_view(req) { return { view: publicView(get(req.id).battle) }; },
  p2_full(req) { return { mons: p2Full(get(req.id).battle) }; },
  apply_world(req) { return applyWorld(get(req.id).battle, req.world); },
  validate(req) {
    const validator = TeamValidator.get(req.formatid || "gen9ou");
    return { errors: validator.validateTeam(JSON.parse(JSON.stringify(req.team))) };
  },
  move_info(req) {
    return { moves: req.names.map((n) => {
      const m = Dex.moves.get(n);
      return { name: m.name, id: m.id, category: m.category };
    }) };
  },
});


// ------------------------------------------------------------ BS0 reconstruction
// Builds a NEW battle at a decision point from a public-information spec (see
// pokeai/cf/reconstruct.py). Nothing here reads another battle's state.
const BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

function freshBoosts() {
  return { atk: 0, def: 0, spa: 0, spd: 0, spe: 0, accuracy: 0, evasion: 0 };
}

function setMonState(battle, side, p, m) {
  // reset everything the lead switch-in / default construction may have touched
  p.boosts = freshBoosts();
  p.volatiles = {};
  p.status = "";
  p.statusState = battle.initEffectState({});
  p.switchFlag = false;
  p.forceSwitchFlag = false;
  p.lastMove = null;
  p.lastMoveUsed = null;
  p.moveThisTurn = "";
  p.moveLastTurnResult = undefined;
  p.moveThisTurnResult = undefined;
  p.newlySwitched = false;
  p.attackedBy = [];
  p.hurtThisTurn = null;
  p.usedItemThisTurn = false;
  p.statsRaisedThisTurn = false;
  p.statsLoweredThisTurn = false;
  p.beingCalledBack = false;
  p.trapped = p.maybeTrapped = p.maybeDisabled = p.maybeLocked = false;
  p.timesAttacked = m.times_attacked || 0;
  if (m.ability !== undefined && m.ability !== null) {
    p.ability = toID(m.ability);
    p.abilityState = battle.initEffectState({ id: p.ability, target: p, ...(m.ability_state || {}) });
  }
  if (m.types && m.types.length) p.setType(m.types);
  if (m.added_type) p.addedType = m.added_type;
  if (m.item_exact !== undefined && toID(m.item_exact) !== p.item) {
    p.lastItem = p.item;
    p.item = toID(m.item_exact);
    p.itemState = battle.initEffectState({ id: p.item, target: p });
  }
  if (m.item_gone && m.item_exact === undefined) {
    p.lastItem = p.item;
    p.item = "";
    p.itemState = battle.initEffectState({ id: "", target: p });
    if (m.item_knocked) p.itemKnockedOff = true;
  }
  for (const k of Object.keys(m.boosts || {})) p.boosts[k] = m.boosts[k];
  for (const slot of p.baseMoveSlots) {
    const key = slot.id;
    if (m.pp_exact && m.pp_exact[key] !== undefined) slot.pp = m.pp_exact[key];
    else if (m.pp_used && m.pp_used[key]) slot.pp = Math.max(0, slot.maxpp - m.pp_used[key]);
  }
  p.moveSlots = p.baseMoveSlots.slice();
  p.activeTurns = m.active_turns || 0;
  p.activeMoveActions = m.active_move_actions || 0;
  if (m.last_move) {
    const mv = battle.dex.getActiveMove(m.last_move);
    p.lastMove = mv;
    p.lastMoveUsed = mv;
    p.lastMoveTargetLoc = 1;
  }
  if (m.forme && toID(m.forme) !== p.species.id && !m.terastallized) {
    p.formeChange(toID(m.forme), null, true);
    battle.log.length = battle.log.length;
  }
  if (m.hp_exact !== undefined && m.hp_exact !== null) p.hp = m.hp_exact;
  else p.hp = hpForPct(m.pct, p.maxhp);
  for (let i = 0; i < (m.regen || 0); i++) p.hp = Math.min(p.maxhp, p.hp + Math.floor(p.maxhp / 3));
  if (m.fainted) {
    p.hp = 0;
    p.fainted = true;
    p.isActive = false;
    p.isStarted = false;
    delete p.terastallized;
    return;
  }
  if (m.status) {
    p.status = toID(m.status);
    p.statusState = battle.initEffectState({ id: p.status, target: p });
    if (p.status === "slp") {
      p.statusState.startTime = m.status_state.startTime;
      p.statusState.time = m.status_state.time;
    } else if (p.status === "tox") {
      p.statusState.stage = m.status_state.stage;
    }
  }
  for (const v of m.volatiles || []) {
    const st = { ...v.state };
    if (st.hp_quarter) { delete st.hp_quarter; st.hp = Math.floor(p.maxhp / 4); }
    if (st.source_side) {
      const foe = battle.sides[st.source_side === "p1" ? 0 : 1];
      st.sourceSlot = `${st.source_side}a`;
      st.source = foe.pokemon[0];
      delete st.source_side;
      if (v.id === "partiallytrapped") {
        st.boundDivisor = st.source.hasItem("bindingband") ? 6 : 8;
        st.sourceEffect = battle.dex.moves.get(st.move);
      }
    }
    p.volatiles[v.id] = battle.initEffectState({ id: v.id, target: p, ...st });
  }
  if (m.terastallized) {
    p.teraType = m.terastallized;
    const log = battle.log.length;
    battle.actions.terastallize(p);
    battle.log.length = log;
    // tera-triggered abilities (Embody Aspect, Tera Shell...) must not re-fire
    p.boosts = freshBoosts();
    for (const k of Object.keys(m.boosts || {})) p.boosts[k] = m.boosts[k];
  }
}

function refreshRequestFlags(battle) {
  // the per-turn block of Battle.endTurn that feeds the move request (no turn++/residuals)
  for (const side of battle.sides) {
    for (const pokemon of side.active) {
      if (!pokemon || pokemon.fainted) continue;
      pokemon.maybeDisabled = false;
      pokemon.maybeLocked = false;
      for (const moveSlot of pokemon.moveSlots) {
        moveSlot.disabled = false;
        moveSlot.disabledSource = "";
      }
      battle.runEvent("DisableMove", pokemon);
      for (const moveSlot of pokemon.moveSlots) {
        const activeMove = battle.dex.getActiveMove(moveSlot.id);
        battle.singleEvent("DisableMove", activeMove, null, pokemon);
        if (activeMove.flags["cantusetwice"] && pokemon.lastMove?.id === moveSlot.id) {
          pokemon.disableMove(pokemon.lastMove.id);
        }
      }
      pokemon.trapped = pokemon.maybeTrapped = false;
      battle.runEvent("TrapPokemon", pokemon);
      if (!pokemon.knownType || battle.dex.getImmunity("trapped", pokemon)) {
        battle.runEvent("MaybeTrapPokemon", pokemon);
      }
      if (battle.gen > 2) {
        for (const source of pokemon.foes()) {
          const species = (source.illusion || source).species;
          if (!species.abilities) continue;
          for (const abilitySlot in species.abilities) {
            const abilityName = species.abilities[abilitySlot];
            if (abilityName === source.ability) continue;
            if (abilitySlot === "H" && species.unreleasedHidden) continue;
            const ability = battle.dex.abilities.get(abilityName);
            if (battle.ruleTable.has("-ability:" + ability.id)) continue;
            if (pokemon.knownType && !battle.dex.getImmunity("trapped", pokemon)) continue;
            battle.singleEvent("FoeMaybeTrapPokemon", ability, {}, pokemon, source);
          }
        }
      }
    }
  }
}

const reconHandlers = {
  reconstruct(req) {
    const slot = { battle: null, out: [] };
    const battle = new Battle({
      formatid: req.formatid || "gen9ou", seed: req.seed, rated: "", send: makeSend(slot),
    });
    slot.battle = battle;
    battles.set(req.id, slot);
    const own = Teams.unpack(req.p1.team);
    const byName = new Map(own.map((s) => [
      (!s.name || toID(s.name) === toID(s.species)) ? Dex.species.get(s.species).baseSpecies : s.name, s]));
    const ordered = req.p1.order.map((n) => {
      const s = byName.get(n);
      if (!s) throw new Error(`own team has no ${n}`);
      return s;
    });
    ordered.forEach((s, i) => { s.gender = req.p1.genders[i]; });
    // Inert-lead bootstrap: the real start-of-battle runs only to put both leads in play. Their
    // item and ability are blanked during it so no entry effect (Intimidate, Eject Pack,
    // terrain/weather setters, seeds, Imposter...) can fire or consume anything, and restored
    // from the sampled/own sets before the public state is patched. Leads are team slot 0.
    const leadSets = { p1: ordered[0], p2: req.p2.team[0] };
    const inert = (s) => ({ ...s, item: "", ability: "" });
    ordered[0] = inert(ordered[0]);
    const p2Team = req.p2.team.map((s, i) => (i === 0 ? inert(s) : s));
    battle.sendUpdates();
    battle.setPlayer("p1", { name: req.p1.name, avatar: "170", team: ordered });
    battle.sendUpdates();
    battle.setPlayer("p2", { name: req.p2.name, avatar: "2", team: p2Team });
    battle.sendUpdates();
    battle.choose("p1", "team 123456");
    battle.choose("p2", "team 123456");
    battle.sendUpdates();
    drain(slot);
    battle.sides.forEach((side, si) => {
      const p = side.pokemon[0];
      const set = leadSets[si === 0 ? "p1" : "p2"];
      p.set.item = set.item;
      p.set.ability = set.ability;
      p.baseAbility = toID(set.ability);
      p.ability = p.baseAbility;
      p.abilityState = battle.initEffectState({ id: p.ability, target: p });
      p.item = toID(set.item);
      p.itemState = battle.initEffectState({ id: p.item, target: p });
      // once-per-battle entry boosts the lead has already received (its boost is public state)
      if (p.ability === "dauntlessshield") p.shieldBoost = true;
      if (p.ability === "intrepidsword") p.swordBoost = true;
    });
    // ---- patch the public/sampled state
    battle.turn = req.turn;
    battle.field.weather = "";
    battle.field.weatherState = battle.initEffectState({});
    battle.field.terrain = "";
    battle.field.terrainState = battle.initEffectState({});
    battle.field.pseudoWeather = {};
    for (const side of battle.sides) {
      side.sideConditions = {};
      side.slotConditions = [{}];
    }
    for (let si = 0; si < 2; si++) {
      const side = battle.sides[si];
      const mons = req.mons[si === 0 ? "p1" : "p2"];
      mons.forEach((m, i) => setMonState(battle, side, side.pokemon[i], m));
      const sd = req.sides[si === 0 ? "p1" : "p2"];
      if (sd.tera_used) side.pokemon.forEach((q) => { q.canTerastallize = null; });
      side.totalFainted = sd.total_fainted;
      side.pokemonLeft = side.pokemon.filter((p) => !p.fainted).length;
      const forcedHere = req.request_state === "switch" && req.switch_sides.includes(side.id);
      side.faintedLastTurn = sd.fainted_last_turn
        ? (side.pokemon.find((p, i) => p.fainted && !(forcedHere && i === 0)) || null) : null;
      for (const [id, st] of Object.entries(sd.conditions)) {
        side.sideConditions[id] = battle.initEffectState({ id, target: side, ...st });
      }
    }
    const refPoke = (r) => (r ? battle.sides[r.side === "p1" ? 0 : 1].pokemon[r.index] : null);
    const f = req.field;
    if (f.weather) {
      battle.field.weather = f.weather.id;
      battle.field.weatherState = battle.initEffectState({
        id: f.weather.id, target: battle.field, source: refPoke(f.weather.source),
        sourceSlot: f.weather.source ? `${f.weather.source.side}a` : undefined,
        duration: f.weather.duration,
      });
    }
    if (f.terrain) {
      battle.field.terrain = f.terrain.id;
      battle.field.terrainState = battle.initEffectState({
        id: f.terrain.id, target: battle.field, source: refPoke(f.terrain.source),
        sourceSlot: f.terrain.source ? `${f.terrain.source.side}a` : undefined,
        duration: f.terrain.duration,
      });
    }
    for (const [id, st] of Object.entries(f.pseudo || {})) {
      battle.field.pseudoWeather[id] = battle.initEffectState({
        id, target: battle.field, source: refPoke(st.source), sourceSlot: st.source ? `${st.source.side}a` : undefined,
        duration: st.duration,
      });
    }
    battle.updateSpeed();
    refreshRequestFlags(battle);
    battle.prng.setSeed(req.prng);
    if (req.rqid !== undefined) battle.rqid = req.rqid - 1;
    battle.requestState = "";
    if (req.request_state === "switch") {
      for (const sid of req.switch_sides) {
        const side = battle.sides[sid === "p1" ? 0 : 1];
        const p = side.pokemon[0];
        p.switchFlag = true;
        if (p.fainted) p.status = "fnt"; // fainted-this-turn mons keep fnt until replaced
        side.faintedThisTurn = p;
      }
      // Engine boundary of a forced replacement (verified against a genuine battle): the turn's
      // residual already ran, the queue is empty and the turn is still open (midTurn = true), so
      // the next choice runs the switch and Battle.endTurn, with no second residual.
      battle.midTurn = true;
      battle.makeRequest("switch");
    } else {
      battle.makeRequest("move");
    }
    battle.log.push("|recon|"); // sendUpdates only emits requests when the log advanced
    battle.sendUpdates();
    return { events: drain(slot) };
  },
  // Engine facts that define a decision boundary (checked by pokeai.cf.recon_world).
  invariants(req) {
    const b = get(req.id).battle;
    return {
      invariants: {
        turn: b.turn, midTurn: b.midTurn, ended: b.ended, requestState: b.requestState,
        queue: b.queue.list.length, faintQueue: b.faintQueue.length,
        sides: b.sides.map((s) => ({
          id: s.id,
          request: s.activeRequest ? Object.keys(s.activeRequest).filter(
            (k) => ["wait", "forceSwitch", "active", "teamPreview"].includes(k)) : null,
          rqid: s.activeRequest ? s.activeRequest.rqid : null,
          active: s.active.map((p) => (p ? { fainted: p.fainted, hp: p.hp, switchFlag: p.switchFlag, isActive: p.isActive } : null)),
          switchFlags: s.pokemon.filter((p) => p.switchFlag).length,
          activeFainted: s.active.some((p) => p && p.fainted),
          pokemonLeft: s.pokemonLeft, aliveCount: s.pokemon.filter((p) => !p.fainted).length,
          faintedThisTurn: !!s.faintedThisTurn, choiceError: s.choice.error,
        })),
      },
    };
  },
};
Object.assign(handlers, reconHandlers);

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
