from dataclasses import replace

import pytest
from metamon.backend.showdown_dex.dex import Dex
from metamon.interface import UniversalMove, UniversalPokemon, UniversalState

import pokeai  # noqa: F401  (import order: metamon.baselines first)
from pokeai.data.matchup import (
    BENCH_F,
    EXTRAS_DIM,
    MOVE_F,
    OPP_MOVE_F,
    PAD,
    THREAT_F,
    damage_frac,
    matchup_extras,
    p_immune_ability,
    p_ko,
    type_mult,
)


def mon(species, moves=(), ability="unknownability", item="unknownitem", hp=1.0, **kw):
    e = Dex.from_gen(9).pokedex[species]
    s = e["baseStats"]
    dex_moves = Dex.from_gen(9).moves
    ums = []
    for m in moves:
        d = dex_moves[m]
        ums.append(
            UniversalMove(
                name=m,
                move_type=d["type"].lower(),
                category=d["category"].lower(),
                base_power=d["basePower"],
                accuracy=1.0 if d["accuracy"] is True else d["accuracy"] / 100,
                priority=d["priority"],
                current_pp=8,
                max_pp=8,
            )
        )
    p = UniversalPokemon(
        name=species, hp_pct=hp, types=" ".join(t.lower() for t in e["types"]),
        item=item, ability=ability, lvl=100, status="nostatus", effect="noeffect",
        moves=ums, atk_boost=0, spa_boost=0, def_boost=0, spd_boost=0, spe_boost=0,
        accuracy_boost=0, evasion_boost=0, base_atk=s["atk"], base_spa=s["spa"],
        base_def=s["def"], base_spd=s["spd"], base_spe=s["spe"], base_hp=s["hp"],
        tera_type="notype", base_species=species,
    )
    return replace(p, **kw)


def state(me, opp, bench=(), weather="noweather", opp_cond="noconditions"):
    return UniversalState(
        format="gen9ou", player_active_pokemon=me, opponent_active_pokemon=opp,
        available_switches=list(bench), player_prev_move=UniversalMove.blank_move(),
        opponent_prev_move=UniversalMove.blank_move(), opponents_remaining=6,
        player_conditions="noconditions", opponent_conditions=opp_cond,
        weather=weather, battle_field="nofield", forced_switch=False,
        battle_won=False, battle_lost=False, can_tera=True, opponent_teampreview=[],
    )


def test_type_chart_orientation_and_dual_types():
    assert type_mult("ground", ("electric",)) == 2
    assert type_mult("electric", ("ground",)) == 0
    assert type_mult("grass", ("dragon", "flying")) == 0.25
    assert type_mult("stellar", ("water",)) == 1  # unknown attack type is neutral


def test_revealed_ability_overrides_prior_and_unknown_uses_dex_prior():
    rotom = mon("rotomwash")  # only legal ability Levitate
    assert p_immune_ability(rotom, "ground") == 1.0
    tusk = mon("greattusk", ability="protosynthesis")
    assert p_immune_ability(tusk, "ground") == 0.0
    gastro = mon("gastrodon")  # Sticky Hold / Storm Drain / Sand Force
    assert p_immune_ability(gastro, "water") == pytest.approx(1 / 3)
    assert p_immune_ability(replace(gastro, ability="stickyhold"), "water") == 0.0


def test_damage_scales_with_stab_boost_screen_weather_and_burn():
    tusk = mon("greattusk", ability="protosynthesis")
    gholdengo = mon("gholdengo", ability="goodasgold")
    base, eff = damage_frac("ground", "physical", 100, "earthquake", tusk, gholdengo,
                            "noweather", "noconditions")
    assert eff == 2 and 0.5 < base < 2
    boosted, _ = damage_frac("ground", "physical", 100, "earthquake",
                             replace(tusk, atk_boost=2), gholdengo, "noweather", "noconditions")
    assert boosted == pytest.approx(2 * base, rel=0.02)
    for cond in ("reflect", "auroraveil"):
        screened, _ = damage_frac("ground", "physical", 100, "earthquake", tusk, gholdengo,
                                  "noweather", cond)
        assert screened == pytest.approx(base / 2, rel=0.02)
    burned, _ = damage_frac("ground", "physical", 100, "earthquake",
                            replace(tusk, status="brn"), gholdengo, "noweather", "noconditions")
    assert burned == pytest.approx(base / 2, rel=0.02)
    nostab, _ = damage_frac("ground", "physical", 100, "earthquake",
                            replace(tusk, types="fighting"), gholdengo, "noweather", "noconditions")
    assert base == pytest.approx(1.5 * nostab, rel=0.02)
    sun, _ = damage_frac("water", "special", 90, "surf", mon("primarina"), gholdengo,
                         "sunnyday", "noconditions")
    rain, _ = damage_frac("water", "special", 90, "surf", mon("primarina"), gholdengo,
                          "raindance", "noconditions")
    assert rain == pytest.approx(3 * sun, rel=0.05)


def test_move_overrides_fixed_damage_and_status_moves():
    corv = mon("corviknight")
    weak = mon("blissey")
    bp, _ = damage_frac("fighting", "physical", 80, "bodypress",
                        replace(corv, def_boost=2), weak, "noweather", "noconditions")
    bp0, _ = damage_frac("fighting", "physical", 80, "bodypress", corv, weak,
                         "noweather", "noconditions")
    assert bp == pytest.approx(2 * bp0, rel=0.02)  # Body Press uses own Def
    st, _ = damage_frac("fighting", "physical", 0, "seismictoss", corv, weak,
                        "noweather", "noconditions")
    assert st == pytest.approx(100 / (2 * 255 + 31 + 32 + 100 + 10), rel=0.01)
    ghost_immune, eff = damage_frac("fighting", "physical", 0, "seismictoss", corv,
                                    mon("gholdengo"), "noweather", "noconditions")
    assert eff == 0 and ghost_immune == 0
    assert damage_frac("normal", "status", 0, "roost", corv, weak, "noweather",
                       "noconditions")[0] == 0


def test_p_ko_roll_window_and_accuracy():
    assert p_ko(1.0, 0.8, 1.0) == 1.0  # min roll 0.85 already KOs
    assert p_ko(0.5, 0.8, 1.0) == 0.0
    assert p_ko(1.0, 0.925, 1.0) == pytest.approx(0.5)
    assert p_ko(1.0, 0.8, 0.8) == pytest.approx(0.8)
    assert p_ko(1.0, 0.0, 1.0) == 0.0


def test_extras_layout_padding_and_opp_revealed_moves_only():
    me = mon("greattusk", ["earthquake", "rapidspin"], ability="protosynthesis")
    opp = mon("gholdengo", ["makeitrain"], ability="goodasgold", hp=0.4)
    bench = [mon("dragonite", ["extremespeed", "earthquake"], ability="multiscale")]
    x = matchup_extras(state(me, opp, bench))
    assert x.shape == (EXTRAS_DIM,)
    eq = x[0:MOVE_F]
    assert eq[0] == pytest.approx(0.5) and eq[1] == 1 and eq[4] == 1.0 and eq[5] == 1
    assert (x[2 * MOVE_F : 4 * MOVE_F] == PAD).all()  # empty own move slots
    o = 4 * MOVE_F
    assert x[o] == pytest.approx(0.0) and x[o + 1] > 0  # steel vs ground/fighting neutral
    assert (x[o + OPP_MOVE_F : o + 4 * OPP_MOVE_F] == PAD).all()  # only one revealed
    b = o + 4 * OPP_MOVE_F + THREAT_F
    assert (x[b : b + BENCH_F] != PAD).all()
    assert (x[b + BENCH_F : b + 5 * BENCH_F] == PAD).all()
    # adding a later-revealed opponent move changes only its own slot
    opp2 = replace(opp, moves=opp.moves + mon("gholdengo", ["shadowball"]).moves)
    x2 = matchup_extras(state(me, opp2, bench))
    assert (x2[o + OPP_MOVE_F : o + 2 * OPP_MOVE_F] != PAD).all()
    assert (x2[:o] == x[:o]).all()


def test_speed_flags_respect_scarf_boost_and_paralysis():
    me = mon("dragapult")
    opp = mon("greattusk")
    s = EXTRAS_DIM - 3
    x = matchup_extras(state(me, opp))
    assert x[s] == 1 and x[s + 1] == 1  # 142 base outspeeds 87 base at max
    x = matchup_extras(state(me, replace(opp, item="choicescarf")))
    assert x[s + 1] == 0
    x = matchup_extras(state(replace(me, status="par"), opp))
    assert x[s + 1] == 0
    x = matchup_extras(state(mon("kingambit"), replace(opp, spe_boost=-6)))
    assert x[s] == 1 and x[s + 1] == 1


def test_encoder_v2_appends_extras_after_unchanged_v1_numbers(tmp_path):
    from pokeai.data.features import ObservationEncoder
    from pokeai.data.matchup import EXTRAS_VERSION

    st = state(mon("greattusk", ["earthquake"], ability="protosynthesis"),
               mon("gholdengo", ["makeitrain"], ability="goodasgold"),
               [mon("dragonite", ["extremespeed"])])
    v1 = ObservationEncoder().encode(st)
    v2_encoder = ObservationEncoder(extras=EXTRAS_VERSION)
    v2 = v2_encoder.encode(st)
    assert (v2["tokens"] == v1["tokens"]).all()
    assert (v2["numbers"][:48] == v1["numbers"]).all()
    assert (v2["numbers"][48:] == matchup_extras(st)).all()
    v2_encoder.save(tmp_path / "enc.json")
    assert ObservationEncoder.load(tmp_path / "enc.json").extras == EXTRAS_VERSION
    ObservationEncoder().save(tmp_path / "v1.json")
    assert ObservationEncoder.load(tmp_path / "v1.json").extras is None
