"""Gen9 OU Showdown decision model."""

# metamon.baselines builds GenData(gen) at import time, and poke-env's GenData
# constructor raises once GenData.from_gen(gen) has run. Import it first so any
# pokeai entry point is safe regardless of what touches poke-env data later.
import metamon.baselines  # noqa: F401
