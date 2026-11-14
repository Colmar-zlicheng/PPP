"""Timing convention of the consolidated LIBERO file.

LIBERO's exporter records the observation of row i after applying action a_i, so the commands that
follow observation i start at action row i + 1 (`action_offset` = 1).
"""
from dataclasses import asdict, dataclass


CORRECTED = "post_action_v1"


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class LiberoTiming:
    contract: str
    observation_timing: str
    source: str = "native_schema_default"

    def action_offset(self, action_source="osc"):
        assert action_source == "osc", f"unsupported action source {action_source!r}"
        return int(self.observation_timing == "post_action")

    def as_dict(self):
        return {**asdict(self), "osc_action_offset": self.action_offset("osc")}


def resolve_timing(cfg=None, h5=None):
    """The dataset node's `TIMING_CONTRACT` (only `post_action_v1`) and the file's observation timing
    (`observation_timing` attribute; LIBERO's native post-action recording when absent)."""
    contract = (cfg or {}).get("TIMING_CONTRACT", CORRECTED)
    if contract != CORRECTED:
        raise ValueError(f"TIMING_CONTRACT must be {CORRECTED!r}, got {contract!r}")
    observation, source = "post_action", "native_schema_default"
    if h5 is not None and "observation_timing" in h5.attrs:
        observation, source = _text(h5.attrs["observation_timing"]), "h5_metadata"
    if observation not in ("pre_action", "post_action"):
        raise ValueError(f"Unknown observation_timing {observation!r}")
    return LiberoTiming(contract, observation, source)
