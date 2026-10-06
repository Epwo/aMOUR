"""
aMOUR – encoders package
Swappable music encoder backends.

    from encoders import get_encoder
    encoder = get_encoder("muq", device=device)
    vector = encoder.encode_chunks(chunks)
"""

from encoders.base import MusicEncoder
from encoders.clap import CLAPEncoder
from encoders.mert import MERTEncoder
from encoders.muq import MuQEncoder, MuQMuLanEncoder
from encoders.omar import OMARRQEncoder

AVAILABLE_ENCODERS: dict[str, type[MusicEncoder]] = {
    "muq": MuQEncoder,
    "mulan": MuQMuLanEncoder,
    "omar": OMARRQEncoder,
    "mert": MERTEncoder,
    "clap": CLAPEncoder,
}

_ALIASES = {
    "muqmulan": "mulan", "omarrq": "omar", "mertv1": "mert", "clapmusic": "clap",
}


def canonical_name(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "")
    key = _ALIASES.get(key, key)
    if key not in AVAILABLE_ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Available: {', '.join(AVAILABLE_ENCODERS)}")
    return key


def get_encoder(name: str, device=None, layers: str | None = None) -> MusicEncoder:
    return AVAILABLE_ENCODERS[canonical_name(name)](device=device, layers=layers)
