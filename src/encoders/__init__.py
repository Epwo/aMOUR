"""
aMOUR – encoders package
Swappable audio encoder backends.

Usage:
    from encoders import get_encoder, AVAILABLE_ENCODERS

    encoder = get_encoder("mert", device=device)
    embedding = encoder.encode_track(audio_array)
"""

from encoders.base import AudioEncoder
from encoders.mert import MERTEncoder
from encoders.clap import CLAPEncoder
from encoders.music2vec import Music2VecEncoder

AVAILABLE_ENCODERS: dict[str, type[AudioEncoder]] = {
    "mert": MERTEncoder,
    "clap": CLAPEncoder,
    "music2vec": Music2VecEncoder,
}


def get_encoder(name: str, device=None, **kwargs) -> AudioEncoder:
    """
    Factory: instantiate an encoder by name.

    Parameters
    ----------
    name   : one of "mert", "clap", "music2vec"
    device : torch.device (default: auto-detect CUDA)
    **kwargs : forwarded to the encoder constructor

    Returns
    -------
    An initialised AudioEncoder ready for .encode_chunks()
    """
    key = name.lower().replace("-", "").replace("_", "")
    # Normalise common aliases
    aliases = {"mertv1": "mert", "mert330m": "mert", "clapmusic": "clap", "m2v": "music2vec"}
    key = aliases.get(key, key)

    if key not in AVAILABLE_ENCODERS:
        raise ValueError(
            f"Unknown encoder '{name}'. "
            f"Available: {', '.join(AVAILABLE_ENCODERS)}"
        )

    return AVAILABLE_ENCODERS[key](device=device, **kwargs)
