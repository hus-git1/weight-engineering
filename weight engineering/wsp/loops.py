"""Detect degenerate repetition ("thinking loops") in generated text."""
import zlib


def compression_ratio(text):
    b = text.encode()
    return len(zlib.compress(b, 9)) / max(1, len(b))


def loop_start(text, window=1500, step=250, threshold=0.2):
    """Return the character index where the text becomes highly repetitive, or None.

    Slides a window over the text; repetitive text compresses extremely well.
    """
    if len(text) < window:
        return None
    for s in range(0, len(text) - window + 1, step):
        if compression_ratio(text[s:s + window]) < threshold:
            return s
    return None


def is_looping(text, finished):
    """A sample counts as looping if it has a highly repetitive region, or ran out of
    tokens while its tail is repetitive."""
    if loop_start(text) is not None:
        return True
    return (not finished) and compression_ratio(text[-3000:]) < 0.3
