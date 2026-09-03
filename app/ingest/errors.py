class IngestError(Exception):
    """Raised when an upload cannot be probed or normalised."""


class UnreadableVideo(IngestError):
    """The file is not a video ffmpeg can decode."""


class NormalisationFailed(IngestError):
    """ffmpeg exited non-zero while normalising."""
