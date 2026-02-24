"""ZSE version information."""

__version__ = "0.1.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Version metadata
VERSION_MAJOR = __version_info__[0]
VERSION_MINOR = __version_info__[1]
VERSION_PATCH = __version_info__[2]

# Build info (populated during release)
BUILD_DATE = "2026-02-23"
GIT_COMMIT = "development"
