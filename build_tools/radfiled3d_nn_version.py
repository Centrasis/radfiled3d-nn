"""Dynamic version provider for scikit-build-core.

**The release TAG is the version, and it reaches the C++ library too.** scikit-build-core puts what
this returns into the CMake cache as `SKBUILD_PROJECT_VERSION`, and `CMakeLists.txt` uses it for
`project(radfiled3d_nn VERSION ...)` — which feeds `include/RadFiled3D/nn/version.hpp.in`, and so
`kVersion`, the C ABI's `rfnn_version()` and Python's `RadFiled3D.nn.version()`. One number, from
one place, for a wheel and the library inside it.

The same rule RadFiled3D uses (`build_tools/radfiled3d_version.py`), deliberately, so the two
repositories release the same way:

* GitHub Actions:  ``GITHUB_REF``     (``refs/tags/1.2.3`` -> ``1.2.3``)
* GitLab CI:       ``CI_COMMIT_TAG``, then ``CI_COMMIT_REF_NAME``

Anything that does not look like a release tag builds as ``0.0.0``. That is not a rough edge to be
tidied away: a wheel built from a branch push MUST NOT claim a release version, or it would outrank
the real one on a local install.

A plain `cmake -S . -B build` outside CI resolves the same way and gets ``0.0.0`` — which is the
honest answer for a working tree that is not a release. Pass ``-DRFNN_VERSION=X.Y.Z`` to say
otherwise.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

__all__ = ["dynamic_metadata", "resolve_version"]


def resolve_version() -> str:
    version = "0.0.0"
    if os.environ.get("CI_COMMIT_TAG"):  # GitLab CI tag
        version = os.environ["CI_COMMIT_TAG"]
    elif os.environ.get("CI_COMMIT_REF_NAME"):  # GitLab CI branch/ref
        version = os.environ["CI_COMMIT_REF_NAME"]
    elif os.environ.get("GITHUB_REF"):  # GitHub Actions ref, e.g. refs/tags/1.2.3
        version = os.environ["GITHUB_REF"].split("/")[-1]

    # Only "X.Y.Z" is a release; a branch name is not a version.
    if re.match(r"^\d+\.\d+\.\d+", version) is None:
        version = "0.0.0"
    return version


def dynamic_metadata(field: str, settings: Mapping[str, Any] | None = None) -> str:
    if field != "version":
        raise ValueError(f"this provider only supplies `version`, not `{field}`")
    if settings:
        raise ValueError("this provider takes no settings")
    return resolve_version()
