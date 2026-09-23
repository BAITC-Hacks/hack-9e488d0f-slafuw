"""Load local server-only environment variables; process env takes precedence."""

import os

from .catalog import ROOT

ALLOWED = {"OPENAI_API_KEY", "OPENAI_MODEL", "NVIDIA_API_KEY", "NVIDIA_EMBEDDING_MODEL",
           "EVENTMATCH_RANKER", "EVENTMATCH_NVIDIA_ARTIFACT", "EVENTMATCH_DB",
           "EVENTMATCH_DEFAULT_YEAR"}


def load_local_env(path=None):
    path = path or ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in ALLOWED:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))
