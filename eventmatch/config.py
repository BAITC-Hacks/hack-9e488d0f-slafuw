"""Server-only environment configuration; never serialize credentials."""
from dataclasses import dataclass, field
import os
from pathlib import Path

from .catalog import ROOT


def load_env(path=None):
    path = Path(path or ROOT / '.env')
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        if not sep or not key.strip().replace('_', '').isalnum():
            raise ValueError('Некорректная строка .env')
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


@dataclass(frozen=True)
class Settings:
    openai_key: str = field(default='', repr=False)
    nvidia_key: str = field(default='', repr=False)
    openai_model: str = 'gpt-4.1-mini-2025-04-14'
    nvidia_model: str = 'nvidia/llama-3.2-nv-embedqa-1b-v2'
    ranker: str = 'baseline'
    artifact: str = str(ROOT / 'data/semantic/nvidia.json')
    database: str = str(ROOT / 'var/eventmatch.sqlite3')
    deadline_seconds: float = 9.0

    @classmethod
    def from_env(cls):
        obj = cls(openai_key=os.getenv('OPENAI_API_KEY', ''),
                  nvidia_key=os.getenv('NVIDIA_API_KEY', ''),
                  openai_model=os.getenv('OPENAI_MODEL', cls.openai_model),
                  nvidia_model=os.getenv('NVIDIA_EMBEDDING_MODEL', cls.nvidia_model),
                  ranker=os.getenv('EVENTMATCH_RANKER', 'baseline'),
                  artifact=os.getenv('NVIDIA_ARTIFACT_PATH', cls.artifact),
                  database=os.getenv('EVENTMATCH_DB', cls.database),
                  deadline_seconds=float(os.getenv('EVENTMATCH_DEADLINE_SECONDS', '9')))
        if obj.ranker not in ('baseline', 'nvidia'):
            raise ValueError('EVENTMATCH_RANKER: baseline или nvidia')
        if not 1 <= obj.deadline_seconds <= 10:
            raise ValueError('EVENTMATCH_DEADLINE_SECONDS: от 1 до 10')
        return obj
