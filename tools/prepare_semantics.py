"""Run once with NVIDIA_API_KEY; only successful real API calls produce an artifact."""
import json
from pathlib import Path

from eventmatch.catalog import load_catalog
from eventmatch.config import Settings, load_env
from eventmatch.providers import NVIDIA, ProviderError
from eventmatch.semantic import prepare


def main():
    load_env()
    settings = Settings.from_env()
    try:
        artifact = prepare(load_catalog(), NVIDIA(settings))
        path = Path(settings.artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
        print(f"Prepared {path.name}: {artifact['api_calls']} NVIDIA calls; "
              f"sha256={artifact['artifact_sha256']}")
    except (ProviderError, ValueError, OSError) as exc:
        raise SystemExit(f'Preparation failed: {exc}') from None


if __name__ == '__main__':
    main()
