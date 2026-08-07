import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Typing imports
from typing import Dict, Optional


class RunDir:
    """A timestamped result directory; '.partial' until finalize()."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, results_root: Path, tag: str, config_text: str) -> 'RunDir':
        stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')
        path = Path(results_root) / f'{stamp}__{tag}.partial'
        path.mkdir(parents=True)
        run = cls(path)
        run.config_path.write_text(config_text)
        run._write_meta({'created_utc': stamp, 'datasets': {}})
        return run

    @classmethod
    def open_partial(cls, path: Path) -> 'RunDir':
        path = Path(path)
        run = cls(path)
        if not path.is_dir() or not run.config_path.is_file() or not run._meta_path.is_file():
            raise FileNotFoundError(f'not a resumable run dir: {path}')
        return run

    @property
    def config_path(self) -> Path:
        return self.path / 'config.yaml'

    @property
    def _meta_path(self) -> Path:
        return self.path / 'run_meta.json'

    def dataset_dir(self, name: str) -> Path:
        directory = self.path / name
        directory.mkdir(exist_ok=True)
        return directory

    def artifacts_dir(self, name: str) -> Path:
        directory = self.dataset_dir(name) / 'artifacts'
        directory.mkdir(exist_ok=True)
        return directory

    def meta(self) -> dict:
        return json.loads(self._meta_path.read_text())

    def update_meta(self, **fields) -> None:
        meta = self.meta()
        meta.update(fields)
        self._write_meta(meta)

    def set_dataset(self, name: str, **fields) -> None:
        meta = self.meta()
        meta['datasets'].setdefault(name, {}).update(fields)
        self._write_meta(meta)

    def dataset_status(self, name: str) -> Optional[str]:
        return self.meta()['datasets'].get(name, {}).get('status')

    def wipe_dataset(self, name: str) -> None:
        target = self.path / name
        if target.exists():
            shutil.rmtree(target)
        meta = self.meta()
        meta['datasets'].pop(name, None)
        self._write_meta(meta)

    def finalize(self) -> Path:
        final = self.path.with_name(self.path.name.removesuffix('.partial'))
        os.replace(self.path, final)
        for ini in final.glob('*/artifacts/nlseeker.ini'):
            ini.write_text(ini.read_text().replace(str(self.path.resolve()), str(final.resolve())))
        self.path = final
        return final

    def _write_meta(self, meta: dict) -> None:
        tmp = self._meta_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, self._meta_path)


def write_ini(artifacts_dir: Path, dataset: str, nlseeker: Dict[str, str],
              alpha: float, n: int) -> Path:
    artifacts_dir = Path(artifacts_dir).resolve()
    lines = ['[Database]', 'dbms=duckdb', f'path={artifacts_dir / "eval.duckdb"}',
             f'index_table={dataset}', '', '[NLSeeker]',
             f'index_name={dataset}', f'schema=nl_{dataset}']
    lines += [f'{key}={value}' for key, value in nlseeker.items()]
    lines += [f'alpha={alpha}', f'n={n}', '']
    ini = artifacts_dir / 'nlseeker.ini'
    ini.write_text('\n'.join(lines))
    return ini
