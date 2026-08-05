import urllib.request

import pytest

from evaluation.nlseeker.__main__ import _ping, build_arg_parser


def test_config_and_resume_are_mutually_exclusive():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--config', 'a.yaml', '--resume', 'dir'])
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parses_config():
    args = build_arg_parser().parse_args(['--config', 'a.yaml'])
    assert args.config == 'a.yaml' and args.resume is None


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def close(self):
        pass


def test_ping_returns_model_ids(monkeypatch):
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda *args, **kwargs: _FakeResponse(b'{"data":[{"id":"m1"}]}'))
    assert _ping('http://x', 'key') == ['m1']
