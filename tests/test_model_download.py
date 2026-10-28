import hashlib
import requests
import pytest
from accuracy_setup import _download_file


class Response:
    def __init__(self, chunks, status=200, headers=None):
        self.chunks, self.status_code, self.headers = chunks, status, headers or {}
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def raise_for_status(self): pass
    def iter_content(self, size):
        for item in self.chunks:
            if isinstance(item, Exception): raise item
            yield item


def test_verified_model_needs_no_network(tmp_path, monkeypatch):
    data = b'known model'
    (tmp_path/'model.bin').write_bytes(data)
    monkeypatch.setattr(requests,'get',lambda *a,**k:pytest.fail('must reuse verified file'))
    _download_file('author/model','model.bin',tmp_path,hashlib.sha256(data).hexdigest())


def test_interrupted_model_resumes_and_verifies(tmp_path, monkeypatch):
    calls = []
    def get(*args,**kwargs):
        calls.append(kwargs['headers'])
        if len(calls) == 1:
            return Response([b'abc',requests.ConnectionError('interrupted')],headers={'Content-Length':'6'})
        return Response([b'def'],206,{'Content-Range':'bytes 3-5/6','Content-Length':'3'})
    monkeypatch.setattr(requests,'get',get)
    _download_file('author/model','model.bin',tmp_path,hashlib.sha256(b'abcdef').hexdigest())
    assert (tmp_path/'model.bin').read_bytes() == b'abcdef'
    assert calls == [{},{'Range':'bytes=3-'}]


def test_checksum_failure_keeps_existing_model(tmp_path, monkeypatch):
    (tmp_path/'model.bin').write_bytes(b'old model')
    monkeypatch.setattr(requests,'get',lambda *a,**k:Response([b'bad model']))
    with pytest.raises(RuntimeError,match='Model download failed'):
        _download_file('author/model','model.bin',tmp_path,hashlib.sha256(b'good model').hexdigest())
    assert (tmp_path/'model.bin').read_bytes() == b'old model'


def test_complete_partial_download_is_recovered_without_network(tmp_path, monkeypatch):
    data = b'complete model'
    (tmp_path/'model.bin.part').write_bytes(data)
    monkeypatch.setattr(requests,'get',lambda *a,**k:pytest.fail('complete partial must be recovered'))
    _download_file('author/model','model.bin',tmp_path,hashlib.sha256(data).hexdigest())
    assert (tmp_path/'model.bin').read_bytes() == data


def test_empty_download_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(requests,'get',lambda *a,**k:Response([]))
    with pytest.raises(RuntimeError):
        _download_file('author/model','model.bin',tmp_path)
    assert not (tmp_path/'model.bin').exists()
