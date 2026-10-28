import hashlib
import json
import verify_installation


def test_manifest_missing_file_cannot_pass_deep_check(tmp_path,monkeypatch):
    monkeypatch.setattr(verify_installation,'REQUIRED',[])
    (tmp_path/'models').mkdir()
    (tmp_path/'models/persian_crnn.onnx').write_bytes(b'fixture')
    (tmp_path/'FILE_MANIFEST.json').write_text(json.dumps({'files':[
        {'path':'static_ctc.py','sha256':hashlib.sha256(b'expected').hexdigest()}
    ]}),encoding='utf-8')
    result=verify_installation.verify(deep=True,root=tmp_path)
    assert not result['ok']
    assert 'static_ctc.py' in result['missing']


def test_manifest_modified_file_cannot_pass_deep_check(tmp_path,monkeypatch):
    monkeypatch.setattr(verify_installation,'REQUIRED',[])
    (tmp_path/'models').mkdir()
    (tmp_path/'models/persian_crnn.onnx').write_bytes(b'fixture')
    (tmp_path/'static_ctc.py').write_bytes(b'changed')
    (tmp_path/'FILE_MANIFEST.json').write_text(json.dumps({'files':[
        {'path':'static_ctc.py','sha256':hashlib.sha256(b'expected').hexdigest()}
    ]}),encoding='utf-8')
    result=verify_installation.verify(deep=True,root=tmp_path)
    assert not result['ok']
    assert result['changed']==['static_ctc.py']
