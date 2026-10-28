import json
import subprocess
from pathlib import Path
import pytest
from ocr_process import OCRProcess


@pytest.fixture
def client(tmp_path):
    script = tmp_path / 'worker.py'
    script.write_text('''import json,sys,time
from pathlib import Path
for line in sys.stdin:
 r=json.loads(line)
 data=Path(r['source']).read_text()
 print(json.dumps({'token':r['token'],'progress':{'field':'national_id'}}),flush=True)
 if data=='sleep':time.sleep(10)
 if data=='crash':sys.exit(2)
 Path(r['output']).write_text(json.dumps({'ok':True,'literal':data}))
 print(r['token'],flush=True)
''', encoding='utf-8')
    worker = OCRProcess(script)
    yield worker
    worker.close()


def request(client, tmp_path, text, timeout=5):
    source = tmp_path / 'source.txt'; output = tmp_path / 'out.json'
    source.write_text(text, encoding='utf-8')
    output.unlink(missing_ok=True)
    return client.read(source, output, timeout)


def test_models_reused_but_result_is_never_reused_between_inputs(client, tmp_path):
    first = request(client,tmp_path,'image A')
    pid = client.process.pid
    second = request(client,tmp_path,'image B')
    assert first['literal']=='image A' and not first['models_reused']
    assert second['literal']=='image B' and second['models_reused']
    assert client.process.pid==pid


def test_deadline_kills_child_and_next_request_starts_fresh(client, tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        request(client,tmp_path,'sleep',.2)
    assert client.process is None
    assert request(client,tmp_path,'new image')['literal']=='new image'


def test_crash_does_not_return_previous_output(client, tmp_path):
    request(client,tmp_path,'old image')
    with pytest.raises(RuntimeError):
        request(client,tmp_path,'crash')
    assert client.process is None
    assert not request(client,tmp_path,'new image')['models_reused']


def test_configuration_change_restarts_models(client, tmp_path, monkeypatch):
    request(client,tmp_path,'old image'); previous=client.process.pid
    monkeypatch.setenv('BINA_ENABLED','0')
    fresh=request(client,tmp_path,'new image')
    assert not fresh['models_reused'] and client.process.pid!=previous


def test_progress_is_delivered_without_replacing_the_result(client, tmp_path):
    source=tmp_path/'source.txt';source.write_text('new image')
    events=[]
    result=client.read(source,tmp_path/'out.json',5,progress=lambda event,payload:events.append((event,payload)))
    assert events==[('local_field_done',{'field':'national_id'})]
    assert result['literal']=='new image'
