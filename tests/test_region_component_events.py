import ast
from pathlib import Path
import shutil,subprocess
import pytest


def test_real_component_pointer_events_on_synthetic_canvas(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node is required only for the isolated JavaScript event test')
    root=Path(__file__).resolve().parent.parent
    module=ast.parse((root/'image_selection.py').read_text(encoding='utf-8'))
    call=next(n.value for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_SELECTOR' for t in n.targets))
    js=next(ast.literal_eval(k.value) for k in call.keywords if k.arg=='js')
    source=tmp_path/'selection.mjs';source.write_text(js,encoding='utf-8')
    result=subprocess.run([node,str(root/'tests/test_region_component.mjs'),str(source)],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert 'passed' in result.stdout
