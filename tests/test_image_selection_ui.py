from types import SimpleNamespace
from PIL import Image
from streamlit.testing.v1 import AppTest
import image_selection as selector

SCRIPT = """
from PIL import Image
from image_selection import image_selection
image_selection(Image.new('RGB',(640,400),'white'),
    [{'id':'card','label':'کارت','box':[.1,.2,.9,.8]}],active='card',key='test')
"""


def test_selector_mounts_without_coordinate_sliders():
    app=AppTest.from_string(SCRIPT,default_timeout=15).run()
    assert not app.exception
    assert len(app.slider)==0


def call_selector(monkeypatch,event,active='card'):
    monkeypatch.setattr(selector,'_SELECTOR',lambda **kwargs:SimpleNamespace(selected=event))
    monkeypatch.setattr(selector.st,'session_state',{})
    return selector.image_selection(Image.new('RGB',(640,400)),[],active=active,key='event')


def test_submitted_crop_preserves_precise_normalized_coordinates(monkeypatch):
    box=[.20125,.225,.8125,.75625]
    event=dict(type='crop',field='card',box=box)
    assert call_selector(monkeypatch,event)==event


def test_uncommitted_draft_and_foreign_field_do_not_change_crop(monkeypatch):
    assert call_selector(monkeypatch,None) is None
    assert call_selector(monkeypatch,dict(field='last_name',box=[.1,.2,.8,.7])) is None


def test_invalid_or_nonfinite_crop_is_not_committed(monkeypatch):
    for box in [[0,0,float('nan'),1],[.4,.3,.401,.7],['bad',0,1,1]]:
        assert call_selector(monkeypatch,dict(field='card',box=box)) is None


def test_long_image_keys_keep_fields_and_images_separate():
    keys=[selector._key_part('fields_'+'a'*90+suffix) for suffix in ('_first','_last','_other_image')]
    assert len(set(keys))==3
