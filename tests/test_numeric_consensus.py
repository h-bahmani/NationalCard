import json
from PIL import Image
from local_worker import select_engine_consensus

VALUE='0017864607'


def test_one_numeric_model_repeated_nine_times_stays_review_only():
    _,ambiguous,_,_,_=select_engine_consensus({'single':[VALUE]*9},{'single':[.99]*9},'national_id')
    assert ambiguous


def test_two_independent_numeric_models_can_agree():
    _,ambiguous,_,_,_=select_engine_consensus({'a':[VALUE]*2,'b':[VALUE]*2},
        {'a':[.99]*2,'b':[.99]*2},'national_id')
    assert not ambiguous


def test_fallback_runs_when_only_one_primary_has_a_valid_candidate(tmp_path,monkeypatch):
    import persian_reader,names_reader,ocr_ensemble,local_worker
    called=[]
    class Legacy:
        def read_scored(self,image):
            called.append(True)
            return VALUE,.99
    class Empty:
        name='bina_rizeh_ppocrv6'
        def read_scored(self,image): return '',0.0
    class Primary:
        name='paddleocr_v5_arabic'
        def read_scored(self,image): return VALUE,.99
    monkeypatch.setattr(persian_reader,'PersianReader',Legacy)
    monkeypatch.setattr(names_reader,'NamesReader',Legacy)
    monkeypatch.setattr(ocr_ensemble,'build_optional_readers',lambda:([Empty(),Primary()],[]))
    path=tmp_path/'field.png'
    Image.new('RGB',(320,80),'white').save(path)
    (tmp_path/'field.png.json').write_text(json.dumps({'national_id':[0,0,1,1]}))
    fields=local_worker.run(str(path))
    assert called
    assert fields['national_id']['engine_evidence']['persian_crnn']['stable']
    assert not fields['national_id']['ambiguous']


def test_strong_persian_digits_skip_medium_model_without_reducing_mobile_evidence(tmp_path):
    import local_worker
    class Reader:
        def __init__(self,name):self.name=name;self.calls=0
        def read_scored(self,image):self.calls+=1;return VALUE,.99
    medium=Reader('bina_rizeh_ppocrv6');mobile=Reader('paddleocr_v5_arabic')
    legacy=Reader('legacy')
    path=tmp_path/'bounded.png'
    Image.new('RGB',(320,80),'white').save(path)
    (tmp_path/'bounded.png.json').write_text(json.dumps({'national_id':[0,0,1,1]}))
    fields=local_worker.run(str(path),readers=(legacy,legacy,[medium,mobile],[]))
    assert medium.calls==0 and mobile.calls==9
    assert legacy.calls==9
    assert not fields['national_id']['ambiguous']
    assert len(fields['national_id']['engine_evidence'])==2
