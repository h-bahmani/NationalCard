import json
from PIL import Image
from local_worker import run
from anti_hallucination import assess_text_field


class Reader:
    def __init__(self, name, score=.99, text='علی'):
        self.name, self.score, self.text, self.calls = name, score, text, 0
    def read_scored(self, image):
        self.calls += 1
        return self.text, self.score


def field(tmp_path, readers):
    path = tmp_path / 'card.png'
    Image.new('RGB',(300,100),'white').save(path)
    path.with_suffix('.png.json').write_text(json.dumps({'first_name':[0,0,1,1]}))
    return run(str(path),readers=readers)['first_name']


def test_three_strong_literal_models_avoid_heavy_recognition(tmp_path):
    heavy=Reader('bina_rizeh_ppocrv6');mobile=Reader('paddleocr_v5_arabic')
    crnn=Reader('persian_crnn');names=Reader('paddle_names')
    row=field(tmp_path,(crnn,names,[heavy,mobile],[]))
    assert row['accepted_text']=='علی' and not row['ambiguous']
    assert row['recognition_path']=='three_model_fast_consensus'
    assert heavy.calls==0 and len(row['engine_evidence'])==3


def test_weak_third_model_keeps_full_primary_path(tmp_path):
    heavy=Reader('bina_rizeh_ppocrv6');mobile=Reader('paddleocr_v5_arabic')
    row=field(tmp_path,(Reader('persian_crnn'),Reader('paddle_names',.7),[heavy,mobile],[]))
    assert row['recognition_path']=='full_primary_consensus' and heavy.calls==3
    assert set(row['engine_evidence'])=={'bina_rizeh_ppocrv6','paddleocr_v5_arabic'}


def test_fast_exception_never_silences_observed_heavy_model_conflict():
    evidence={k:dict(value='علی',stable=True,mean_score=.99)
              for k in ('paddleocr_v5_arabic','persian_crnn','paddle_names')}
    evidence['bina_rizeh_ppocrv6']=dict(value='رضا',stable=True,mean_score=.99)
    assert assess_text_field('first_name','علی',evidence,False)['decision']=='review'


def test_cheap_model_failure_does_not_break_successful_primary_reading(tmp_path):
    class Broken:
        def read_scored(self,image):raise RuntimeError('engine failed')
    row=field(tmp_path,(Broken(),Broken(),[Reader('bina_rizeh_ppocrv6'),Reader('paddleocr_v5_arabic')],[]))
    assert row['accepted_text']=='علی' and not row['ambiguous']


def test_three_literal_votes_with_two_strong_models_can_take_fast_path(tmp_path):
    heavy=Reader('bina_rizeh_ppocrv6');mobile=Reader('paddleocr_v5_arabic',.82)
    row=field(tmp_path,(Reader('persian_crnn',.94),Reader('paddle_names',.96),[heavy,mobile],[]))
    assert heavy.calls==0 and row['recognition_path']=='three_model_fast_consensus'
    assert row['accepted_text']=='علی'
