from local_worker import _engine_candidate, select_engine_consensus
from validation import checksum_valid, normalize


def number(prefix):
    remainder = sum(int(c) * (10-i) for i, c in enumerate(prefix)) % 11
    return prefix + str(remainder if remainder < 2 else 11-remainder)


def test_repeated_literal_beats_accidental_valid_checksum_within_model():
    observed = '1234567890'
    accidental = number('938271640')
    assert not checksum_valid(observed) and checksum_valid(accidental)
    row = _engine_candidate([observed]*8+[accidental], [.99]*8+[.7], 'national_id')
    assert row['value'] == observed and row['support'] == 8


def test_strong_persian_number_beats_weak_checksum_valid_alternative():
    observed = number('938271640')
    accidental = number('731826450')
    text, ambiguous, _, _, source = select_engine_consensus(
        {'persian_crnn':[observed]*8, 'paddleocr_v5_arabic':[accidental]*8},
        {'persian_crnn':[.98]*8, 'paddleocr_v5_arabic':[.65]*8}, 'national_id')
    assert normalize(text) == observed and source == 'persian_crnn' and ambiguous


def test_two_strong_disagreeing_numbers_never_become_accepted():
    a, b = number('938271640'), number('731826450')
    assert select_engine_consensus({'a':[a]*3,'b':[b]*3}, {'a':[.99]*3,'b':[.99]*3}, 'national_id')[1]


def test_rare_reads_among_missing_views_are_not_stable():
    a = number('938271640')
    assert not _engine_candidate([a,a]+['']*7,[.99,.99]+[0]*7,'national_id')['stable']


def test_two_weak_agreeing_engines_do_not_override_one_strong_transcription():
    a, b = number('938271640'), number('731826450')
    text, ambiguous, *_ = select_engine_consensus(
        {'persian_crnn':[a]*3,'bina_rizeh_ppocrv6':[b]*3,'paddleocr_v5_arabic':[b]*3},
        {'persian_crnn':[.99]*3,'bina_rizeh_ppocrv6':[.6]*3,'paddleocr_v5_arabic':[.7]*3}, 'national_id')
    assert text == a and ambiguous


def test_border_recovery_preserves_literal_observations_and_does_not_repair_digits():
    from PIL import Image
    from local_worker import _numeric_border_recovery
    original='1234567890'; observed=number('938271640')
    class Reader:
        def read_scored(self,image):return observed,.99
    texts,scores,detail=_numeric_border_recovery(Image.new('RGB',(200,40),'white'),Reader(),[original]*9,[.8]*9)
    assert texts and set(texts)=={observed}
    assert detail['original_readings']==[original]*9
    assert detail['changed_value'] and detail['selected_family']=='border'


def test_valid_stable_literal_is_not_reprocessed_with_new_geometry():
    from PIL import Image
    from local_worker import _numeric_border_recovery
    a=number('938271640')
    class Reader:
        def read_scored(self,image):raise AssertionError('unnecessary recovery')
    assert _numeric_border_recovery(Image.new('RGB',(200,40),'white'),Reader(),[a]*9,[.99]*9)==([a]*9,[.99]*9,None)


def test_weak_checksum_valid_reading_still_gets_a_visual_retry():
    from PIL import Image
    from local_worker import _numeric_border_recovery
    a,b=number('938271640'),number('731826450')
    class Reader:
        def read_scored(self,image):return b,.99
    texts,_,detail=_numeric_border_recovery(Image.new('RGB',(200,40),'white'),Reader(),[a]*9,[.91]*9)
    assert set(texts)=={b} and detail['changed_value']


def test_border_family_change_stays_review_only_even_with_agreement(tmp_path):
    import json
    from PIL import Image
    from local_worker import run
    a=number('938271640')
    class Legacy:
        def __init__(self):self.calls=0
        def read_scored(self,image):
            self.calls+=1
            return ('1234567890',.8) if self.calls<=9 else (a,.99)
    class Primary:
        name='paddleocr_v5_arabic'
        def read_scored(self,image):return a,.99
    path=tmp_path/'card.png';Image.new('RGB',(300,80),'white').save(path)
    path.with_suffix('.png.json').write_text(json.dumps({'national_id':[0,0,1,1]}))
    legacy=Legacy();row=run(str(path),readers=(legacy,legacy,[Primary()],[]))['national_id']
    assert normalize(row['text'])==a and row['ambiguous'] and row['decision']=='review'


def test_number_preview_is_provisional_and_arrives_before_remaining_fields(tmp_path):
    import json
    from PIL import Image
    from local_worker import run
    a=number('938271640')
    class Reader:
        def __init__(self,name):self.name=name
        def read_scored(self,image):return a,.99
    path=tmp_path/'card.png';Image.new('RGB',(300,100),'white').save(path)
    path.with_suffix('.png.json').write_text(json.dumps({'national_id':[0,0,1,.4],'first_name':[0,.5,1,1]}))
    events=[];legacy=Reader('legacy')
    run(str(path),readers=(legacy,legacy,[Reader('paddleocr_v5_arabic')],[]),progress=events.append)
    assert [e['field'] for e in events]==['national_id','first_name']
    assert events[0]['preview']['value']==a and events[0]['preview']['provisional'] is True
    assert 'confidence_percent' in events[0]['preview'] and 'preview' not in events[1]
