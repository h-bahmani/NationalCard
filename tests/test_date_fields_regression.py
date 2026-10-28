from copy import deepcopy
import pytest
from local_worker import select_engine_consensus
from presentation import date_value,field_display,result_field_value
from cardreader_module import _validate_field_box,read_prepared
from pipeline import REGIONS
from company_policy import build_company_result
from test_company_reliability import strong_result
from test_ocr_gate_regression import prepared,SCRIPT
from streamlit.testing.v1 import AppTest


def test_high_scoring_persian_date_wins_review_tie_without_digit_repair():
    text,ambiguous,_,evidence,engine=select_engine_consensus(
        {'paddleocr_v5_arabic':['١۴٥۴/١١/١٧']*5,'persian_crnn':['۱۴۰۴/۱۱/۱۷']*8},
        {'paddleocr_v5_arabic':[.8]*5,'persian_crnn':[.99]*8},'expiry_date')
    assert date_value(text)=='1404/11/17'
    assert ambiguous and engine=='persian_crnn'
    assert evidence['paddleocr_v5_arabic']['value']=='1454/11/17'


@pytest.mark.parametrize('text',['11/17/1404','1404/17/11','1404/07/31','17/11/404','abc','۱۴۰۴/۱۷/۱۱','1404/11/1','1404/1/17'])
def test_malformed_date_does_not_appear_as_a_date_in_main_result(text):
    row=dict(text=text,raw_selected_text=text,ambiguous=True)
    shown=field_display('birth_date',row)
    assert shown['value']=='—' and shown['raw_value']==text
    assert result_field_value({'birth_date':row},'birth_date')==('—','missing','')


def test_date_review_is_canonical_year_first_and_stays_review():
    assert result_field_value({'expiry_date':dict(status='review',raw_text='۱۴۰۴/۱۱/۱۷')},'expiry_date')==('1404/11/17','raw','')


@pytest.mark.parametrize('name,other',[('birth_date','expiry_date'),('expiry_date','birth_date')])
def test_wrong_date_row_cannot_be_committed(name,other):
    with pytest.raises(ValueError,match='کادر'):
        _validate_field_box(name,REGIONS['smart'][other],'smart')
    _validate_field_box(name,REGIONS['smart'][name],'smart')


def test_crop_covering_two_dates_is_rejected():
    with pytest.raises(ValueError,match='هر دو تاریخ'):
        _validate_field_box('birth_date',[.5,.58,.85,.91],'smart')


def test_wrong_date_crop_is_blocked_before_any_ocr(monkeypatch):
    monkeypatch.setattr('cardreader_module.pipeline.extract',lambda *a,**k:pytest.fail('wrong row reached OCR'))
    with pytest.raises(ValueError,match='پایان اعتبار'):
        read_prepared(prepared(True),field_boxes={'birth_date':REGIONS['smart']['expiry_date']})


def add_expiry(result,value='1404/11/17',ambiguous=True):
    row=dict(text=value,raw_selected_text=value,decision='review' if ambiguous else 'accepted',ambiguous=ambiguous,
             engine_evidence={e:dict(value=value,mean_score=.99,stable=True) for e in ('bina_rizeh_ppocrv6','paddleocr_v5_arabic')})
    result['raw_result']['fields']['expiry_date']=row
    result['fields']['expiry_date']=dict(value=None if ambiguous else value,raw_text=value)
    return result


def test_expiry_is_available_for_correction_even_when_not_in_legacy_required_fields():
    result=add_expiry(strong_result())
    out=build_company_result(result)
    assert 'expiry_date' in out['next_action']['fields']
    assert out['status']=='review' and out['fields']['expiry_date']['value'] is None


@pytest.mark.parametrize('expiry',['1370/01/01','1369/12/29'])
def test_equal_or_earlier_expiry_never_exports_two_accepted_dates(expiry):
    out=build_company_result(add_expiry(strong_result(),expiry,False))
    assert 'date_order_conflict' in out['reasons']
    assert out['fields']['birth_date']['value'] is None
    assert out['fields']['expiry_date']['value'] is None
    assert {'birth_date','expiry_date'} <= set(out['next_action']['fields'])


def test_actual_app_shows_expiry_choice_and_isolates_date_direction():
    fixture=add_expiry(strong_result())
    fixture.update(company=build_company_result(fixture),number_preview='0017864607')
    script=SCRIPT.replace("with patch('streamlit.file_uploader'",
        "with patch('cardreader_module.read_prepared',return_value=st.session_state['fixture_result']), patch('streamlit.file_uploader'")
    app=AppTest.from_string(script,default_timeout=40)
    app.session_state['fixture_finding']=deepcopy(prepared(True).detection)
    app.session_state['fixture_decision']=fixture['decision']|dict(allowed=True,status='accepted')
    app.session_state['fixture_result']=fixture
    app.run()
    next(b for b in app.button if b.label=='خواندن اطلاعات').click().run()
    assert not app.exception
    chooser=next(s for s in app.selectbox if s.label=='فیلد نیازمند بررسی')
    assert 'تاریخ پایان اعتبار' in chooser.options
    markup=' '.join(x.value for x in app.markdown)
    assert '<b dir="ltr" style="unicode-bidi:isolate">1404/11/17</b>' in markup
