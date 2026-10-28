from copy import deepcopy
from pathlib import Path
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
import cardreader_module as module


def prepared(verified=False):
    image=Image.new('RGB',(800,500),'white')
    finding=dict(detected=verified,kind='smart',method='layout_match',
                 card_quad=[[0,0],[799,0],[799,499],[0,499]])
    if verified:
        finding.update(inliers=40,matches=44,reference_coverage=.3)
    else:
        finding.update(ocr_candidate=True,method='full_frame_card_candidate')
    return module.PreparedCard(image,finding,dict(allowed=True,status='candidate'),image,None)


def test_gate_blocks_legacy_cached_candidate_without_mutating_it():
    card=prepared()
    gated=module.apply_ocr_gate(card)
    assert not gated.decision['allowed']
    assert gated.decision['status']=='selection_required'
    assert card.decision['allowed']


def test_direct_read_cannot_bypass_gate(monkeypatch):
    monkeypatch.setattr(module.pipeline,'extract',lambda *a,**k:pytest.fail('unverified automatic crop reached OCR'))
    result=module.read_prepared(prepared())
    assert result['company']['next_action']['type']=='select_card_region'
    assert all(f['value'] is None for f in result['company']['fields'].values())


def test_verified_simple_card_keeps_ocr_access():
    card=prepared(True)
    assert module.apply_ocr_gate(card) is card
    assert card.decision['allowed']


def test_configured_threshold_is_not_overridden_by_default():
    card=prepared(True)
    card.detection.update(inliers=12,matches=18,reference_coverage=.08)
    assert not module.apply_ocr_gate(card,.85).decision['allowed']
    assert module.apply_ocr_gate(card,.50).decision['allowed']


SCRIPT='''
import io,runpy
from pathlib import Path
from unittest.mock import patch
import streamlit as st
from PIL import Image
from cardreader_module import PreparedCard
data=io.BytesIO()
Image.new('RGB',(800,500),'white').save(data,'PNG')
data.name='regression.png'
finding=st.session_state['fixture_finding']
card=PreparedCard(Image.new('RGB',(800,500),'white'),finding,
    st.session_state.get('fixture_decision',dict(allowed=True,status='candidate')),Image.new('RGB',(800,500),'white'),None)
with patch('streamlit.file_uploader',return_value=data), patch('cardreader_module.prepare',return_value=card):
    runpy.run_path(str(Path('app.py').resolve()),run_name='__main__')
'''


@pytest.mark.parametrize('verified',[False,True])
def test_actual_app_only_offers_read_on_verified_automatic_card(verified):
    app=AppTest.from_string(SCRIPT,default_timeout=40)
    app.session_state['fixture_finding']=deepcopy(prepared(verified).detection)
    app.run()
    assert not app.exception
    buttons={b.label for b in app.button}
    assert ('خواندن اطلاعات' in buttons) is verified
    if not verified:
        assert 'بررسی کادر انتخابی' in buttons
        html=' '.join(m.value for m in app.markdown)
        assert 'کارت آمادهٔ خواندن است' not in html
        assert 'mini-state green' not in html


def test_initial_ui_ocr_profile_preserves_original_pixels_like_api():
    replacement="""def remember_profile(image,profile):
    st.session_state.setdefault('used_ocr_profiles',[]).append(profile)
    return image
with patch('quality_recovery.enhance_card',side_effect=remember_profile), patch('streamlit.file_uploader'"""
    script=SCRIPT.replace("with patch('streamlit.file_uploader'",replacement)
    app=AppTest.from_string(script,default_timeout=40)
    app.session_state['fixture_finding']=deepcopy(prepared(True).detection)
    app.run()
    assert not app.exception
    # The first call supplies OCR; later calls build the optional profile gallery.
    assert app.session_state['used_ocr_profiles'][0]=='raw'


def test_rejected_package_is_not_presented_as_a_card_crop_task():
    from classification import card_classification
    app=AppTest.from_string(SCRIPT,default_timeout=40)
    finding=deepcopy(prepared(False).detection)
    finding['intake_rejected']=True
    decision=dict(allowed=False,status='not_detected')
    decision['classification']=card_classification(finding,decision)
    app.session_state['fixture_finding']=finding
    app.session_state['fixture_decision']=decision
    app.run()
    assert not app.exception
    html=' '.join(m.value for m in app.markdown)
    assert 'وضعیت: رد' in html
    assert 'کارت ملی پیدا نشد' in html
    assert 'نوع کارت تأیید نشد' not in html
    assert not any(b.label in ('خواندن اطلاعات','بررسی کادر انتخابی') for b in app.button)
    override=next(t for t in app.toggle if t.label=='کارت در عکس هست؛ انتخاب دستی محدوده')
    override.set_value(True).run()
    assert not app.exception
    assert any(b.label=='بررسی کادر انتخابی' for b in app.button)
