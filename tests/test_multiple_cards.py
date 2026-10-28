import numpy as np
from PIL import Image

from company_policy import selection_policy, build_company_result
from intake import assess_image
import detector


def test_second_disjoint_verified_card_requires_selection(monkeypatch):
    image = Image.new('RGB', (1000,700), 'white')
    first = dict(detected=True,kind='smart',card_quad=[[10,10],[410,10],[410,260],[10,260]])
    second = dict(detected=True,kind='smart',card_quad=[[510,350],[910,350],[910,600],[510,600]])
    monkeypatch.setattr(detector,'_detect_card_base',lambda *a, **k: (second,None,None))
    detector.mark_multiple_cards(image,first)
    assert first['multiple_cards']
    assert len(first['card_candidates']) == 2
    assert selection_policy(first)['reason'] == 'multiple_cards'
    assert assess_image(image,first)['allowed'] is False


def test_overlapping_match_is_not_a_second_card(monkeypatch):
    image = Image.new('RGB', (1000,700), 'white')
    first = dict(detected=True,kind='smart',card_quad=[[10,10],[410,10],[410,260],[10,260]])
    monkeypatch.setattr(detector,'_detect_card_base',lambda *a, **k: (first.copy(),None,None))
    detector.mark_multiple_cards(image,first)
    assert not first.get('multiple_cards')


def test_multiple_cards_never_release_identity_values():
    from test_company_reliability import strong_result
    payload = strong_result()
    payload['detection']['multiple_cards'] = True
    result = build_company_result(payload)
    assert result['status'] == 'review'
    assert result['next_action']['type'] == 'select_card_region'
    assert 'multiple_cards' in result['reasons']
    assert all(f['value'] is None for f in result['fields'].values())


def test_manual_crop_only_checks_inside_selected_region(monkeypatch):
    import cardreader_module
    import text_detector
    image = Image.new('RGB',(1000,700),'white')
    finding = dict(detected=True,kind='smart',card_quad=[[0,0],[399,0],[399,279],[0,279]])
    monkeypatch.setattr(detector,'_detect_card_base',lambda *a,**k:(finding,Image.new('RGB',(800,500)),None))
    def no_text(*args,**kwargs):
        raise RuntimeError('use templates')
    monkeypatch.setattr(text_detector,'field_regions',no_text)
    seen = []
    def check(selected, detection):
        seen.append((selected.size, np.asarray(detection['card_quad'])))
        return detection
    monkeypatch.setattr(detector,'mark_multiple_cards',check)
    card = cardreader_module.prepare_manual(image,[.1,.1,.5,.5])
    assert seen[0][0] == (400,280)
    assert np.allclose(seen[0][1],finding['card_quad'])
    assert card.decision['allowed']
