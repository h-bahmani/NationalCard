from PIL import Image
import text_detector


def test_downward_shifted_expiry_never_uses_father_row(monkeypatch):
    # Realistic six-row layout shifted down in an otherwise aligned card.
    boxes=[[.55,.30,.74,.35],[.67,.42,.74,.48],[.66,.54,.75,.60],
           [.54,.65,.74,.705],[.64,.773,.73,.834],[.53,.869,.72,.923]]
    monkeypatch.setattr(text_detector,'detect_text',lambda image:[dict(box=b,score=.99) for b in boxes])
    regions,sources,_=text_detector.field_regions(Image.new('RGB',(800,500)),'smart')
    assert regions['expiry_date'][1]>.84
    assert regions['expiry_date'][3]>.923
    assert regions['father_name'][3]<regions['expiry_date'][1]
    assert regions['birth_date'][1]<.65 and regions['birth_date'][3]>.705
    assert sources['expiry_date']=='text_detector_ordered'


def test_missing_text_keeps_safe_template_fallback(monkeypatch):
    from pipeline import REGIONS
    monkeypatch.setattr(text_detector,'detect_text',lambda image:[])
    regions,sources,_=text_detector.field_regions(Image.new('RGB',(800,500)),'smart')
    assert regions==REGIONS['smart']
    assert set(sources.values())=={'template_fallback'}


def test_same_detected_line_cannot_fill_two_fields(monkeypatch):
    monkeypatch.setattr(text_detector,'detect_text',lambda image:[dict(box=[.54,.65,.74,.705],score=.99)])
    _,sources,_=text_detector.field_regions(Image.new('RGB',(800,500)),'smart')
    assert list(sources.values()).count('text_detector_ordered')==1
