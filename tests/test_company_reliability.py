"""Decision regressions: no identity output on weak or contradictory evidence."""
from copy import deepcopy
import io

import pytest
from PIL import Image

from company_policy import build_company_result, selection_policy, _clip
from exporting import export_result
from images import load_image
from quality import quality_report
from validation import checksum_valid


def strong_result(kind='smart'):
    values = dict(national_id='0017864607', first_name='علی', last_name='احمدی',
                  father_name='رضا', birth_date='1370/01/01', document_number='12345')
    assert checksum_valid(values['national_id'])
    raw = {'fields': {}}
    for name, value in values.items():
        raw['fields'][name] = dict(text=value, raw_selected_text=value, ambiguous=False,
            decision='accepted', accepted_text=value,
            engine_evidence={e: dict(value=value, mean_score=.97, stable=True)
                             for e in ('bina_rizeh_ppocrv6', 'paddleocr_v5_arabic')})
    out = export_result(raw, kind)
    out.update(detection=dict(detected=True, kind=kind, method='layout_match',
                              inliers=30, matches=35, reference_coverage=.4),
               decision={'quality': dict(laplacian_variance=150, contrast_std=50,
                                        brightness=145, processed_size=[800, 500])})
    return out


@pytest.mark.parametrize('kind', ['smart', 'old'])
def test_complete_consensus_can_be_accepted(kind):
    out = build_company_result(strong_result(kind))
    assert out['status'] == 'accepted'
    assert out['next_action']['type'] == 'none'
    assert out['fields']['national_id']['value'] == '0017864607'
    assert ('birth_date' in out['policy']['required_fields']) == (kind == 'smart')


@pytest.mark.parametrize('name', ['national_id', 'first_name', 'last_name', 'father_name', 'birth_date'])
@pytest.mark.parametrize('failure', ['one_engine', 'low_scores', 'ambiguous', 'no_evidence', 'disagreement', 'unstable'])
def test_each_weak_required_field_blocks_auto_accept(name, failure):
    result = strong_result()
    field = result['raw_result']['fields'][name]
    if failure == 'one_engine':
        field['engine_evidence'].pop('paddleocr_v5_arabic')
    elif failure == 'low_scores':
        for row in field['engine_evidence'].values():
            row['mean_score'] = .2
    elif failure == 'ambiguous':
        field['ambiguous'] = True
    elif failure == 'no_evidence':
        field['engine_evidence'] = {}
    elif failure == 'unstable':
        for row in field['engine_evidence'].values():
            row['stable'] = False
    else:
        field['engine_evidence']['paddleocr_v5_arabic']['value'] = 'different'
    out = build_company_result(result)
    assert out['status'] == 'review'
    assert out['fields'][name]['value'] is None
    assert name in out['next_action']['fields']


def test_invalid_checksum_never_exported_even_if_legacy_value_exists():
    result = strong_result()
    result['national_id'] = '1234567890'
    out = build_company_result(result)
    assert out['fields']['national_id']['value'] is None
    assert out['status'] == 'review'


def test_quality_low_requests_new_image_even_with_good_ocr():
    result = strong_result()
    result['decision']['quality'].update(laplacian_variance=0, contrast_std=0, brightness=0)
    out = build_company_result(result)
    assert out['status'] == 'review'
    assert out['next_action']['type'] == 'replace_image'
    assert 'image_quality_low' in out['reasons']


def test_small_source_cannot_gain_quality_by_interpolation():
    stream = io.BytesIO()
    Image.new('RGB', (160, 100), 'white').save(stream, 'PNG')
    image = load_image(stream.getvalue())
    assert image.height == 320
    report = quality_report(image)
    assert report['original_size'] == [160, 100]
    result = strong_result()
    result['decision']['source_quality'] = report
    out = build_company_result(result)
    assert out['image_quality']['low_source_resolution']
    assert out['status'] == 'review'


def test_detection_threshold_opens_selection_and_blocks_accept():
    result = strong_result()
    result['detection'].update(inliers=10, matches=20, reference_coverage=.15)
    out = build_company_result(result)
    assert out['next_action']['selection_required']
    assert out['next_action']['type'] == 'select_card_region'
    assert out['status'] == 'review'


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 0, 1.1])
def test_bad_threshold_rejected(value):
    with pytest.raises(ValueError):
        build_company_result(strong_result(), min_confidence=value)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), None, 'invalid'])
def test_nonfinite_confidence_never_becomes_high(value):
    assert _clip(value) == 0


def test_manual_edit_invalidates_company_machine_acceptance():
    from review import apply_review
    result = strong_result()
    result['company'] = build_company_result(result)
    raw_before = deepcopy(result['raw_result'])
    reviewed = apply_review(result, {'national_id': '0017864607', 'first_name': 'حسین'})
    assert reviewed['company']['status'] == 'review'
    assert reviewed['company']['fields']['first_name']['value'] is None
    assert reviewed['company']['fields']['first_name']['display_value'] == 'حسین'
    assert reviewed['raw_result'] == raw_before


def test_unknown_detection_always_requests_selection():
    assert selection_policy({'detected': False})['required']


def test_an_explicit_crop_does_not_enter_a_selection_loop():
    assert not selection_policy({'detected':False,'user_selected':True})['required']
    result=strong_result()
    result['detection'].update(detected=False,user_selected=True)
    out=build_company_result(result)
    assert out['status']=='review'
    assert out['document']['type'] is None
    assert out['next_action']['type']=='replace_image'
    assert all(f['value'] is None for f in out['fields'].values())


def test_api_returns_selection_before_ocr_when_card_confidence_is_low(monkeypatch):
    import cardreader_module
    import pipeline
    image=Image.new('RGB',(800,500))
    prepared=cardreader_module.PreparedCard(image,
        {'detected':True,'kind':'smart','method':'layout_match'},
        {'allowed':True,'status':'accepted'},image,None)
    monkeypatch.setattr(cardreader_module,'prepare',lambda *a,**k:prepared)
    monkeypatch.setattr(pipeline,'extract',lambda *a,**k:pytest.fail('must request crop before OCR'))
    stream=io.BytesIO();image.save(stream,'PNG')
    result=cardreader_module.CardReader().analyze(stream.getvalue())
    assert result['company']['next_action']['type']=='select_card_region'
    assert result['company']['review_required']


def test_manual_verification_retains_geometric_confidence():
    finding = strong_result()['detection']
    assert not selection_policy(dict(detected=True, user_selected=True, method='user_selection_then_layout_match',
                                     selection_detection=finding))['required']


@pytest.mark.parametrize('failure,action', [('engine', 'service_setup_required'), ('worker', 'retry_processing')])
def test_engine_failure_is_not_reported_as_bad_photo(failure, action):
    result = strong_result()
    if failure == 'engine':
        result['raw_result']['fields']['first_name']['ocr_engine_errors'] = [{'engine': 'bina'}]
    else:
        result['raw_result']['errors'] = [{'stage': 'local', 'code': 'timeout'}]
    out = build_company_result(result)
    assert out['status'] == 'review'
    assert out['next_action']['type'] == action
    assert out['confidence']['percent'] < 50
    assert all(f['value'] is None for f in out['fields'].values())


def test_tiny_card_in_large_image_cannot_pass_as_high_resolution():
    from quality import attach_quality
    result = strong_result()
    image = Image.new('RGB', (2400, 1600), 'white')
    result['detection']['card_quad'] = [[300,300],[460,300],[460,400],[300,400]]
    attach_quality(result['decision'], image, Image.new('RGB', (800,500)), result['detection'])
    assert result['decision']['quality']['native_card_size'] == [160,100]
    assert build_company_result(result)['image_quality']['low_source_resolution']


def test_temp_access_failure_is_controlled(monkeypatch):
    import pipeline
    def denied(*args, **kwargs):
        raise PermissionError('temporary folder denied')
    monkeypatch.setattr(pipeline.tempfile, 'TemporaryDirectory', denied)
    result = pipeline.local_read(Image.new('RGB', (800, 500)), 'smart', regions={})
    assert result['ok'] is False
    assert result['error'] == 'worker_failed'


@pytest.mark.parametrize('data', [b'', b'not an image', b'\x89PNG\r\n\x1a\ntruncated'])
def test_corrupt_upload_has_clear_validation_error(data):
    with pytest.raises(ValueError):
        load_image(data)
