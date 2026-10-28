from company_policy import detection_confidence, build_company_result


def test_verified_detection_has_operational_confidence():
    row = detection_confidence({
        'detected': True, 'method': 'layout_match',
        'inliers': 14, 'matches': 18, 'reference_coverage': .35,
    })
    assert row['state'] == 'verified'
    assert row['percent'] >= 70
    assert row['calibrated'] is False


def test_not_detected_is_rejected_company_result():
    payload = {
        'card_type': None,
        'detection': {'detected': False, 'method': 'layout_match'},
        'decision': {'quality': {}},
        'national_id': None,
        'fields': {},
        'raw_result': {'fields': {}},
    }
    out = build_company_result(payload)
    assert out['status'] == 'rejected'
    assert out['review_required'] is True
    assert out['document']['confidence']['calibrated'] is False


def test_manual_unverified_is_review_not_accept():
    payload = {
        'card_type': 'smart',
        'detection': {'detected': False, 'user_selected': True, 'method': 'user_selection'},
        'decision': {'quality': {}},
        'national_id': None,
        'fields': {},
        'raw_result': {'fields': {}},
    }
    out = build_company_result(payload)
    assert out['status'] == 'review'
    assert out['review_required'] is True
