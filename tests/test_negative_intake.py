import pytest
from classification import card_classification
from company_policy import build_company_result,selection_policy
from PIL import Image
import cardreader_module as module


@pytest.mark.parametrize('extra',[{}, {'search_limited':True}])
def test_weak_rectangle_is_not_identity_card_evidence(extra):
    finding=dict(detected=False,kind='old',ocr_candidate=True,
                 reference_verification=dict(score=.2,inliers=8,probable=False),**extra)
    assert card_classification(finding,{'evidence':{}})['status']=='rejected'


@pytest.mark.parametrize('finding,decision',[
    ({'partial_card':True},{}),
    ({'user_selected':True},{}),
    ({'yolo_presence':True},{}),
    ({},{'evidence':{'large_face_count':1}}),
    ({'reference_verification':{'probable':True,'appearance':.5}},{}),
])
def test_credible_card_or_person_scene_retains_manual_recovery(finding,decision):
    assert card_classification({'detected':False,**finding},decision)['status']=='review'


def test_verified_card_wins_over_barcode_presence():
    finding=dict(detected=True,kind='old',inliers=40,matches=45,reference_coverage=.3)
    assert card_classification(finding,{'evidence':{'barcode_regions':[[0,0,300,50]]}})['status']=='accepted'


def test_rejected_intake_returns_replace_image_without_field_crop_tasks(monkeypatch):
    finding=dict(detected=False,kind='old',ocr_candidate=True,intake_rejected=True)
    card=module.PreparedCard(Image.new('RGB',(800,500)),finding,{'allowed':True,'status':'candidate'},Image.new('RGB',(800,500)),None)
    monkeypatch.setattr(module.pipeline,'extract',lambda *a,**k:pytest.fail('rejected image reached OCR'))
    result=module.read_prepared(card)['company']
    assert result['status']=='rejected'
    assert result['next_action']['type']=='replace_image'
    assert not result['next_action']['selection_required']
    assert result['next_action']['fields']==[]
    assert result['document']['type'] is None
    assert all(f['value'] is None for f in result['fields'].values())
