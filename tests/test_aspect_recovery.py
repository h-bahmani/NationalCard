"""Aspect correction must recover geometry, not accept arbitrary rectangles."""
from pathlib import Path
import numpy as np
from PIL import Image
import detector


def test_squashed_reference_recovers_card_in_original_coordinates():
    source = Image.open(Path(detector.ROOT)/'reference_smart.jpg').convert('RGB')
    image = source.resize((200,250),Image.Resampling.LANCZOS)
    finding,aligned,roi = detector._detect_stretched_closeup(image)
    assert finding is not None and finding['detected']
    assert finding['kind'] == 'smart'
    assert finding['reference_verification']['verified']
    assert finding['aspect_ratio_corrected']
    assert aligned.size == (800,500) and roi is not None
    points = np.asarray(finding['card_quad'])
    assert np.ptp(points[:,0]) > .8*image.width
    assert np.ptp(points[:,1]) > .8*image.height
    assert points[:,0].max() < 1.15*image.width
    assert points[:,1].max() < 1.15*image.height


def test_blank_portrait_is_not_a_card():
    finding,aligned,roi = detector._detect_stretched_closeup(Image.new('RGB',(200,250),'white'))
    assert finding is None and aligned is None and roi is None


def test_primary_match_without_masked_confirmation_is_rejected(monkeypatch):
    finding = dict(detected=True,kind='smart',matches=40,inliers=35,reference_coverage=.5,
                   card_quad=[[0,0],[799,0],[799,499],[0,499]])
    monkeypatch.setattr(detector,'_detect_card_base',lambda *a,**k:(finding.copy(),None,None))
    monkeypatch.setattr(detector,'verify_reference_layout',lambda *a,**k:({'verified':False},None))
    assert detector._detect_stretched_closeup(Image.new('RGB',(200,250)))[0] is None
