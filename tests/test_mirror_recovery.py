from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageOps
import detector


@pytest.mark.parametrize('kind',['smart','old'])
def test_mirrored_reference_is_aligned_in_readable_orientation(kind):
    image=Image.open(Path(detector.ROOT)/f'reference_{kind}.jpg').convert('RGB')
    finding,aligned,_=detector.detect_card(ImageOps.mirror(image))
    assert finding['detected']
    assert finding.get('horizontal_mirror_corrected')
    assert finding['kind']==kind
    assert aligned.size==(800,500)
    assert np.mean(np.abs(np.asarray(aligned,dtype=float)-np.asarray(image,dtype=float)))<18


def test_mirror_quad_maps_back_to_original_pixels(monkeypatch):
    quad=[[10,5],[80,5],[80,40],[10,40]]
    monkeypatch.setattr(detector,'_detect_card_base',lambda *a,**k:({'detected':True,'card_quad':quad},None,None))
    finding,_,_=detector._detect_card_mirrored(Image.new('RGB',(101,51)))
    assert finding['card_quad']==[[90,5],[20,5],[20,40],[90,40]]


@pytest.mark.parametrize('width,height',[(378,320),(1000,1600),(320,542)])
def test_bounded_search_covers_a_small_card_at_left_center(width,height):
    boxes=detector._search_windows(width,height)
    card=(.08*width,.58*height,.32*width,.78*height)
    assert len(boxes)<=30
    assert any(l<=card[0] and t<=card[1] and r>=card[2] and b>=card[3] for l,t,r,b in boxes)
    assert all(0<=l<r<=width and 0<=t<b<=height for l,t,r,b in boxes)
