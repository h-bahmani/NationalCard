import numpy as np
from PIL import Image
import detector


def test_old_small_crop_pixels_are_preserved():
    assert detector._search_windows(439,514).index((136,370,302,493))<24


def test_native_windows_finish_before_mirror_retries(monkeypatch):
    calls=[]
    image=Image.new('RGB',(439,514))
    def crop(im,box,**kwargs):
        calls.append((box,kwargs['_mirror_retry']))
        if box==(136,370,302,493):
            return {'detected':True,'kind':'smart'},image,None
        return None,None,None
    monkeypatch.setattr(detector,'_detect_zoomed_crop',crop)
    monkeypatch.setattr(detector,'verify_reference_layout',lambda *a,**k:({'verified':False},None))
    finding,_,_,_=detector._multiscale_layout_retry(image)
    assert finding['detected']
    assert all(not mirror for _,mirror in calls)


def test_weak_local_hint_does_not_skip_rotation(monkeypatch):
    image=Image.new('RGB',(700,700))
    miss={'detected':False,'kind':None,'method':'layout_match'}
    monkeypatch.setattr(detector,'_detect_card_base',lambda *a,**k:(miss.copy(),None,None))
    monkeypatch.setattr(detector,'_detect_card_mirrored',lambda *a,**k:(None,None,None))
    monkeypatch.setattr('intake.large_faces',lambda im:[[0,0,400,400]])
    hint=dict(detected=False,kind='smart',card_quad=[[0,0],[99,0],[99,59],[0,59]],
              reference_verification={'score':.4,'kind':'smart'})
    monkeypatch.setattr(detector,'_multiscale_layout_retry',lambda *a,**k:(hint,image,None,{'attempted':True}))
    monkeypatch.setattr(detector,'_color_card_candidates',lambda *a,**k:[])
    monkeypatch.setattr(detector,'_geometric_card_candidates',lambda *a,**k:[])
    rotated=dict(detected=True,kind='smart',card_quad=[[0,0],[699,0],[699,699],[0,699]])
    monkeypatch.setattr(detector,'_detect_card_rotated_local',lambda *a,**k:(rotated,image,None))
    finding,_,_=detector._detect_card_single(image)
    assert finding['detected']
    assert finding['rotation_retry']
