"""Public local module: consistent image intake, extraction and JSON contract."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from PIL import Image
from pathlib import Path
from settings import apply_runtime_env
apply_runtime_env(Path(__file__).resolve().parent)
from detector import detect_card,number_crop
from intake import assess_image
from images import load_image
import pipeline
from exporting import export_result
from presentation import number_display

@dataclass
class PreparedCard:
    image:Image.Image
    detection:dict
    decision:dict
    aligned:Image.Image|None
    roi:Image.Image|None

def apply_ocr_gate(card:PreparedCard,selection_threshold=0.85)->PreparedCard:
    """One pre-OCR decision for UI, API and direct module callers; never mutate cached intake."""
    from company_policy import selection_policy
    if card.detection.get('intake_rejected') and not card.detection.get('detected'):
        return PreparedCard(card.image,card.detection,{**card.decision,'allowed':False},card.aligned,card.roi)
    if selection_policy(card.detection,selection_threshold)['required'] and card.decision.get('allowed'):
        decision={**card.decision,'allowed':False,'status':'selection_required',
                  'title':'محدودهٔ کارت نیاز به بررسی دارد',
                  'message':'تأیید خودکار کافی نیست؛ اگر کارت ملی در عکس است، محدودهٔ کامل آن را انتخاب کن.'}
        return PreparedCard(card.image,card.detection,decision,card.aligned,card.roi)
    return card

def prepare(image:Image.Image, progress=None, search_budget=60.0)->PreparedCard:
    detection,aligned,roi=detect_card(
        image,
        progress=progress,
        search_budget=search_budget
    )
    decision=assess_image(image,detection,aligned)
    if decision['status']=='not_detected':
        from detector import suggest_card_crop
        _,quad=suggest_card_crop(image)
        if quad:
            xs=[point[0]/image.width for point in quad];ys=[point[1]/image.height for point in quad]
            detection['geometric_candidate']=[max(0,min(xs)),max(0,min(ys)),min(1,max(xs)),min(1,max(ys))]
            decision['message']='یک محدوده شبیه کارت پیدا کردم، اما ملی‌بودن آن مشخص نیست. کادر را بررسی کن؛ اگر درست است، خواندن از این محدوده را انتخاب کن.'
        # Last-resort OCR path for a tightly cropped card photo. A failed verifier is
        # never upgraded to a positive card detection; it only unlocks review-only OCR.
        # This specifically prevents a blurry/angled card filling the frame from being
        # blocked before the number line can be read.
        from card_verifier import verify_reference_layout
        from pipeline import REGIONS
        from regions import number_region
        aspect=max(image.size)/max(1,min(image.size))
        if min(image.size)>=180 and 1.35<=aspect<=1.90:
            candidates=[]
            for angle in (0,90,180,270):
                view=image.rotate(angle,expand=True) if angle else image
                view_aspect=max(view.size)/max(1,min(view.size))
                if not 1.35<=view_aspect<=1.90:continue
                try:
                    diag,verified_aligned=verify_reference_layout(view,context='manual')
                except Exception:
                    continue
                score=float(diag.get('score') or 0)
                inliers=int(diag.get('inliers') or 0)
                if score>=.15 and inliers>=6:
                    kind=diag.get('kind') if diag.get('kind') in REGIONS else 'smart'
                    candidates.append((score, inliers, angle, view, verified_aligned, diag, kind))
            if candidates:
                # Without a valid projective alignment, a weak appearance score cannot
                # reliably tell 0° from 90°. Preserve the uploaded orientation in that
                # case; the operator can still rotate manually if needed.
                geometrically_aligned=[row for row in candidates if row[4] is not None and any(alt.get('valid_geometry') for alt in (row[5].get('alternatives') or []))]
                if geometrically_aligned:
                    score,inliers,angle,view,verified_aligned,diag,kind=max(geometrically_aligned,key=lambda row:(row[0],row[1]))
                    aligned=verified_aligned
                else:
                    score,inliers,angle,view,verified_aligned,diag,kind=next((row for row in candidates if row[2]==0),max(candidates,key=lambda row:(row[0],row[1])))
                    aligned=view.resize((800,500),Image.Resampling.LANCZOS)
                detection.update({'detected':False,'ocr_candidate':True,'kind':kind,
                    'label':'کادر احتمالی کارت ملی','method':'full_frame_card_candidate',
                    'manual_verification':'unverified','reference_verification':{k:v for k,v in diag.items() if k!='alternatives'},
                    'candidate_rotation_degrees':angle,'card_quad':[[0,0],[image.width-1,0],[image.width-1,image.height-1],[0,image.height-1]],
                    'number_box':number_region(aligned,kind)[1], 'field_boxes':dict(REGIONS[kind])})
                decision.update({'status':'candidate','allowed':False,'title':'نوع کارت تأیید نشد',
                    'message':'این کادر فقط یک پیشنهاد است. اگر کارت ملی در عکس است، محدودهٔ کامل آن را انتخاب کن.',
                    'evidence':{**(decision.get('evidence') or {}),'ocr_candidate':True,'card_type_verified':False}})
    if decision['allowed'] and aligned is not None:
        try:
            from text_detector import field_regions
            from quality_recovery import enhance_card
            regions,sources,boxes=field_regions(enhance_card(aligned,'auto'),detection['kind'])
            # National ID is high-risk: keep the validated template/number-region box.
            # Text localization is allowed to help the other fields but must never
            # shrink or vertically shift the number line (especially leading zeros).
            base=detection['number_box']
            regions['national_id']=base
            sources['national_id']='validated_number_region'
            detection.update(field_boxes=regions,field_box_sources=sources,text_boxes=boxes,number_box=regions['national_id'])
        except Exception as error:
            # A localization failure must not disable the existing OCR path.
            detection['text_detection_error']=type(error).__name__
    from classification import card_classification
    from quality import attach_quality
    attach_quality(decision,image,aligned,detection)
    decision['classification']=card_classification(detection,decision)
    if decision['classification']['status']=='rejected':
        detection['intake_rejected']=True
        decision.update(status='not_detected',allowed=False,
                        title=decision['classification']['title'],message=decision['classification']['message'])
    return PreparedCard(image,detection,decision,aligned,roi)

def prepare_manual(image:Image.Image,box,kind=None)->PreparedCard:
    """Re-check an explicitly selected card region before OCR.

    Manual selection tells us *where* to look, not what the document is.  The selected
    crop is therefore verified again with the strict layout matcher and the blur-tolerant
    masked reference verifier.  OCR can still be offered when verification is only
    probable/unknown, but the UI must not call that a confirmed national card.
    """
    if kind not in (None,'smart','old'):raise ValueError('Unknown card layout.')
    if len(box)!=4 or not all(__import__('math').isfinite(v) for v in box):raise ValueError('Invalid card crop.')
    l,t,r,b=box
    if not 0<=l<r<=1 or not 0<=t<b<=1 or min((r-l)*image.width,(b-t)*image.height)<40:
        raise ValueError('Select a card crop at least 40 pixels wide and high.')
    left,top,right,bottom=round(l*image.width),round(t*image.height),round(r*image.width),round(b*image.height)
    selected=image.crop((left,top,right,bottom))

    from detector import _detect_card_base,suggest_card_crop,_map_rectified_quad_to_source,TYPES
    from card_verifier import verify_reference_layout
    from regions import number_region

    finding,matched,_=_detect_card_base(selected)
    if not finding.get('detected'):
        from detector import _detect_card_mirrored
        mirrored,mirrored_aligned,_=_detect_card_mirrored(selected)
        if mirrored is not None:
            finding,matched=mirrored,mirrored_aligned

    # A manually selected card can be perfectly valid yet only ~100 px wide.
    # Upscale *the selected crop only* before feature matching so the person/background
    # no longer dominates and SIFT has a usable working scale.  Coordinates are mapped
    # back to the original selected crop; no synthetic OCR value is created here.
    manual_zoom_factor=1.0
    if not finding.get('detected') and min(selected.size) < 240:
        manual_zoom_factor=min(4.0,max(1.0,300.0/max(1,min(selected.size))))
        if manual_zoom_factor>1.05:
            zoomed=selected.resize((round(selected.width*manual_zoom_factor),round(selected.height*manual_zoom_factor)),Image.Resampling.LANCZOS)
            zoom_finding,zoom_aligned,_=_detect_card_base(zoomed,_recovery=False)
            if not zoom_finding.get('detected'):
                from detector import _detect_card_mirrored
                mirrored,mirrored_aligned,_=_detect_card_mirrored(zoomed,_recovery=False)
                if mirrored is not None:
                    zoom_finding,zoom_aligned=mirrored,mirrored_aligned
            if zoom_finding.get('detected'):
                if zoom_finding.get('card_quad'):
                    q=__import__('numpy').asarray(zoom_finding['card_quad'],dtype='float32')/manual_zoom_factor
                    zoom_finding['card_quad']=q.tolist()
                zoom_finding['manual_zoom_factor']=round(float(manual_zoom_factor),3)
                finding,matched=zoom_finding,zoom_aligned

    if not finding.get('detected'):
        from detector import _detect_card_rotated_local
        rotated_finding,rotated_aligned,_=_detect_card_rotated_local(selected)
        if rotated_finding is not None:
            finding,matched=rotated_finding,rotated_aligned
    aligned=matched
    rectified_by_geometry=False
    ref_diag=None
    global_quad=None

    if finding.get('detected') and matched is not None:
        chosen_kind=finding['kind']
        global_quad=(__import__('numpy').asarray(finding['card_quad'],dtype='float32')+__import__('numpy').float32([left,top])).tolist()
    else:
        chosen_kind=None
        rectified,local_quad=suggest_card_crop(selected)
        if rectified is not None:
            rectified_by_geometry=True
            retry,retry_aligned,_=_detect_card_base(rectified)
            if retry.get('detected') and retry_aligned is not None:
                finding=retry;aligned=retry_aligned;chosen_kind=retry['kind']
                mapped=_map_rectified_quad_to_source(retry.get('card_quad',[[0,0],[799,0],[799,499],[0,499]]),local_quad)
                global_quad=(mapped+__import__('numpy').float32([left,top])).tolist()
            else:
                ref_diag,ref_aligned=verify_reference_layout(rectified,context='manual')
                aligned=ref_aligned or rectified
                if ref_diag.get('verified') or ref_diag.get('probable'):
                    chosen_kind=ref_diag.get('kind')
                    local_ref_quad=ref_diag.get('quad') or [[0,0],[799,0],[799,499],[0,499]]
                    mapped=_map_rectified_quad_to_source(local_ref_quad,local_quad)
                    global_quad=(mapped+__import__('numpy').float32([left,top])).tolist()
        if chosen_kind is None:
            ref_diag,ref_aligned=verify_reference_layout(selected,context='manual')
            if ref_aligned is not None:
                aligned=ref_aligned
            elif aligned is None:
                aligned=selected.resize((800,500),Image.Resampling.LANCZOS)
            if ref_diag.get('verified') or ref_diag.get('probable'):
                chosen_kind=ref_diag.get('kind')
                if ref_diag.get('quad'):
                    global_quad=(__import__('numpy').asarray(ref_diag['quad'],dtype='float32')+__import__('numpy').float32([left,top])).tolist()

    if chosen_kind is None and ref_diag and float(ref_diag.get('score') or 0)>=.15 and int(ref_diag.get('inliers') or 0)>=6:
        # Best template is only a crop-layout hint; it does not change the unverified status.
        chosen_kind=ref_diag.get('kind')
    chosen_kind=chosen_kind or kind or 'smart'
    if aligned is None:
        aligned=selected.resize((800,500),Image.Resampling.LANCZOS)
    if global_quad is None:
        global_quad=[[left,top],[right,top],[right,bottom],[left,bottom]]

    strict_verified=bool(finding.get('detected'))
    reference_verified=bool(ref_diag and ref_diag.get('verified'))
    reference_probable=bool(ref_diag and ref_diag.get('probable'))
    verified=strict_verified or reference_verified

    from pipeline import REGIONS
    regions=dict(REGIONS[chosen_kind]);sources={k:'template_fallback' for k in regions};boxes=[]
    try:
        from text_detector import field_regions
        from quality_recovery import enhance_card
        regions,sources,boxes=field_regions(enhance_card(aligned,'auto'),chosen_kind)
        # Never replace the high-risk number box with a narrower text-detector box.
        _,validated_number_box=number_region(aligned,chosen_kind)
        regions['national_id']=validated_number_box
        sources['national_id']='validated_number_region'
    except Exception:
        _,validated_number_box=number_region(aligned,chosen_kind)
        regions['national_id']=validated_number_box
        sources['national_id']='validated_number_region'

    detection={'detected':verified,'kind':chosen_kind,'label':TYPES[chosen_kind],
        'method':('user_selection_then_layout_match' if strict_verified else
                  'user_selection_reference_verify' if reference_verified else 'user_selection'),
        'user_selected':True,'manual_geometry_rectified':rectified_by_geometry,
        'horizontal_mirror_corrected':bool(finding.get('horizontal_mirror_corrected')),
        'manual_zoom_factor':round(float(manual_zoom_factor),3),
        'manual_verification':'verified' if verified else 'probable' if reference_probable else 'unverified',
        'reference_probable':reference_probable,
        'suspected_kind':(ref_diag or {}).get('kind') if ref_diag else finding.get('suspected_kind'),
        'reference_verification':ref_diag,
        'card_quad':global_quad,
        'number_box':regions['national_id'],'field_boxes':regions,'field_box_sources':sources,'text_boxes':boxes}
    detection['selection_detection']={**finding}
    from detector import mark_multiple_cards
    selected_finding={**detection,'card_quad':(__import__('numpy').asarray(global_quad,dtype='float32')-__import__('numpy').float32([left,top])).tolist()}
    mark_multiple_cards(selected,selected_finding)
    if selected_finding.get('multiple_cards'):
        detection['multiple_cards']=True
        detection['card_candidates']=[{**row,'quad':(__import__('numpy').asarray(row['quad'])+__import__('numpy').float32([left,top])).tolist()}
            for row in selected_finding['card_candidates']]
        for row in detection['card_candidates']:
            xs=[p[0]/image.width for p in row['quad']];ys=[p[1]/image.height for p in row['quad']]
            row['box']=[max(0,min(xs)),max(0,min(ys)),min(1,max(xs)),min(1,max(ys))]
    if ref_diag:
        detection['selection_detection'].update({
            'reference_verified':reference_verified,'reference_probable':reference_probable,
            'reference_score':ref_diag.get('score'),'suspected_kind':ref_diag.get('kind')})
        if reference_verified:
            detection['selection_detection'].update({'detected':True,'kind':chosen_kind,'method':'reference_verify'})

    decision={'status':'manual','allowed':True,'title':'محدودهٔ کارت بررسی شد',
        'message':'OCR فقط روی محدودهٔ انتخاب‌شده اجرا می‌شود.',
        'evidence':{'user_selected':True,'authenticity_checked':False,'manual_verification':detection['manual_verification']}}
    if detection.get('multiple_cards'):
        decision.update(status='selection_required',allowed=False,message='فقط محدودهٔ یک کارت را انتخاب کن.')
    from quality import attach_quality
    attach_quality(decision,image,aligned,detection)
    from classification import card_classification
    decision['classification']=card_classification(detection,decision)
    decision['title']=decision['classification']['title']
    decision['message']=decision['classification']['message']
    return PreparedCard(image,detection,decision,aligned,number_crop(aligned,chosen_kind))

def read_prepared(card:PreparedCard,number_box=None,field_boxes=None,
                  min_confidence=0.90,selection_threshold=0.85,allow_review_ocr=False,**options)->dict[str,Any]:
    from company_policy import _threshold
    min_confidence=_threshold(min_confidence)
    selection_threshold=_threshold(selection_threshold)
    if allow_review_ocr and card.aligned is not None and card.detection.get('detected') and not card.detection.get('multiple_cards'):
        # API/company mode: low card-confidence should trigger REVIEW, not suppress OCR.
        # Keep the original decision/detection evidence so company_policy can still
        # report card_confidence_low and review_required.
        pass
    else:
        card=apply_ocr_gate(card,selection_threshold)
    if not card.decision['allowed'] or card.aligned is None:
        payload={'schema_version':'1.0','status':card.decision['status'],
                'decision':card.decision,'detection':card.detection,
                'national_id':None,'fields':{},'suggestions':{},'extraction_boxes':{},'number_preview':None,
                'review_required':True,'authenticity_checked':False}
        from company_policy import build_company_result
        payload['company']=build_company_result(payload,min_confidence=min_confidence,selection_threshold=selection_threshold)
        return payload
    kind=card.detection['kind'];box=number_box or card.detection.get('number_box')
    if box:_validate_box(box)
    for field_name,field_box in (field_boxes or {}).items():
        _validate_field_box(field_name,field_box,kind)
    if box:
        w,h=card.aligned.size
        roi=card.aligned.crop(tuple(round(v*(w if i%2==0 else h)) for i,v in enumerate(box)))
    else:roi=number_crop(card.aligned,kind)
    resolved={**card.detection.get('field_boxes',{}),**(field_boxes or {})}
    resolved.pop('national_id',None)
    result=pipeline.extract(card.aligned,roi,kind,number_box=box,field_boxes=resolved,
                            protected_fields=set(field_boxes or {}),**options)
    shown=number_display(result);export=export_result(result,kind)
    clear=shown['status'] in ('cloud','candidate')
    status='extracted' if clear else 'partial'
    if any(e.get('stage')=='local' for e in result.get('errors',[])) and not clear:status='processing_error'
    from images import fingerprint
    original_box=card.detection.get('number_box') or pipeline.REGIONS[kind]['national_id']
    changed_number=number_box is not None and tuple(number_box)!=tuple(original_box)
    payload={**export,'schema_version':'1.0','status':status,'image_fingerprint':fingerprint(card.image),
            'decision':card.decision,'detection':card.detection,
            'extraction_boxes':{**pipeline.REGIONS[kind],**resolved,'national_id':box or pipeline.REGIONS[kind]['national_id']},
            'field_overrides':{**(field_boxes or {}),**({'national_id':number_box} if changed_number else {})},
            'number_preview':shown['value'],'review_required':True,
            'authenticity_checked':False}
    from company_policy import build_company_result
    payload['company']=build_company_result(payload,min_confidence=min_confidence,selection_threshold=selection_threshold)
    payload['review_required']=payload['company']['review_required']
    return payload

class CardReader:
    """Embedding entry point. Bytes in, serializable results out. Offline by default."""
    def analyze(self,image_bytes:bytes,frame:int=0,number_box=None,field_boxes=None,card_box=None,kind='smart',
                min_confidence=0.90,selection_threshold=0.85,allow_review_ocr=False)->dict[str,Any]:
        from company_policy import _threshold
        _threshold(min_confidence);_threshold(selection_threshold)
        if number_box is not None:_validate_box(number_box)
        for box in (field_boxes or {}).values():_validate_box(box)
        from presentation import FIELD_NAMES
        if set(field_boxes or {})-set(FIELD_NAMES):raise ValueError('Unknown field override.')
        image=load_image(image_bytes,frame=frame)
        prepared=prepare_manual(image,card_box,kind) if card_box is not None else prepare(
        image,
        search_budget=12.0
    )   
        return read_prepared(prepared,
                             number_box=number_box,field_boxes=field_boxes,cloud=False,key='',
                             min_confidence=min_confidence,selection_threshold=selection_threshold,
                             allow_review_ocr=allow_review_ocr)

def _validate_box(box):
    import math
    if len(box)!=4 or not all(math.isfinite(v) for v in box):raise ValueError('Invalid crop coordinates.')
    l,t,r,b=box
    if not 0<=l<r<=1 or not 0<=t<b<=1 or (r-l)*800<8 or (b-t)*500<8:
        raise ValueError('Crop must be positive, inside [0,1], and at least 8 pixels.')


def _validate_field_box(name,box,kind):
    _validate_box(box)
    from field_geometry import check_field_assignment
    check_field_assignment(name,box,kind)

if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description='Local card OCR module; JSON to stdout.')
    p.add_argument('image');p.add_argument('--output');a=p.parse_args()
    result=CardReader().analyze(Path(a.image).read_bytes())
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output:Path(a.output).write_text(text,encoding='utf-8')
    else:print(text)
