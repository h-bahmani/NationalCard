"""Local layout matching for two Iranian national-card fronts. Not authenticity detection."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np
import time
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parent/'templates'
TYPES={'smart':'کارت ملی هوشمند', 'old':'کارت ملی قدیمی'}
# Only the printed number, never birth/expiry date; fractions in aligned card coordinates.
from regions import NUMBER_SEARCH,LEGACY_NUMBER_ROIS,number_region
from card_verifier import verify_reference_layout
NUMBER_ROIS=LEGACY_NUMBER_ROIS


@lru_cache(maxsize=2)
def _layout_reference(kind):
    with np.load(ROOT/(kind+'.npz'), allow_pickle=False) as data:
        points = data['points'].copy()
        descriptors = data['descriptors'].copy()
    points.setflags(write=False)
    descriptors.setflags(write=False)
    return points, descriptors


def gray_image(rgb):
    return cv2.createCLAHE(2,(8,8)).apply(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))

def feature_grays(rgb):
    """Conservative feature views for difficult photos; geometry is unchanged."""
    base=gray_image(rgb)
    yield 'clahe',base
    # Mild unsharp masking can restore SIFT extrema after defocus/motion blur.
    blur=cv2.GaussianBlur(base,(0,0),1.15)
    sharp=cv2.addWeighted(base,1.75,blur,-.75,0)
    yield 'unsharp',sharp
    # A normalized view helps low-contrast / uneven-light captures.
    norm=cv2.normalize(base,None,0,255,cv2.NORM_MINMAX)
    if not np.array_equal(norm, base):
        yield 'normalized',norm


def _detect_card_base(image:Image.Image, *, _ratio=.70, _normalize=False, _recovery=True):
    array=np.asarray(image.convert('RGB'))
    h,w=array.shape[:2]
    scale=min(1,1600/max(h,w))
    work=cv2.resize(array,None,fx=scale,fy=scale) if scale<1 else array
    feature_image=cv2.normalize(work,None,0,255,cv2.NORM_MINMAX) if _normalize else work
    sift=cv2.SIFT_create(nfeatures=3600,contrastThreshold=.015,edgeThreshold=12)
    candidates=[];partial=[];saw_features=False
    for feature_view,gray in feature_grays(feature_image):
        keys,desc=sift.detectAndCompute(gray,None)
        if desc is None or len(keys)<10:continue
        saw_features=True
        for kind in TYPES:
            points,refs=_layout_reference(kind)
            pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(refs,desc,k=2)
            good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<_ratio*n.distance]
            if len(good)<10: continue
            src=np.float32([points[m.queryIdx] for m in good])
            dst=np.float32([keys[m.trainIdx].pt for m in good])
            matrix,mask=cv2.findHomography(src,dst,cv2.RANSAC,4.5)
            affine,affine_mask=cv2.estimateAffine2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=4.5)
            # Prefer the simpler transform when projective distortion adds few inliers.
            if affine is not None and affine_mask is not None and (mask is None or int(mask.sum()) < 1.3*int(affine_mask.sum())):
                matrix=np.vstack([affine,[0,0,1]])
                mask=affine_mask
            if matrix is None or mask is None: continue
            inlier=mask.ravel().astype(bool); count=int(inlier.sum()); ratio=count/len(good)
            hull=cv2.convexHull(src[inlier]) if count>=3 else None
            coverage=cv2.contourArea(hull)/(800*500) if hull is not None else 0
            if count<10 or ratio<.43 or coverage<.055: continue
            if _normalize and (count<12 or ratio<.52 or coverage<.13):continue
            if _ratio>.70 and (count<12 or ratio<.52 or coverage<.13):continue
            corners=np.float32([[0,0],[799,0],[799,499],[0,499]]).reshape(-1,1,2)
            quad=cv2.perspectiveTransform(corners,matrix).reshape(4,2)
            if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)): continue
            area=cv2.contourArea(quad.astype(np.float32)); wh,ww=work.shape[:2]
            if not .07*wh*ww<area<1.3*wh*ww: continue
            edges=np.linalg.norm(quad-np.roll(quad,-1,axis=0),axis=1)
            if edges.min()<30 or edges.max()/edges.min()>5: continue
            if ((quad[:,0]<-.15*ww)|(quad[:,0]>1.15*ww)|(quad[:,1]<-.15*wh)|(quad[:,1]>1.15*wh)).any():
                if count>=20 and ratio>=.58 and coverage>=.11:
                    partial.append({'detected':False,'kind':None,'suspected_kind':kind,
                        'partial_card':True,'inliers':count,'matches':len(good),
                        'card_quad':(quad/scale).tolist(),'method':'partial_layout_match','feature_view':feature_view})
                continue
            aligned=cv2.warpPerspective(work,np.linalg.inv(matrix),(800,500),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
            # Prefer strong geometric support; tiny bonus keeps the unmodified view first on ties.
            view_bonus=.02 if feature_view=='clahe' else 0
            candidates.append((count*ratio+view_bonus,kind,quad/scale,aligned,count,len(good),coverage,feature_view))
        # Once the native contrast view has strong geometric support, extra
        # sharpening cannot add useful identity evidence. Both templates were tested.
        strong = [c for c in candidates if c[4] >= 25 and c[4]/c[5] >= .65 and c[6] >= .16]
        if strong:
            break
    if not saw_features:
        if _recovery and not _normalize:return _detect_card_base(image,_normalize=True,_recovery=True)
        return {'detected':False,'kind':None,'method':'layout_match'},None,None
    if not candidates:
        if _recovery and _ratio==.70 and not partial:
            return _detect_card_base(image,_ratio=.77,_normalize=_normalize,_recovery=True)
        if partial:return max(partial,key=lambda x:x['inliers']),None,None
        if _recovery and not _normalize:return _detect_card_base(image,_normalize=True,_recovery=True)
        return {'detected':False,'kind':None,'method':'layout_match'},None,None
    # Keep only the strongest preprocessing view per card type; otherwise the
    # same template found in two views would look like an ambiguous classification.
    best_by_kind={}
    for item in candidates:
        kind=item[1]
        if kind not in best_by_kind or item[0]>best_by_kind[kind][0]:best_by_kind[kind]=item
    candidates=sorted(best_by_kind.values(),key=lambda x:x[0],reverse=True)
    top=candidates[0]
    if len(candidates)>1 and candidates[1][0]>.8*top[0]:
        return {'detected':False,'kind':None,'method':'layout_match','ambiguous_layout':True},None,None
    _,kind,quad,aligned,count,matches,coverage,feature_view=top
    result={'detected':True,'kind':kind,'label':TYPES[kind],'method':'layout_match',
            'horizontal_mirror_corrected':bool(cv2.contourArea(quad.astype(np.float32),oriented=True)<0),
            'feature_normalization':_normalize,'feature_view':feature_view,'matching_threshold':_ratio,'inliers':count,'matches':matches,'reference_coverage':round(coverage,3),
            'card_quad':quad.tolist()}
    card_image=Image.fromarray(aligned)
    roi,actual_box=number_region(card_image,kind)
    result['number_box']=actual_box
    return result,card_image,roi




def _detect_zoomed_crop(image, box, *, target_long_side=1500, _mirror_retry=True):
    """Run layout matching on a source crop, optionally upscaled, then map the quad back.

    Upscaling does not invent detail; it only gives SIFT a friendlier scale when a
    genuinely visible card occupies a small portion of a large person/photo scene.
    """
    L,T,R,B=[int(round(v)) for v in box]
    L=max(0,L);T=max(0,T);R=min(image.width,R);B=min(image.height,B)
    if R-L<80 or B-T<50:
        return None,None,None
    crop=image.crop((L,T,R,B))
    factor=max(1.0, min(2.5, target_long_side/max(crop.size)))
    work=crop if factor<=1.01 else crop.resize((round(crop.width*factor),round(crop.height*factor)),Image.Resampling.LANCZOS)
    result,aligned,roi=_detect_card_base(work,_recovery=False)
    if not result.get('detected') and _mirror_retry:
        mirrored,mirrored_aligned,mirrored_roi=_detect_card_mirrored(work,_recovery=False)
        if mirrored is not None:
            result,aligned,roi=mirrored,mirrored_aligned,mirrored_roi
    if not result.get('detected'):
        return None,None,None
    quad=np.asarray(result.get('card_quad',[]),np.float32)
    if quad.shape==(4,2):
        quad=quad/max(factor,1e-6)+np.float32([L,T])
        result['card_quad']=quad.tolist()
    result['local_zoom_factor']=round(float(factor),3)
    result['source_crop']=[L,T,R,B]
    return result,aligned,roi

def _search_windows(width, height):
    """Thirty bounded views covering both sides, upper regions and held cards.

    The old insertion/truncation order silently dropped left-side and coarse views.
    """
    specifications = []
    # Retain the small-window coverage that already worked on real held cards.
    for fw in (.32,.38):
        for cx in (.70,.50,.30):
            for cy in (.92,.84,.72,.58):
                specifications.append((fw,None,cx,cy))
    # Keep the previously successful small-card path ahead of expensive coarse views.
    specifications.extend((.64,.64,cx,.83) for cx in (.50,.17,.83))
    specifications.append((.46,None,.20,.65))
    specifications.extend([(.64,.64,.50,.17),(.64,.64,.17,.17)])
    boxes = []
    for fw, fh, cx, cy in specifications:
        # Preserve the original crop pixels: a one-pixel shift changes SIFT extrema.
        quantize = round if fh is not None else int
        ww = min(width,max(96,quantize(width*fw)))
        hh = min(height,max(60,quantize(height*fh if fh else ww/1.586*1.18)))
        x = max(0,min(width-ww,quantize(width*cx-ww/2)))
        y = max(0,min(height-hh,quantize(height*cy-hh/2)))
        box = (x,y,x+ww,y+hh)
        if box not in boxes:
            boxes.append(box)
    return boxes


def _multiscale_layout_retry(image, deadline=None, *, _allow_mirror=True):
    """Search overlapping local views when a card is small inside a person/photo scene.

    This is local-only and still requires the Iranian template/layout matcher. It is
    intentionally bounded so a failed full-image match does not turn into an unbounded
    sliding-window search.
    """
    W,H=image.size
    if min(W,H)<320:
        return None,None,None,{'attempted':False,'reason':'image_small'}
    boxes=_search_windows(W,H)
    unique=[]
    for b in boxes:
        if b not in unique:unique.append(b)
    attempts=[]
    best_candidate=None
    for index,box in enumerate(unique[:30]):
        if deadline is not None and time.monotonic() >= deadline:
            break
        result,aligned,roi=_detect_zoomed_crop(image,box,_mirror_retry=False)
        attempts.append({'box':list(map(int,box)),'detected':bool(result),'reference_verified':False})
        if result is not None:
            result['method']='multiscale_local_layout_match'
            result['multiscale_attempts']=index+1
            return result,aligned,roi,{'attempted':True,'count':index+1,'hit':True,'attempts':attempts}
        # The strict descriptor matcher can reject a visibly recognizable but blurred
        # held card.  On the already-bounded local crop, use the masked static card
        # references as a second verifier.  It must still produce a valid homography.
        L,T,R,B=box
        crop=image.crop((L,T,R,B))
        ref_scale=1.0
        ref_diag,ref_aligned=verify_reference_layout(crop,context='proposal')
        if not ref_diag.get('verified') and min(crop.size)<80:
            zoom=min(4.0,300/min(crop.size))
            zoomed=crop.resize((round(crop.width*zoom),round(crop.height*zoom)),Image.Resampling.LANCZOS)
            zoom_diag,zoom_aligned=verify_reference_layout(zoomed,context='proposal')
            if zoom_diag.get('verified'):
                ref_diag,ref_aligned,ref_scale=zoom_diag,zoom_aligned,zoom
        attempts[-1]['reference_score']=ref_diag.get('score',0)
        attempts[-1]['reference_verified']=bool(ref_diag.get('verified'))
        if ref_diag.get('inliers',0)>=7 and ref_diag.get('score',0)>=.25 and ref_diag.get('kind') in TYPES:
            if best_candidate is None or ref_diag['score']>best_candidate[0]:
                points=np.asarray(ref_diag.get('quad') or [[0,0],[crop.width-1,0],[crop.width-1,crop.height-1],[0,crop.height-1]],np.float32)
                source_quad=(points/ref_scale+np.float32([L,T])).tolist()
                candidate=ref_aligned if ref_aligned is not None else crop.resize((800,500),Image.Resampling.LANCZOS)
                best_candidate=(ref_diag['score'],candidate,source_quad,ref_diag)
        if ref_diag.get('verified') and ref_aligned is not None and ref_diag.get('quad'):
            quad=(np.asarray(ref_diag['quad'],np.float32)/ref_scale+np.float32([L,T])).tolist()
            kind=ref_diag['kind']
            roi,box_number=number_region(ref_aligned,kind)
            result={'detected':True,'kind':kind,'label':TYPES[kind],
                    'horizontal_mirror_corrected':bool(cv2.contourArea(np.asarray(quad,np.float32),oriented=True)<0),
                    'method':'multiscale_reference_verify','card_quad':quad,
                    'reference_verification':ref_diag,'number_box':box_number,
                    'multiscale_attempts':index+1}
            return result,ref_aligned,roi,{'attempted':True,'count':index+1,'hit':True,'attempts':attempts}
    diagnostic={'attempted':True,'count':len(attempts),'hit':False,'attempts':attempts}
    # Finish the previously working native windows before new mirror recovery.
    # Interleaving two extra mirror matches per window starved later native crops.
    if _allow_mirror and (deadline is None or time.monotonic()<deadline):
        mirrored,mirrored_aligned,mirrored_roi,mirror_diag=_multiscale_layout_retry(
            ImageOps.mirror(image),deadline=deadline,_allow_mirror=False)
        diagnostic['mirror_search']=mirror_diag
        if mirrored is not None and mirrored.get('detected'):
            quad=np.asarray(mirrored['card_quad'],np.float32)
            quad[:,0]=W-1-quad[:,0]
            mirrored.update(card_quad=quad.tolist(),
                            horizontal_mirror_corrected=not bool(mirrored.get('horizontal_mirror_corrected')),
                            method='mirror+'+mirrored['method'])
            return mirrored,mirrored_aligned,mirrored_roi,diagnostic
    if best_candidate is not None:
        _,candidate,quad,diag=best_candidate
        kind=diag['kind'];roi,number_box=number_region(candidate,kind)
        finding={'detected':False,'ocr_candidate':True,'kind':kind,'label':TYPES[kind],
            'method':'local_card_candidate','card_quad':quad,'number_box':number_box,'reference_verification':diag}
        return finding,candidate,roi,diagnostic
    return None,None,None,diagnostic

def _map_rectified_quad_to_source(rectified_quad, source_quad):
    """Map coordinates from the 800x500 rectified view back to the source image."""
    ref=np.float32([[0,0],[799,0],[799,499],[0,499]])
    source=np.asarray(source_quad,np.float32)
    points=np.asarray(rectified_quad,np.float32).reshape(-1,1,2)
    transform=cv2.getPerspectiveTransform(ref,source)
    return cv2.perspectiveTransform(points,transform).reshape(-1,2)

def _remote_proposal_retry(image, progress=None):
    """Use optional object detectors only to propose crops; layout match must confirm."""
    try:
        from roboflow_detector import detect_candidates
        if progress is None:
            proposals,diagnostic=detect_candidates(image)
        else:
            try:
                proposals,diagnostic=detect_candidates(image, progress=progress)
            except TypeError:
                # Keep compatibility with injected/older proposal functions used by embedders/tests.
                proposals,diagnostic=detect_candidates(image)
    except Exception as exc:
        return None,None,None,{'enabled':True,'error':type(exc).__name__}
    for proposal in proposals:
        l,t,r,b=proposal['box']
        # Small margin preserves card border/corners while still removing person/background clutter.
        pad_x=.06*(r-l);pad_y=.08*(b-t)
        L=max(0,int(l-pad_x));T=max(0,int(t-pad_y));R=min(image.width,int(r+pad_x));B=min(image.height,int(b+pad_y))
        if R-L<40 or B-T<25:continue
        crop=image.crop((L,T,R,B))
        # First try the native crop; then upscale only if needed. This keeps old
        # behavior stable while making small held cards much easier to match.
        retry,retry_aligned,retry_roi=_detect_card_base(crop)
        retry_quad_is_global=False
        if not retry.get('detected'):
            zoomed,zoomed_aligned,zoomed_roi=_detect_zoomed_crop(image,(L,T,R,B),target_long_side=1900)
            if zoomed is not None and zoomed.get('detected'):
                retry,retry_aligned,retry_roi=zoomed,zoomed_aligned,zoomed_roi
                retry_quad_is_global=True
        if not retry.get('detected'):
            # A tight object-detection crop often exposes an outer quadrilateral that was
            # invisible in a full person/photo scene. Rectify then require layout confirmation.
            suggested,local_quad=suggest_card_crop(crop)
            if suggested is not None:
                retry,retry_aligned,retry_roi=_detect_card_base(suggested)
                if retry.get('detected'):
                    refined=_map_rectified_quad_to_source(retry.get('card_quad',[[0,0],[799,0],[799,499],[0,499]]),local_quad)
                    retry['card_quad']=(refined+np.float32([L,T])).tolist()
                    retry_quad_is_global=True
                else:
                    ref_diag,ref_aligned=verify_reference_layout(suggested,context='proposal')
                    if ref_diag.get('verified') and ref_aligned is not None:
                        kind=ref_diag['kind']
                        local_ref_quad=ref_diag.get('quad') or [[0,0],[799,0],[799,499],[0,499]]
                        refined=_map_rectified_quad_to_source(local_ref_quad,local_quad)
                        retry_roi,number_box=number_region(ref_aligned,kind)
                        retry={'detected':True,'kind':kind,'label':TYPES[kind],
                               'method':'roboflow_proposal+reference_verify',
                               'card_quad':(refined+np.float32([L,T])).tolist(),
                               'reference_verification':ref_diag,'number_box':number_box}
                        retry_aligned=ref_aligned;retry_quad_is_global=True
            if not retry.get('detected'):
                # Even when border extraction fails, the YOLO crop itself can still
                # carry enough static Iranian-card layout evidence for verification.
                ref_diag,ref_aligned=verify_reference_layout(crop,context='proposal')
                if ref_diag.get('verified') and ref_aligned is not None and ref_diag.get('quad'):
                    kind=ref_diag['kind']
                    retry_roi,number_box=number_region(ref_aligned,kind)
                    retry={'detected':True,'kind':kind,'label':TYPES[kind],
                           'method':'roboflow_proposal+reference_verify',
                           'card_quad':(np.asarray(ref_diag['quad'],np.float32)+np.float32([L,T])).tolist(),
                           'reference_verification':ref_diag,'number_box':number_box}
                    retry_aligned=ref_aligned;retry_quad_is_global=True
                elif ref_diag.get('probable'):
                    proposal['reference_probable']=True
                    proposal['reference_score']=ref_diag.get('score')
                    proposal['suspected_kind']=ref_diag.get('kind')
        elif retry.get('card_quad') and not retry_quad_is_global:
            retry['card_quad']=(np.asarray(retry['card_quad'],np.float32)+np.float32([L,T])).tolist()
            retry_quad_is_global=True
        if retry.get('detected'):
            if not str(retry.get('method') or '').startswith('roboflow_proposal+reference'):
                retry['method']='roboflow_proposal+layout_match'
            retry['proposal_detector']=proposal.get('model_id')
            retry['proposal_confidence']=round(float(proposal.get('confidence',0)),4)
            retry['proposal_box']=[round(float(v),1) for v in proposal['box']]
            retry['roboflow_diagnostic']=diagnostic
            return retry,retry_aligned,retry_roi,diagnostic
    return None,None,None,diagnostic



def _map_quad_from_rotated(quad, original, angle):
    """Map a quad from PIL's rotated view back into original-image coordinates."""
    points=np.asarray(quad,dtype=np.float32).copy()
    x=points[:,0].copy();y=points[:,1].copy()
    angle%=360
    if angle==90:
        points[:,0]=original.width-1-y;points[:,1]=x
    elif angle==180:
        points[:,0]=original.width-1-x;points[:,1]=original.height-1-y
    elif angle==270:
        points[:,0]=y;points[:,1]=original.height-1-x
    return points


def _detect_card_rotated_local(image, deadline=None):
    """Try local template matching on rotated views; never calls cloud/YOLO."""
    for angle in (90,270,180):
        if deadline is not None and time.monotonic() >= deadline:
            break
        rotated=image.rotate(angle,expand=True)
        result,aligned,roi=_detect_card_base(rotated)
        if result.get('detected') and aligned is not None:
            if result.get('card_quad'):
                result['card_quad']=_map_quad_from_rotated(result['card_quad'],image,angle).tolist()
            result['method']='rotation_retry+layout_match'
            result['coarse_rotation_degrees']=angle
            return result,aligned,roi
    return None,None,None

def _detect_card_mirrored(image, **options):
    finding, aligned, roi = _detect_card_base(ImageOps.mirror(image), **options)
    if finding.get('detected'):
        points=np.asarray(finding['card_quad'],np.float32)
        points[:,0]=image.width-1-points[:,0]
        finding.update(card_quad=points.tolist(),horizontal_mirror_corrected=not bool(finding.get('horizontal_mirror_corrected')),method='mirror+layout_match')
        return finding,aligned,roi
    return None,None,None


def _detect_stretched_closeup(image):
    """Recover a close-up saved with a nonuniform resize, with two layout checks.

    SIFT is scale/rotation invariant, but strong anisotropic compression destroys
    its local descriptors before a homography can be estimated. Normalize only
    bounded whole-image views; never infer card presence from aspect ratio alone.
    Native detection always runs first so existing successful paths are retained.
    """
    if min(image.size) < 120 or not .65 <= image.width / image.height <= 1.35:
        return None,None,None
    for angle in (0,90):
        source = image.rotate(angle,expand=True) if angle else image
        normalized = source.resize((800,500),Image.Resampling.LANCZOS)
        finding,aligned,roi = _detect_card_base(normalized,_recovery=False)
        if not finding.get('detected'):
            continue
        matches = max(1,finding.get('matches',0))
        if (finding.get('inliers',0) < 20 or finding.get('inliers',0)/matches < .60
                or finding.get('reference_coverage',0) < .20):
            continue
        reference,_ = verify_reference_layout(normalized,context='proposal')
        if not reference.get('verified') or reference.get('kind') != finding.get('kind'):
            continue
        quad = np.asarray(finding['card_quad'],np.float32)
        if abs(cv2.contourArea(quad)) < .55 * 800 * 500:
            continue
        quad *= np.float32([(source.width-1)/799,(source.height-1)/499])
        if angle:
            quad = _map_quad_from_rotated(quad,image,angle)
        finding.update(card_quad=quad.tolist(),method='aspect_recovery+layout_match',
                       aspect_ratio_corrected=True,aspect_recovery_rotation=angle,
                       reference_verification=reference)
        return finding,aligned,roi
    return None,None,None


def _detect_card_single(image:Image.Image, progress=None, search_budget=60.0):
    """Coarse orientation normalization before final warp; quads stay in input coordinates."""
    search_budget = max(3.0, min(float(search_budget), 60.0))
    deadline = time.monotonic() + search_budget
    if progress:
        try: progress("detect_stage", {"stage": "local", "label": "تطبیق مستقیم کارت ملی"})
        except Exception: pass
    result,aligned,roi=_detect_card_base(image)
    if not result.get('detected'):
        mirrored, mirrored_aligned, mirrored_roi = _detect_card_mirrored(image)
        if mirrored is not None:
            return mirrored,mirrored_aligned,mirrored_roi
        # A close-up card turned sideways must reach orientation recovery before
        # weak scene proposals or sliding windows can consume the search budget.
        if image.height > image.width and 1.3 <= image.height/image.width <= 1.95:
            rotated,rotated_aligned,rotated_roi=_detect_card_rotated_local(image,deadline=deadline)
            if rotated is not None:
                rotated['rotation_retry']=True
                return rotated,rotated_aligned,rotated_roi
        stretched,stretched_aligned,stretched_roi=_detect_stretched_closeup(image)
        if stretched is not None:
            return stretched,stretched_aligned,stretched_roi
        local_diag = {'attempted': False, 'reason': 'geometry_first'}
        local_hint = None
        # In a portrait scene a person dominates whole-image feature matching.
        # Search local crops first; close-up cards still get geometry-first repair.
        from intake import large_faces
        scene_faces=large_faces(image)
        person_scene=any(w*h>.20*image.width*image.height for _,_,w,h in scene_faces)
        if person_scene:
            local_retry,local_aligned,local_roi,local_diag=_multiscale_layout_retry(image,deadline=deadline)
            if local_retry is not None and local_retry.get('detected'):
                local_retry['multiscale_diagnostic']=local_diag
                return local_retry,local_aligned,local_roi
            if local_retry is not None:
                local_hint=(local_retry,local_aligned,local_roi)
        # If texture matching fails, try several plausible card-shaped quadrilaterals.
        # A held card is often not the largest rectangle in a person/photo scene, so using
        # only one contour is brittle. Geometry only proposes; Iranian-card verification is
        # still required before the result can be called a national card.
        if progress:
            try: progress("detect_stage", {"stage": "geometry", "label": "بررسی چند کادر مستطیلی محتمل"})
            except Exception: pass
        geometry_attempts=[]
        proposals=_color_card_candidates(image,max_candidates=4)+_geometric_card_candidates(image,max_candidates=8)
        aspect=max(image.size)/min(image.size)
        if 1.3<=aspect<=1.95:
            whole_quad=[[0,0],[image.width-1,0],[image.width-1,image.height-1],[0,image.height-1]]
            proposals.insert(0,(image.resize((800,500),Image.Resampling.LANCZOS),whole_quad,0.0))
        best_candidate=None
        for candidate_index,(suggested,quad,proposal_score) in enumerate(proposals,start=1):
            if time.monotonic() >= deadline:
                break
            retry,retry_aligned,retry_roi=_detect_card_base(suggested)
            if not retry.get('detected'):
                mirrored,mirrored_aligned,mirrored_roi=_detect_card_mirrored(suggested)
                if mirrored is not None:
                    retry,retry_aligned,retry_roi=mirrored,mirrored_aligned,mirrored_roi
            row={'index':candidate_index,'proposal_score':round(float(proposal_score),4),'verified':False}
            if retry.get('detected'):
                retry['card_quad']=_map_rectified_quad_to_source(
                    retry.get('card_quad',[[0,0],[799,0],[799,499],[0,499]]),quad).tolist()
                retry['method']='geometric_rectify+layout_match'
                retry['geometric_fallback']=True
                retry['geometric_candidate_index']=candidate_index
                retry['multiscale_diagnostic']=local_diag
                row['verified']=True;row['method']='layout_match';geometry_attempts.append(row)
                retry['geometry_diagnostic']={'attempts':geometry_attempts,'hit':True}
                if progress:
                    try: progress("detect_done", {"detected": True, "method": retry.get("method")})
                    except Exception: pass
                return retry,retry_aligned,retry_roi
            ref_diag,ref_aligned=verify_reference_layout(suggested,context='proposal')
            row.update({'reference_score':ref_diag.get('score',0),'reference_probable':bool(ref_diag.get('probable'))})
            if ref_diag.get('inliers',0)>=7 and ref_diag.get('score',0)>=.25 and ref_diag.get('kind') in TYPES:
                if best_candidate is None or ref_diag['score']>best_candidate[0]:
                    # Preserve a bounded review path without calling weak evidence verified.
                    candidate_quad=_map_rectified_quad_to_source(ref_diag['quad'],quad).tolist() if ref_diag.get('quad') else quad
                    best_candidate=(ref_diag['score'],ref_aligned if ref_aligned is not None else suggested,candidate_quad,ref_diag)
            if ref_diag.get('verified') and ref_aligned is not None:
                kind=ref_diag['kind']
                source_quad=ref_diag.get('quad') or [[0,0],[799,0],[799,499],[0,499]]
                mapped=_map_rectified_quad_to_source(source_quad,quad).tolist()
                retry_roi,number_box=number_region(ref_aligned,kind)
                row['verified']=True;row['method']='reference_verify';geometry_attempts.append(row)
                retry={'detected':True,'kind':kind,'label':TYPES[kind],
                       'card_quad':mapped,'method':'geometric_rectify+reference_verify',
                       'geometric_fallback':True,'geometric_candidate_index':candidate_index,
                       'reference_verification':ref_diag,'number_box':number_box,
                       'geometry_diagnostic':{'attempts':geometry_attempts,'hit':True},
                       'multiscale_diagnostic':local_diag}
                if progress:
                    try: progress("detect_done", {"detected": True, "method": retry.get("method")})
                    except Exception: pass
                return retry,ref_aligned,retry_roi
            geometry_attempts.append(row)
        if geometry_attempts:
            result['geometry_diagnostic']={'attempts':geometry_attempts,'hit':False}
        # First search bounded local views. This is the missing path for a small card
        # held in front of a person: background/face texture no longer dominates SIFT.
        if progress:
            try: progress("detect_stage", {"stage": "multiscale", "label": "جست‌وجوی چندمقیاسی کارت کوچک/دور"})
            except Exception: pass
        if not local_diag.get('attempted'):
            local_retry,local_aligned,local_roi,local_diag=_multiscale_layout_retry(image,deadline=deadline)
        else:
            local_retry,local_aligned,local_roi=None,None,None
        result['multiscale_diagnostic']=local_diag
        if local_retry is not None and local_retry.get('detected'):
            local_retry['multiscale_diagnostic']=local_diag
            if progress:
                try: progress("detect_done", {"detected": True, "method": local_retry.get("method")})
                except Exception: pass
            return local_retry,local_aligned,local_roi
        if local_retry is not None:
            local_hint=(local_retry,local_aligned,local_roi)
        # A card held sideways can defeat the first layout pass. Try the local templates
        # on rotated views before invoking any optional remote detector.
        rotated_result,rotated_aligned,rotated_roi=_detect_card_rotated_local(image,deadline=deadline)
        if rotated_result is not None:
            rotated_result['multiscale_diagnostic']=local_diag
            rotated_result['rotation_retry']=True
            if progress:
                try: progress("detect_done", {"detected": True, "method": rotated_result.get("method")})
                except Exception: pass
            return rotated_result,rotated_aligned,rotated_roi
        if time.monotonic() >= deadline:
            result['search_limited']=True
        if local_hint is not None:
            hint,hint_aligned,_=local_hint
            diag=hint.get('reference_verification') or {}
            if best_candidate is None or diag.get('score',0)>best_candidate[0]:
                best_candidate=(diag.get('score',0),hint_aligned,hint['card_quad'],diag)
        if best_candidate is not None:
            _,candidate,source_quad,diag=best_candidate
            kind=diag['kind']
            roi,number_box=number_region(candidate,kind)
            result={'detected':False,'ocr_candidate':True,'kind':kind,'label':TYPES[kind],
                    'method':'bounded_geometric_candidate','card_quad':source_quad,
                    'reference_verification':diag,'number_box':number_box,
                    'search_limited':bool(result.get('search_limited'))}
            return result,candidate,roi
        if result.get('search_limited'):
            return result,None,None
        # Learned object detection is optional proposal generation only. It is especially
        # useful for held-card photos, but local recognition works without an API key.
        if progress:
            try: progress("detect_stage", {"stage": "yolo", "label": "YOLO در حال جست‌وجوی کارت در تصویر"})
            except Exception: pass
        remote,remote_aligned,remote_roi,remote_diag=_remote_proposal_retry(image, progress=progress)
        if remote is not None:
            remote['multiscale_diagnostic']=local_diag
            remote['yolo_presence']=True
            if progress:
                try: progress("detect_done", {"detected": True, "method": remote.get("method"), "model_id": remote.get("proposal_detector")})
                except Exception: pass
            return remote,remote_aligned,remote_roi
        if remote_diag and remote_diag.get('enabled'):
            result['roboflow_diagnostic']=remote_diag
            best=remote_diag.get('best')
            if best:
                result['yolo_presence']=True
                result['yolo_box']=best.get('box')
                result['yolo_confidence']=round(float(best.get('confidence',0)),4)
                result['yolo_model']=best.get('model_id')
                result['yolo_class']=best.get('class')
        if progress:
            try: progress("detect_done", {"detected": False, "yolo_presence": bool(result.get("yolo_presence")), "model_id": result.get("yolo_model")})
            except Exception: pass
        return result,aligned,roi
    if progress:
        try: progress("detect_done", {"detected": True, "method": result.get("method", "local")})
        except Exception: pass
    quad=np.asarray(result['card_quad'],np.float32)
    edge=quad[1]-quad[0]
    angle=int(round(float(np.degrees(np.arctan2(edge[1],edge[0])))/90))*90
    angle%=360
    if not angle:return result,aligned,roi
    corrected=image.rotate(angle,expand=True)
    refined,new_aligned,new_roi=_detect_card_base(corrected)
    if not refined.get('detected') or refined.get('kind')!=result.get('kind'):return result,aligned,roi
    points=np.asarray(refined['card_quad'],np.float32)
    x,y=points[:,0].copy(),points[:,1].copy()
    if angle==90:points[:,0]=image.width-1-y;points[:,1]=x
    elif angle==180:points[:,0]=image.width-1-x;points[:,1]=image.height-1-y
    else:points[:,0]=y;points[:,1]=image.height-1-x
    refined['card_quad']=points.tolist();refined['coarse_rotation_degrees']=angle
    return refined,new_aligned,new_roi


def mark_multiple_cards(image, detection):
    """Look for a second independently matched card outside the first polygon."""
    if not detection.get('detected') or not detection.get('card_quad'):
        return detection
    quad = np.asarray(detection['card_quad'], dtype=np.float32)
    area = abs(cv2.contourArea(quad))
    if area > .85 * image.width * image.height:
        return detection
    center = quad.mean(axis=0)
    mask_quad = center + 1.08 * (quad-center)
    masked = np.array(image.convert('RGB'))
    cv2.fillConvexPoly(masked, np.int32(mask_quad), (255,255,255))
    other, _, _ = _detect_card_base(Image.fromarray(masked), _recovery=False)
    if not other.get('detected'):
        other, _, _ = _detect_card_base(ImageOps.mirror(Image.fromarray(masked)), _recovery=False)
        if other.get('detected'):
            second=np.asarray(other['card_quad'],np.float32)
            second[:,0]=image.width-1-second[:,0]
            other['card_quad']=second.tolist()
    if not other.get('detected') or not other.get('card_quad'):
        return detection
    second = np.asarray(other['card_quad'], dtype=np.float32)
    intersection, _ = cv2.intersectConvexConvex(quad, second)
    if intersection / max(1.0, min(area, abs(cv2.contourArea(second)))) > .20:
        return detection
    detection['multiple_cards'] = True
    detection['card_candidates'] = [
        {'kind': detection.get('kind'), 'quad': quad.tolist()},
        {'kind': other.get('kind'), 'quad': second.tolist()},
    ]
    for row in detection['card_candidates']:
        points = np.asarray(row['quad'])
        row['box'] = [max(0.0,float(points[:,0].min()/image.width)), max(0.0,float(points[:,1].min()/image.height)),
                      min(1.0,float(points[:,0].max()/image.width)), min(1.0,float(points[:,1].max()/image.height))]
    return detection


def detect_card(image:Image.Image, progress=None, search_budget=60.0):
    finding, aligned, roi = _detect_card_single(
    image,
    progress=progress,
    search_budget=search_budget
)
    mark_multiple_cards(image, finding)
    return finding, aligned, roi


def number_crop(aligned,kind):
    return number_region(aligned,kind)[0]


def draw_detection(image,result):
    a=np.array(image.convert('RGB'))
    if result['detected']:
        cv2.polylines(a,[np.int32(result['card_quad'])],True,(25,170,145),max(2,a.shape[1]//300))
    return Image.fromarray(a)


def draw_number_highlight(image,result,kind=None,box=None):
    """Map the exact number ROI back onto the original image, including perspective."""
    a=np.array(image.convert('RGB'));kind=kind or result.get('kind')
    if kind not in NUMBER_ROIS:return image.copy()
    l,t,r,b=box or result.get('number_box') or NUMBER_ROIS[kind]
    box=np.float32([[l*800,t*500],[r*800,t*500],[r*800,b*500],[l*800,b*500]])
    if result.get('card_quad'):
        ref=np.float32([[0,0],[799,0],[799,499],[0,499]])
        transform=cv2.getPerspectiveTransform(ref,np.float32(result['card_quad']))
        quad=cv2.perspectiveTransform(box.reshape(-1,1,2),transform).reshape(4,2)
    else:
        quad=box*np.float32([image.width/800,image.height/500])
    cv2.polylines(a,[np.int32(quad)],True,(14,139,91),max(1,image.width//500))
    return Image.fromarray(a)


def _quad_iou(a,b):
    """Axis-aligned IoU used only to de-duplicate geometric proposals."""
    ax1,ay1,ax2,ay2=a;bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1);iy1=max(ay1,by1);ix2=min(ax2,bx2);iy2=min(ay2,by2)
    inter=max(0.0,ix2-ix1)*max(0.0,iy2-iy1)
    if inter<=0:return 0.0
    aa=max(1.0,(ax2-ax1)*(ay2-ay1));bb=max(1.0,(bx2-bx1)*(by2-by1))
    return inter/max(aa+bb-inter,1.0)


def _color_card_candidates(image,max_candidates=4):
    """Cyan regions only propose crops; template verification remains mandatory."""
    rgb=np.asarray(image.convert('RGB'))
    scale=min(1.0,1000/max(image.size))
    work=cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else rgb
    hsv=cv2.cvtColor(work,cv2.COLOR_RGB2HSV)
    contours=[]
    for low,high in ((65,125),(35,65)):
        mask=cv2.inRange(hsv,np.array([low,12,55],np.uint8),np.array([high,255,255],np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=2)
        found,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    rows=[]
    for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:16]:
        rect=cv2.minAreaRect(contour)
        width,height=rect[1]
        area=width*height
        if not max(200,.002*mask.size)<area<.80*mask.size or min(width,height)<15:
            continue
        aspect=max(width,height)/min(width,height)
        if not 1.20<aspect<2.25 or cv2.contourArea(contour)/area<.40:
            continue
        points=cv2.boxPoints(rect).astype(np.float32)
        sums=points.sum(axis=1);diff=np.diff(points,axis=1).ravel()
        source=np.float32([points[sums.argmin()],points[diff.argmin()],points[sums.argmax()],points[diff.argmax()]])
        if len(np.unique(source,axis=0))!=4:
            continue
        if np.linalg.norm(source[1]-source[0])<np.linalg.norm(source[3]-source[0]):
            source=np.roll(source,-1,axis=0)
        source=source/scale
        center=source.mean(axis=0)
        source=center+1.05*(source-center)
        transform=cv2.getPerspectiveTransform(source,np.float32([[0,0],[799,0],[799,499],[0,499]]))
        aligned=cv2.warpPerspective(rgb,transform,(800,500),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
        score=.75*max(0,1-abs(aspect-1.586)/.7)+.25*min(1,area/(.2*mask.size))
        rows.append((Image.fromarray(aligned),source.tolist(),float(score)))
    return sorted(rows,key=lambda r:r[2],reverse=True)[:max_candidates]


def _geometric_card_candidates(image,max_candidates=8):
    """Return several card-shaped perspective proposals, without classifying any of them.

    Multiple edge/contrast views are intentional: a blurred laminated card can have a weak
    border in one view while still being obvious in another.  Every returned proposal must
    later pass the Iranian-card layout/reference verifier, so geometry alone cannot produce
    a positive national-card decision.
    """
    a=np.asarray(image.convert('RGB'));scale=min(1,1400/max(image.size))
    work=cv2.resize(a,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else a
    gray=cv2.cvtColor(work,cv2.COLOR_RGB2GRAY)
    clahe=cv2.createCLAHE(2.0,(8,8)).apply(gray)
    views=[gray,clahe]
    image_area=float(work.shape[0]*work.shape[1])
    proposals=[]
    for view in views:
        smooth=cv2.GaussianBlur(view,(5,5),0)
        for low,high in ((24,78),(38,118),(55,165)):
            edges=cv2.Canny(smooth,low,high)
            edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=1)
            contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:140]:
                perimeter=cv2.arcLength(contour,True)
                if perimeter<80:continue
                for epsilon in (.018,.026,.035):
                    quad=cv2.approxPolyDP(contour,epsilon*perimeter,True)
                    if len(quad)!=4 or not cv2.isContourConvex(quad):continue
                    pts=quad.reshape(4,2).astype(np.float32);area=float(cv2.contourArea(pts))
                    # Permit small held cards, but never tiny noise or nearly the whole frame.
                    if not .0035*image_area<area<.985*image_area:continue
                    sums=pts.sum(axis=1);diff=np.diff(pts,axis=1).ravel()
                    ordered=np.float32([pts[sums.argmin()],pts[diff.argmin()],pts[sums.argmax()],pts[diff.argmax()]])
                    if len(np.unique(ordered,axis=0))!=4:continue
                    width=max(np.linalg.norm(ordered[1]-ordered[0]),np.linalg.norm(ordered[2]-ordered[3]))
                    height=max(np.linalg.norm(ordered[3]-ordered[0]),np.linalg.norm(ordered[2]-ordered[1]))
                    if min(width,height)<24:continue
                    aspect=max(width,height)/max(min(width,height),1e-6)
                    if not 1.25<aspect<2.15:continue
                    if width<height:ordered=np.roll(ordered,-1,axis=0)
                    source=ordered/max(scale,1e-6)
                    xs=source[:,0];ys=source[:,1]
                    bbox=(float(xs.min()),float(ys.min()),float(xs.max()),float(ys.max()))
                    if any(_quad_iou(bbox,row['bbox'])>.78 for row in proposals):continue
                    # Card aspect is ~1.586. Rank by contour size and aspect closeness, not
                    # just size, so a hand-held card can outrank a large monitor/window.
                    aspect_score=max(0.0,1.0-abs(aspect-1.586)/.70)
                    area_score=min(1.0,area/(.22*image_area))
                    score=.62*aspect_score+.38*area_score
                    matrix=cv2.getPerspectiveTransform(source,np.float32([[0,0],[799,0],[799,499],[0,499]]))
                    aligned=cv2.warpPerspective(a,matrix,(800,500),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
                    proposals.append({'score':score,'bbox':bbox,'quad':source.tolist(),'image':Image.fromarray(aligned)})
                    if width<height:
                        # Also try an upright but vertically stretched scan. Turning
                        # every portrait rectangle sideways loses this common case.
                        alternate=np.roll(source,-1,axis=0)
                        alternate_matrix=cv2.getPerspectiveTransform(alternate,np.float32([[0,0],[799,0],[799,499],[0,499]]))
                        alternate_image=cv2.warpPerspective(a,alternate_matrix,(800,500),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
                        proposals.append({'score':score-.005,'bbox':bbox,'quad':alternate.tolist(),'image':Image.fromarray(alternate_image)})
                    break
    proposals.sort(key=lambda row:row['score'],reverse=True)
    return [(row['image'],row['quad'],row['score']) for row in proposals[:max_candidates]]


def suggest_card_crop(image):
    """Best geometric crop suggestion only; never classifies it as a national card."""
    candidates=_color_card_candidates(image,max_candidates=1) or _geometric_card_candidates(image,max_candidates=1)
    if not candidates:return None,None
    aligned,quad,_=candidates[0]
    return aligned,quad

def draw_presence(image,result):
    """Draw the best available card presence evidence for the simple UI."""
    a=np.array(image.convert('RGB'))
    if result.get('detected') and result.get('card_quad'):
        cv2.polylines(a,[np.int32(result['card_quad'])],True,(8,139,91),max(2,a.shape[1]//280))
    elif result.get('yolo_box'):
        l,t,r,b=[int(round(v)) for v in result['yolo_box']]
        cv2.rectangle(a,(l,t),(r,b),(214,145,35),max(2,a.shape[1]//280))
    return Image.fromarray(a)
