"""Bundled PP-OCRv4 DB text localization, CPU ONNX; not an ID classifier.

Model: PaddleOCR via RapidAI/RapidOCR, Apache-2.0. Postprocessing is
implemented here with OpenCV; returned scores describe text, not identity.
"""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np

@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort
    ort.disable_telemetry_events()
    options=ort.SessionOptions()
    options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    return ort.InferenceSession(str(Path(__file__).parent/'models/text_detector.onnx'),
        sess_options=options,providers=['CPUExecutionProvider'])

def detect_text(image):
    rgb=np.asarray(image.convert('RGB'));h,w=rgb.shape[:2]
    scale=min(1.5,960/max(h,w))
    rh=max(32,round(h*scale/32)*32);rw=max(32,round(w*scale/32)*32)
    # Upstream model preprocessing uses OpenCV BGR with ImageNet normalization.
    pixels=cv2.resize(rgb[:,:,::-1],(rw,rh)).astype(np.float32)/255
    pixels=(pixels-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    try:
        session=_session()
        prob=session.run(None,{session.get_inputs()[0].name:pixels.transpose(2,0,1)[None]})[0][0,0]
    except Exception:
        # Field localization is optional because fixed template ROIs are always available.
        # Missing/broken ONNX Runtime must never stop card extraction.
        return []
    contours,_=cv2.findContours((prob>.28).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    result=[];ph,pw=prob.shape
    for contour in contours:
        x,y,bw,bh=cv2.boundingRect(contour)
        if bw<3 or bh<3:continue
        mask=np.zeros((bh,bw),np.uint8)
        cv2.drawContours(mask,[contour-np.array([[[x,y]]])],-1,1,-1)
        score=float(cv2.mean(prob[y:y+bh,x:x+bw],mask)[0])
        if score<.5:continue
        # Expand the DB shrunken text polygon to retain dots and leading digits.
        pad=max(2,cv2.contourArea(contour)*1.4/max(cv2.arcLength(contour,True),1))
        box=[max(0,(x-pad)/pw),max(0,(y-pad)/ph),min(1,(x+bw+pad)/pw),min(1,(y+bh+pad)/ph)]
        if (box[3]-box[1])*h<5:continue
        result.append({'box':box,'score':round(score,3)})
    return sorted(result,key=lambda p:(p['box'][1],p['box'][0]))

def field_regions(image,kind):
    """Assign detected value lines to template rows; preserve fixed safe fallback.

    Right-side printed labels are excluded. Box provenance is explicit so a
    downstream consumer can distinguish localization from fallback geometry.
    """
    from pipeline import REGIONS
    boxes=detect_text(image);regions={};sources={}
    # Assign rows jointly from top to bottom. Independent nearest-row selection
    # can give the father's name to expiry when the alignment is shifted down.
    candidates=[]
    for item in boxes:
        x1,y1,x2,y2=item['box']
        if x1<.32 or x2>.84 or x2<.54 or y2-y1>.11:
            continue
        candidates.append(item)
    candidates.sort(key=lambda item:(sum(item['box'][1::2])/2,item['box'][0]))
    # Join split words on the same value line before assigning a field.
    lines=[]
    for item in candidates:
        b=item['box'];cy=(b[1]+b[3])/2
        if lines and abs(cy-(lines[-1][1]+lines[-1][3])/2)<.022 and b[0]-lines[-1][2]<.06:
            a=lines[-1];lines[-1]=[min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
        else:lines.append(list(b))
    names=list(REGIONS[kind])
    from functools import lru_cache
    @lru_cache(None)
    def assign(index,previous):
        if index==len(names):return 0.0,()
        name=names[index];base=REGIONS[kind][name];cy=(base[1]+base[3])/2
        cost,tail=assign(index+1,previous)
        best=(cost+.12,(None,)+tail)
        for j in range(previous+1,len(lines)):
            x1,y1,x2,y2=lines[j];yc=(y1+y2)/2
            if abs(yc-cy)>(.10 if kind=='smart' else .065):continue
            if name in ('national_id','birth_date','expiry_date','document_number') and x1<.39:continue
            if previous>=0 and yc-(lines[previous][1]+lines[previous][3])/2<.045:continue
            cost,tail=assign(index+1,j)
            option=(cost+abs(yc-cy),(j,)+tail)
            if option[0]<best[0]:best=option
        return best
    _,assignment=assign(0,-1)
    for name,j in zip(names,assignment):
        base=REGIONS[kind][name]
        # This recovery targets the two date rows. Preserve the already-tested
        # name/number crop geometry: tighter crops can reduce name-model agreement.
        if kind!='smart' or name not in ('birth_date','expiry_date'):
            l,t,r,b=base;cy=(t+b)/2;legacy=[]
            for item in boxes:
                x1,y1,x2,y2=item['box'];yc=(y1+y2)/2
                if abs(yc-cy)>(.055 if kind=='smart' else .045):continue
                min_x=.32 if name in ('first_name','last_name','father_name','birth_place') else .39
                if x1<min_x or x2>.84 or x2<.54 or y2-y1>.11:continue
                overlap=max(0,min(b,y2)-max(t,y1))/max(y2-y1,.001)
                legacy.append((overlap-abs(yc-cy),item))
            if legacy:
                item=max(legacy,key=lambda p:p[0])[1];x1,y1,x2,y2=item['box']
                regions[name]=(max(.33,min(l,x1-.009)),min(t,max(0,y1-.006)),min(.865,max(r,x2+.009)),max(b,min(1,y2+.006)))
                sources[name]='text_detector'
            else:regions[name]=base;sources[name]='template_fallback'
            continue
        if j is None:
            regions[name]=base;sources[name]='template_fallback';continue
        x1,y1,x2,y2=lines[j]
        pad=max(.01,(y2-y1)*.22)
        # Keep horizontal safety margins but not the old vertical template union:
        # that union can include a neighboring name and clip the actual date.
        regions[name]=(max(.33,min(base[0],x1-.012)),max(0,y1-pad),
                       min(.865,max(base[2],x2+.012)),min(1,y2+pad))
        sources[name]='text_detector_ordered'
    return regions,sources,boxes
