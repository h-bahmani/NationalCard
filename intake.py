"""Operational acceptance decisions; these are not identity/authenticity checks."""
from pathlib import Path
import cv2
import numpy as np

def barcode_regions(image):
    gray=np.asarray(image.convert('L'));h,w=gray.shape
    scale=min(1,1000/max(w,h));gray=cv2.resize(gray,None,fx=scale,fy=scale) if scale<1 else gray
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    edge=cv2.convertScaleAbs(gx-gy)
    _,mask=cv2.threshold(cv2.GaussianBlur(edge,(7,7),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,25),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        if bw<100 or bh<20 or bw/bh<2.3:continue
        band=gray[y:y+bh,x:x+bw]
        threshold=cv2.threshold(band,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        profiles=threshold[max(0,bh//5):max(1,4*bh//5)]<128
        transitions=np.count_nonzero(np.diff(profiles.astype(np.int8),axis=1),axis=1)
        coherence=np.mean(np.abs(profiles.mean(axis=0)-.5)*2)
        if np.median(transitions)>=32 and coherence>.55:
            boxes.append([round(x/scale),round(y/scale),round(bw/scale),round(bh/scale)])
    return boxes

def large_faces(image):
    cascade_path=Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'
    if not cascade_path.exists():return []
    gray=np.asarray(image.convert('L'));h,w=gray.shape
    scale=min(1,1000/max(w,h));small=cv2.resize(gray,None,fx=scale,fy=scale) if scale<1 else gray
    faces=cv2.CascadeClassifier(str(cascade_path)).detectMultiScale(small,1.1,5,minSize=(60,60))
    return [[round(v/scale) for v in box] for box in faces if box[2]*box[3]>.075*small.size]

def assess_image(image,detection,aligned=None):
    faces=large_faces(image)
    evidence={'large_face_count':len(faces),'authenticity_checked':False}
    if detection.get('multiple_cards'):
        return dict(status='selection_required',allowed=False,title='بیش از یک کارت در تصویر است',
            message='محدودهٔ یک کارت را انتخاب کن تا اطلاعات کارت‌ها با هم مخلوط نشود.',evidence=evidence)
    if detection.get('partial_card'):
        return dict(status='retake',allowed=False,title='احتمالاً کارت ملی در تصویر است',
            message='بخشی از قالب کارت ملی شناسایی شد؛ برای جلوگیری از حذف رقم یا جابه‌جایی فیلدها، عکس کامل و نزدیک از خود کارت بگیرید.',evidence=evidence)
    if detection.get('detected'):
        quad=np.float32(detection['card_quad'])
        area=abs(cv2.contourArea(quad))/(image.width*image.height)
        edges=np.linalg.norm(quad-np.roll(quad,-1,axis=0),axis=1)
        outside_faces=[f for f in faces if cv2.pointPolygonTest(quad,(f[0]+f[2]/2,f[1]+f[3]/2),False)<0]
        small=bool(edges.min()<110 or area<.08)
        evidence.update(card_area_fraction=round(area,4),card_short_edge_pixels=round(float(edges.min()),1),
                        person_in_scene=bool(outside_faces),small_card=small)
        # Presence/classification and OCR quality are separate decisions. A held card is
        # still a detected Iranian national card; the OCR layer may later ask for review.
        return dict(status='accepted',allowed=True,title='کارت ملی تشخیص داده شد',
                    message=('کارت ملی داخل عکس فرد پیدا شد؛ همان ناحیه جدا و بزرگ‌نمایی می‌شود و OCR روی خود کارت اجرا می‌شود.'
                             if outside_faces else 'کارت ملی شناسایی شد و آمادهٔ خواندن است.'),evidence=evidence)
    if detection.get('yolo_presence'):
        best=(detection.get('roboflow_diagnostic') or {}).get('best') or {}
        evidence.update(yolo_card_present=True,yolo_confidence=detection.get('yolo_confidence') or best.get('confidence'),
                        yolo_model=detection.get('yolo_model') or best.get('model_id'))
        return dict(status='card_present',allowed=False,title='کارت در تصویر پیدا شد',
            message='YOLO یک کارت هویتی را پیدا کرد، اما الگوی کارت ملی ایرانی هنوز با اطمینان کافی تأیید نشده است.',evidence=evidence)
    bars=barcode_regions(image);evidence['barcode_regions']=bars
    if bars:
        return dict(status='unsupported',allowed=False,title='تصویر برای استخراج کارت ملی پذیرفته نیست',
            message='الگوی بارکد یا برچسب کالا دیده شد و روی کارت ملی شناسایی نشد. عکس روی کارت ملی را بارگذاری کنید.',evidence=evidence)
    if faces:
        return dict(status='retake',allowed=False,title='از خود کارت ملی عکس بگیرید',
            message='عکس فرد برای استخراج مشخصات کارت مناسب نیست؛ کارت باید کامل، نزدیک و واضح داخل تصویر باشد.',evidence=evidence)
    return dict(status='not_detected',allowed=False,title='کارت ملی تشخیص داده نشد',
        message='تطبیق خودکار موفق نبود. اگر کارت در عکس هست، محدوده‌اش را روی تصویر انتخاب کنید تا خواندن ادامه پیدا کند.',evidence=evidence)
