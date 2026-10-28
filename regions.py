"""Locate the printed number row inside a padded template search region."""
import cv2
import numpy as np
from PIL import Image

NUMBER_SEARCH={'smart':(.50,.225,.845,.355),'old':(.39,.11,.84,.265)}

def focus_row(crop):
    gray=np.asarray(crop.convert('L'));h,w=gray.shape
    # Local dark ink is less sensitive to the colored security background.
    ink=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,np.ones((7,13),np.uint8))
    hsv=cv2.cvtColor(np.asarray(crop.convert('RGB')),cv2.COLOR_RGB2HSV)
    ink[((hsv[:,:,0]>5)&(hsv[:,:,0]<35)&(hsv[:,:,1]>120))]=0
    _,binary=cv2.threshold(ink,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    n,_,stats,centers=cv2.connectedComponentsWithStats(binary)
    parts=[]
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if area>=2 and max(2,h*.045)<=bh<=h*.75 and bw<=max(18,bh*1.6):
            parts.append((x,y,bw,bh,float(centers[i][1])))
    groups=[]
    for p in parts:
        group=[q for q in parts if abs(q[4]-p[4])<max(5,h*.17)]
        if len(group)<5:continue
        l=min(q[0] for q in group);r=max(q[0]+q[2] for q in group)
        t=min(q[1] for q in group);b=max(q[1]+q[3] for q in group)
        if r-l<w*.42 or b-t>h*.8:continue
        groups.append((len(group)*(r-l),l,t,r,b))
    if not groups:return crop,(0,0,w,h)
    _,l,t,r,b=max(groups)
    pad=max(3,round((b-t)*.25))
    # Keep the full horizontal search region: small leading zero components
    # must not disappear merely because their ink component is tiny.
    box=(max(0,l-max(8,pad*2)),max(0,t-pad),min(w,r+pad),min(h,b+pad))
    return crop.crop(box),box

LEGACY_NUMBER_ROIS={'smart':(.57,.25,.83,.34),'old':(.423,.125,.825,.255)}

def number_region(aligned,kind):
    outer=NUMBER_SEARCH[kind];w,h=aligned.size
    pixels=tuple(round(v*(w if i%2==0 else h)) for i,v in enumerate(outer))
    crop=aligned.crop(pixels);_,box=focus_row(crop)
    base=LEGACY_NUMBER_ROIS[kind]
    # Keep the validated legacy geometry unless neighboring ink needs more
    # room on the leading-digit side. Never narrow the original region.
    left=min(base[0],float((pixels[0]+box[0])/w))
    actual=(max(outer[0],left),base[1],base[2],base[3])
    focused=aligned.crop(tuple(round(v*(w if i%2==0 else h)) for i,v in enumerate(actual)))
    return focused,actual
