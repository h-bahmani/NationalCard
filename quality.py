"""Scale-aware photo hints, not pass/fail or calibrated OCR probabilities."""
import cv2
import numpy as np

def attach_quality(decision, image, aligned, detection):
    """Track native card pixels even after detector rectification/upscaling."""
    decision['quality'] = quality_report(aligned if aligned is not None else image)
    decision['source_quality'] = quality_report(image)
    quad = detection.get('card_quad')
    if quad is not None:
        points = np.asarray(quad, dtype=float)
        if points.shape == (4, 2) and np.isfinite(points).all():
            original = image.info.get('cardreader_original_size', image.size)
            points *= np.array([original[0] / image.width, original[1] / image.height])
            edges = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
            native_size = [round(float((edges[0] + edges[2]) / 2)),
                           round(float((edges[1] + edges[3]) / 2))]
            decision['quality']['native_card_size'] = native_size
            if min(native_size) < 180:
                decision['quality']['tips'].append('کارت در عکس اصلی کوچک است؛ نزدیک‌تر عکس بگیر. بزرگ‌نمایی جای جزئیات اصلی را نمی‌گیرد.')


def quality_report(image):
    gray=np.asarray(image.convert('L'))
    factor=min(1,800/max(image.size))
    sample=cv2.resize(gray,None,fx=factor,fy=factor) if factor<1 else gray
    brightness=float(sample.mean());contrast=float(sample.std())
    sharpness=float(cv2.Laplacian(sample,cv2.CV_32F).var())
    gx=cv2.Sobel(sample,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(sample,cv2.CV_32F,0,1,ksize=3)
    ex=float((gx*gx).mean());ey=float((gy*gy).mean())
    directional=max(ex,ey)/max(min(ex,ey),1e-6)
    white=float((sample>250).mean());tips=[]
    original_size=image.info.get('cardreader_original_size',list(image.size))
    if min(original_size)<250:tips.append('کارت کوچک است؛ نزدیک‌تر و با وضوح بیشتر عکس بگیر.')
    if brightness<65:tips.append('نور کم است؛ کارت را در نور یکنواخت و بدون فلاش قرار بده.')
    if brightness>225 and contrast<35:tips.append('عکس خیلی روشن است؛ نور مستقیم یا نوردهی را کمتر کن.')
    if contrast<24:tips.append('نوشته‌ها کم‌رنگ‌اند؛ نور و زاویهٔ عکس را عوض کن.')
    if sharpness<45:tips.append('ممکن است عکس تار باشد؛ روی نوشته‌ها فوکوس کن و گوشی را ثابت نگه دار.')
    if sharpness<80 and directional>3.5:tips.append('تاری جهت‌دار دیده می‌شود؛ هنگام عکس‌گرفتن حرکت دوربین را کمتر کن.')
    return {'possibly_blurry':sharpness<45,'possibly_motion_blur':sharpness<80 and directional>3.5,
        'ocr_recovery_recommended':sharpness<80 or contrast<28,'processed_size':list(image.size),'original_size':original_size,'brightness':round(brightness,1),
        'contrast_std':round(contrast,1),'laplacian_variance':round(sharpness,1),'gradient_directionality':round(directional,2),
        'very_light_fraction':round(white,3),'tips':tips,
        'heuristic_only':True,'message':'این نشانه‌ها راهنمای عکس‌گرفتن‌اند، نه درصد صحت تشخیص.'}
