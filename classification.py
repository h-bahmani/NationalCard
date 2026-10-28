"""Simple three-state UI classification.

The UI intentionally avoids yellow/amber "maybe" states.  Company decisions are
accepted / review / rejected.  Detailed evidence remains available in diagnostics.
"""
from company_policy import detection_confidence


def card_classification(detection, decision):
    finding = detection.get('selection_detection') or detection
    conf = detection_confidence(detection)
    pct = conf['percent']
    blurry = (decision.get('quality') or {}).get('possibly_blurry', False)
    if detection.get('multiple_cards'):
        return {'status':'review','tone':'neutral','title':'یک کارت را انتخاب کن',
                'message':'چند کارت در عکس دیده شد؛ فقط محدودهٔ کارت موردنظر را مشخص کن.','confidence':conf}

    if finding.get('detected') or detection.get('detected'):
        return {
            'status': 'accepted', 'tone': 'good',
            'title': 'کارت ملی شناسایی شد',
            'message': f'تطبیق کارت انجام شد · امتیاز اطمینان عملیاتی {pct}٪. ' +
                       ('کیفیت تصویر پایین است؛ فیلدهای OCR جداگانه بازبینی می‌شوند.' if blurry else
                        'اطلاعات هر فیلد پس از OCR و اعتبارسنجی جداگانه بررسی می‌شود.'),
            'confidence': conf,
        }

    reference=detection.get('reference_verification') or {}
    probable_layout=bool((reference.get('probable') or detection.get('reference_probable')) and
                         float(reference.get('appearance') or 0)>=.35)
    person_scene=bool((decision.get('evidence') or {}).get('large_face_count'))
    # A rectangle, a handful of feature matches, or an expired search budget is
    # not evidence that this is an identity card (e.g. a router/SIM package).
    needs_review=(detection.get('yolo_presence') or probable_layout or detection.get('partial_card') or
                  detection.get('user_selected') or person_scene)
    if needs_review and decision.get('status')!='unsupported':
        return {
            'status': 'review', 'tone': 'neutral',
            'title': 'کادر انتخابی نیاز به بازبینی دارد' if detection.get('user_selected') else 'محدودهٔ کارت مشخص نیست',
            'message': f'امتیاز اطمینان عملیاتی {pct}٪. ' +
                       ('محدودهٔ انتخاب‌شده فقط برای بازبینی خوانده می‌شود؛ تأیید کارت انجام نشده است.' if detection.get('user_selected') else
                        'اگر کارت ملی در عکس است، محدودهٔ کامل آن را انتخاب کن.'),
            'confidence': conf,
        }

    conf=detection_confidence({**detection,'intake_rejected':True})
    return {
        'status': 'rejected', 'tone': 'bad',
        'title': 'کارت ملی پیدا نشد',
        'message': 'در این تصویر الگوی کارت ملی پیدا نشد. عکس واضحِ روی کارت ملی را بارگذاری کن.',
        'confidence': conf,
    }
