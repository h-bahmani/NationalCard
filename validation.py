"""Transcription and checksum are independent; invalid test numbers remain visible."""
import re
DIGIT_MAP=str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩','01234567890123456789')


def normalize(value):
    if not isinstance(value,str): return None
    out=re.sub(r'[\s\u200c-\u200f\u202a-\u202e\u2066-\u2069\-–]','',value.translate(DIGIT_MAP))
    return out if re.fullmatch(r'[0-9]{10}',out) else None


def checksum_valid(value):
    if not value or not re.fullmatch(r'[0-9]{10}',value) or len(set(value))==1: return False
    remainder=sum(int(value[i])*(10-i) for i in range(9))%11
    return int(value[-1])==(remainder if remainder<2 else 11-remainder)


def make_result(card,raw,readable,model):
    value=normalize(raw) if readable else None
    return {'status':'extracted' if value else 'partial_reading',
            'national_id':value,'raw_text':raw,'checksum_valid':checksum_valid(value),
            'card':card,'ocr_model':model,'text_model':'openai/gpt-oss-120b',
            'text_review':None,
            'note':'عدد خوانده‌شده مستقل از بررسی رقم کنترل است. تایید اصالت یا هویت انجام نمی‌شود.'}
