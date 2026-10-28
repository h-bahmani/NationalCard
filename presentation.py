"""Literal digit normalization, date checks and disagreement-aware display, never digit repair."""
import re
from validation import DIGIT_MAP,normalize,checksum_valid
FIELD_NAMES={'first_name':'نام','last_name':'نام خانوادگی','father_name':'نام پدر',
             'birth_date':'تاریخ تولد','expiry_date':'تاریخ پایان اعتبار',
             'document_number':'شماره شناسنامه','birth_place':'محل تولد'}
def plain(value):
    return re.sub(r'[\u200c-\u200f\u202a-\u202e\u2066-\u2069]','',str(value or '')).strip()
def date_value(raw):
    value=plain(raw).translate(DIGIT_MAP)
    # Card dates have two printed month/day digits. A single OCR digit can be
    # a truncated 15, not an observed 01; never complete it with a zero.
    match=re.fullmatch(r'(\d{4})\s*[/.-]\s*(\d{2})\s*[/.-]\s*(\d{2})',value)
    if not match:return None
    y,m,d=map(int,match.groups())
    # Structural Solar Hijri check; no date completion or year guessing.
    max_day=31 if m<=6 else 30
    if not 1200<=y<=1599 or not 1<=m<=12 or not 1<=d<=max_day:return None
    return f'{y:04d}/{m:02d}/{d:02d}'

def numeric_consensus(readings,parser=normalize):
    candidates=list(dict.fromkeys(v for raw in readings if (v:=parser(raw)) is not None))
    raw_distinct=list(dict.fromkeys(plain(v).translate(DIGIT_MAP) for v in readings if plain(v)))
    if len(candidates)==1 and len(raw_distinct)==1:return candidates[0],False
    return (candidates[0] if len(candidates)==1 else None),len(raw_distinct)>1


def result_field_value(fields, name):
    """Primary UI values: dates must be structurally valid and year-first.

    Malformed date strings remain in evidence, never in the main date tile.
    This does not reorder tokens, repair digits, or promote a review candidate.
    """
    row = fields.get(name, {})
    status = str(row.get('status') or '')
    value = row.get('value')
    accepted = status in {'accepted','read','candidate','cloud','reviewed'} and value not in (None,'','—')
    suggested = str(row.get('suggested_value') or row.get('candidate_value') or '').strip()
    raw = str(row.get('raw_text') or row.get('raw_selected_text') or row.get('text') or '').strip()
    if name in ('birth_date','expiry_date'):
        candidate = date_value(value) if accepted else date_value(row.get('display_value')) or date_value(raw)
        return (candidate,'accepted' if accepted else 'raw','') if candidate else ('—','missing','')
    if accepted:return str(value),'accepted',suggested
    if raw and raw!='—':return raw,'raw',suggested
    if suggested and suggested!='—':return suggested,'suggested',''
    return '—','missing',''

def field_display(name, data):
    """Return accepted values separately from literal review-only OCR evidence."""
    text_fields = ('first_name', 'last_name', 'father_name', 'birth_place')
    raw = plain(data.get('raw_selected_text') or data.get('text'))
    suggested = plain(data.get('suggested_text') or data.get('suggested_value'))
    if name in text_fields and data.get('decision'):
        accepted = plain(data.get('accepted_text'))
        if data.get('decision') == 'accepted' and accepted:
            return {'value': accepted, 'status': 'read', 'numeric': False,
                    'auto_corrected': False, 'raw_value': raw or accepted,
                    'suggested_value': suggested if suggested and suggested != accepted else None}
        if raw:
            return {'value': raw, 'status': 'review', 'numeric': False,
                    'raw_value': raw, 'suggested_value': suggested or None}
        if suggested:
            return {'value': suggested, 'status': 'review', 'numeric': False,
                    'raw_value': '', 'suggested_value': suggested}
        return {'value': '—', 'status': 'unread', 'numeric': False,
                'raw_value': '', 'suggested_value': None}

    if not raw:
        return {'value': '—', 'status': 'unread', 'numeric': False}
    if name in ('first_name', 'last_name', 'father_name') and (any(c.isdigit() for c in raw) or sum(c.isalpha() for c in raw) < 2):
        return {'value': '—', 'status': 'unread', 'numeric': False}
    if name in ('birth_date', 'expiry_date'):
        val = date_value(raw)
        if not val:
            return {'value':'—','status':'invalid','numeric':True,'raw_value':raw}
        if data.get('ambiguous'):
            return {'value': val, 'status': 'ambiguous', 'numeric': True, 'raw_value': raw}
        return {'value': val, 'status': 'read', 'numeric': True, 'raw_value': raw}
    if name == 'document_number':
        val = raw.translate(DIGIT_MAP).replace(' ', '')
        valid = bool(re.fullmatch(r'[0-9]{1,12}', val))
        return {'value': val if valid else raw, 'status': ('ambiguous' if data.get('ambiguous') else 'read') if valid else 'review', 'numeric': True, 'raw_value': raw}
    return {'value': raw, 'status': 'ambiguous' if data.get('ambiguous') else 'read', 'numeric': False, 'raw_value': raw}

def number_display(result):
    """Accept valid IDs; keep literal OCR digits visible as review-only otherwise.

    Review-only text is displayed to an operator but is never promoted into the
    normalized national_id export field. No digit is repaired or inferred here.
    """
    cloud=normalize(result.get('national_id') or '')
    field_map=result.get('fields',{}) or {}
    data=field_map.get('national_id',{}) or {}
    selected=normalize(data.get('text','') or '')
    local=normalize(result.get('candidate') or '')

    # Preserve the legacy API contract only for checksum-valid literals. Invalid but
    # fully observed digits stay visible to the operator without being exported as ID.
    if 'national_id' not in field_map and local:
        return {'value':local,'status':'candidate' if checksum_valid(local) else 'review'}

    for source,value in (('cloud',cloud),('candidate',selected),('candidate',local)):
        if value and checksum_valid(value):
            if source == 'candidate' and data.get('ambiguous'):
                continue
            return {'value':value,'status':source}

    # Search literal engine readings, but accept only when exactly one checksum-valid
    # number exists. Multiple valid alternatives are a review case, not a guess.
    readings=data.get('readings',[]) or []
    valid=list(dict.fromkeys(v for raw in readings if (v:=normalize(raw)) and checksum_valid(v)))
    if len(valid)==1 and not data.get('ambiguous'):
        return {'value':valid[0],'status':'candidate'}

    # Keep a literal, incomplete OCR candidate visible for the operator. This value is
    # explicitly review-only; exporting still leaves national_id empty unless accepted.
    raw_candidates=[data.get('raw_selected_text'),data.get('text'),result.get('candidate'),
                    (result.get('raw_result') or {}).get('raw_text')]
    for raw in raw_candidates:
        if not isinstance(raw,str) or not raw.strip():
            continue
        literal=plain(raw).translate(DIGIT_MAP)
        digits=re.sub(r'[^0-9]','',literal)
        if 1<=len(digits)<=20:
            return {'value':digits,'status':'review'}
    return {'value':'—','status':'unread'}

