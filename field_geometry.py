"""Prevent a selected date row from being assigned to the other date field."""
def check_field_assignment(name, box, kind):
    if kind != 'smart' or name not in ('birth_date','expiry_date'):
        return
    from pipeline import REGIONS
    from presentation import FIELD_NAMES
    other = 'expiry_date' if name == 'birth_date' else 'birth_date'
    own = REGIONS[kind][name]; opposite = REGIONS[kind][other]
    own_y = (own[1]+own[3])/2; other_y = (opposite[1]+opposite[3])/2
    center = (box[1]+box[3])/2
    if box[1] <= min(own_y,other_y) and box[3] >= max(own_y,other_y):
        raise ValueError('کادر شامل هر دو تاریخ است؛ فقط نوشتهٔ «'+FIELD_NAMES[name]+'» را انتخاب کن.')
    if abs(center-other_y)+.04 < abs(center-own_y):
        raise ValueError('این کادر روی «'+FIELD_NAMES[other]+'» است. برای «'+FIELD_NAMES[name]+'» ردیف خودش را انتخاب کن یا نام فیلد را تغییر بده.')
