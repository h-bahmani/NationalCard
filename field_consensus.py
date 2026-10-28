"""Keep local text and remote suggestions separate when they disagree."""
import re
from presentation import FIELD_NAMES,field_display
from validation import DIGIT_MAP

def reading_key(name,value):
    text=str(value or '').translate(DIGIT_MAP).translate(str.maketrans({'ي':'ی','ى':'ی','ك':'ک'}))
    return re.sub(r'[\s\u200c]','',text)

def merge_cloud_fields(local,remote,protected=()):
    fields=dict(local)
    for name,suggestion in remote.items():
        if name not in FIELD_NAMES or name in protected:continue
        suggestion=suggestion if isinstance(suggestion,dict) else {'text':str(suggestion)}
        cloud_text=str(suggestion.get('text') or '').strip()
        if not cloud_text:continue
        current=fields.get(name,{})
        local_text=str(current.get('text') or '').strip()
        shown=field_display(name,current)
        local_useful=shown['status'] in ('read','ambiguous') and shown['value']!='—'
        if name in ('birth_date','expiry_date'):
            local_useful=local_useful or sum(c.isdigit() for c in local_text)>=4
        if not local_useful:
            fields[name]={**suggestion,'ambiguous':True,'source':suggestion.get('source','cloud_vision'),
                'readings':list(suggestion.get('readings',[]))+[cloud_text],
                'cross_check':'no_usable_local_reading'}
            continue
        same=reading_key(name,local_text)==reading_key(name,cloud_text)
        fields[name]={**current,'readings':list(current.get('readings',[]))+[local_text,cloud_text],
            'ambiguous':bool(current.get('ambiguous')) or not same,
            'cloud_text':cloud_text,'cloud_disagreement':not same,
            'cross_check':'agreement' if same else 'disagreement_local_preserved'}
    return fields
