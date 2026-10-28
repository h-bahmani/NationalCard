"""Normalized export keeps accepted values separate from raw OCR evidence."""
from presentation import FIELD_NAMES,field_display,number_display


def export_result(result,kind=None):
    display=number_display(result)
    clear=display['status'] in ('cloud','candidate')
    record={'card_type':kind,'national_id':display['value'] if clear else None,
            'reading_status':display['status'],'checksum_valid':result.get('checksum_valid',False),
            'provider':result.get('provider','local'),'model':result.get('model','persian_crnn'),
            'fields':{},'elapsed_seconds':result.get('elapsed_seconds')}
    for key,data in result.get('fields',{}).items():
        if key not in FIELD_NAMES:continue
        shown=field_display(key,data)
        accepted=shown['status']=='read' and shown['value'] not in (None,'','—')
        record['fields'][key]={
            'value':shown['value'] if accepted else None,
            'status':'read' if accepted else 'unread',
            'raw_text':data.get('raw_selected_text') or data.get('text',''),
            'source':data.get('source','local'),
            'candidate_value':None,
            'suggested_value':data.get('suggested_text') or shown.get('suggested_value'),
            'decision':data.get('decision'),
            'auto_corrected':bool((data.get('correction') or {}).get('auto_corrected')),
            'correction_reason':(data.get('correction') or {}).get('reason'),
        }
    record['raw_result']=result
    from suggestions import attach_suggestions
    return attach_suggestions(record)
