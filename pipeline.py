"""Independent offline extraction with bounded subprocess lifetime and cloud circuit breaker."""
import json,os,subprocess,sys,tempfile,time
from pathlib import Path
from validation import normalize,checksum_valid
from groq_client import controlled_call,ProviderError
ROOT=Path(__file__).resolve().parent
# Template-aligned field regions, excluding printed labels and portrait.


def _local_needs_cloud(fields,candidate):
    """Cloud assist is useful only when accepted local evidence is incomplete/ambiguous."""
    if not candidate:
        return True
    national=(fields or {}).get('national_id',{})
    if national.get('ambiguous'):
        return True
    required=('first_name','last_name','father_name','birth_date')
    for name in required:
        row=(fields or {}).get(name,{})
        if not row.get('text') or row.get('ambiguous'):
            return True
    return False

REGIONS={
 'smart':{'national_id':(.57,.25,.83,.34),'first_name':(.50,.35,.82,.46),
          'last_name':(.60,.46,.82,.57),'birth_date':(.57,.60,.82,.69),
          'father_name':(.60,.70,.82,.79),'expiry_date':(.57,.80,.82,.89)},
 'old':{'national_id':(.423,.125,.825,.255),'first_name':(.43,.25,.82,.35),
        'last_name':(.43,.36,.82,.46),'father_name':(.43,.47,.82,.57),
        'document_number':(.43,.58,.82,.68),'birth_place':(.43,.69,.82,.79)}}

def local_read(image,kind,timeout=180,runner=subprocess.run,regions=None,progress=None):
    if regions is None:
        from regions import number_region
        _,box=number_region(image,kind)
        regions={**REGIONS[kind],'national_id':box}
    try:
        with tempfile.TemporaryDirectory(prefix='national_ocr_') as directory:
            source=Path(directory)/'card.png';output=Path(directory)/'result.json'
            image.save(source)
            Path(str(source)+'.json').write_text(json.dumps({**(regions or REGIONS[kind]),'__kind__':kind}),encoding='utf-8')
            env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONIOENCODING':'utf-8'}
            if runner is subprocess.run:
                from ocr_process import worker
                return worker.read(source,output,timeout,progress=progress)
            process=runner([sys.executable,str(ROOT/'local_worker.py'),str(source),str(output)],
                           capture_output=True,timeout=timeout,env=env)
            if process.returncode!=0 or not output.exists():
                return {'ok':False,'error':'worker_crashed','message':'اجرای OCR محلی متوقف شد. برنامه را با START_WINDOWS.bat همین نسخه اجرا کنید.',
                        'diagnostic':str(getattr(process,'stderr',b''))[-1600:]}
            return json.loads(output.read_text(encoding='utf-8'))
    except subprocess.TimeoutExpired:
        return {'ok':False,'error':'timeout','message':'زمان پردازش محلی تمام شد. می‌توانید دوباره تلاش کنید.'}
    except (OSError,ValueError,RuntimeError):
        return {'ok':False,'error':'worker_failed','message':'خواندن محلی کامل نشد؛ دسترسی پوشهٔ موقت و نصب مدل محلی را بررسی کنید.'}

def extract(aligned,roi,kind,key='',cloud=False,details=True,circuit=None,force=False,cache=None,
            local=local_read,remote=controlled_call,stats=None,cloud_fields=False,number_box=None,field_boxes=None,protected_fields=None,progress=None,cloud_if_needed=False):
    start=time.monotonic();circuit={} if circuit is None else circuit
    protected=set(field_boxes or {}) if protected_fields is None else set(protected_fields)
    def run_local():
        options = {'progress': progress} if local is local_read and progress else {}
        if number_box is None and not field_boxes:return local(aligned,kind,**options)
        regions={**REGIONS[kind],**(field_boxes or {})}
        if number_box is not None:regions['national_id']=number_box
        return local(aligned,kind,regions=regions,**options)
    result={'national_id':None,'candidate':None,'fields':{},'errors':[],'source':'local',
            'checksum_valid':False,'cloud_skipped':False}
    # Offline path runs even without a key, or after an earlier provider failure.
    local_attempted=bool(field_boxes or details or not cloud or not key or circuit.get('blocked'))
    if local_attempted and progress:
        try: progress('local_start', {'kind': kind})
        except Exception: pass
    local_result=run_local() if local_attempted else {'ok':True,'fields':{}}
    if local_attempted and progress:
        try: progress('local_done', {'ok': bool(local_result.get('ok')), 'elapsed_seconds': local_result.get('elapsed_seconds'), 'models_reused': local_result.get('models_reused')})
        except Exception: pass
    if local_result.get('ok'):
        result['local_runtime'] = {k: local_result[k] for k in ('elapsed_seconds', 'models_reused') if k in local_result}
        result['fields']=local_result.get('fields',{})
        raw=result['fields'].get('national_id',{}).get('text','')
        result['candidate']=normalize(raw)
    else:result['errors'].append({'stage':'local','code':local_result.get('error'),'message':local_result.get('message'),
                                'diagnostic':local_result.get('diagnostic','')})
    need_cloud = (not cloud_if_needed) or _local_needs_cloud(result.get('fields',{}), result.get('candidate'))
    if cloud and key and not circuit.get('blocked') and need_cloud:
        try:
            if progress:
                try: progress('cloud_start', {})
                except Exception: pass
            options={'force':force,'cache':cache,'stats':stats}
            if cloud_fields:options['document']=aligned
            read=remote('vision',roi,key,**options)
            if progress:
                try: progress('cloud_done', {'provider': read.get('provider','groq'), 'model': read.get('model')})
                except Exception: pass
            value=normalize(read['raw']) if read['readable'] else None
            if value:
                result['national_id']=value;result['source']='cloud'
                local_value=result['candidate']
                result['disagreement']=bool(local_value and local_value!=value)
                result['cloud_raw']=read['raw']
                result['local_fields']=dict(result['fields'])
                if isinstance(read.get('fields'),dict):
                    from field_consensus import merge_cloud_fields
                    result['cloud_fields']=read['fields']
                    result['fields']=merge_cloud_fields(result['fields'],read['fields'],protected)
                result['provider']=read.get('provider','groq');result['model']=read.get('model')
            else:result['errors'].append({'stage':'cloud','code':'partial','message':'مدل تصویر عدد کامل نخواند.'})
        except Exception as error:
            if progress:
                try: progress('cloud_error', {'error': str(error)})
                except Exception: pass
            result['errors'].append({'stage':'cloud','code':getattr(error,'code',type(error).__name__),
                 'http_status':getattr(error,'http_status',None),'message':str(error) if isinstance(error,ProviderError) else 'ارتباط با سرویس کامل نشد.'})
            
            if not getattr(type(remote),'manages_circuit',False):circuit['blocked']=True
    elif cloud:
        result['cloud_skipped']=True
        result['cloud_skip_reason']='local_complete' if key and not circuit.get('blocked') and not need_cloud else 'unavailable'
        if progress and result['cloud_skip_reason']=='local_complete':
            try: progress('cloud_skip', {'reason':'local_complete'})
            except Exception: pass
    if not result['national_id'] and not local_attempted:
        fallback=run_local()
        if fallback.get('ok'):
            result['fields']=fallback.get('fields',{})
            result['candidate']=normalize(result['fields'].get('national_id',{}).get('text',''))
        else:result['errors'].append({'stage':'local','code':fallback.get('error'),'message':fallback.get('message')})
    # A local OCR candidate is never promoted by checksum alone.
    result['checksum_valid']=checksum_valid(result['national_id'] or result['candidate'])
    attempts=getattr(remote,'attempts',[])
    result['provider_attempts']=attempts if isinstance(attempts,list) else []
    result['elapsed_seconds']=round(time.monotonic()-start,2)
    return result
