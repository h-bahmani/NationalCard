"""Try each configured service at most once, then let the pipeline use offline OCR."""
from groq_client import controlled_call,ProviderError,scrub
from openrouter_client import read_number,safe
from validation import normalize
from document_reader import read_document
class VisionRouter:
    manages_circuit=True
    def __init__(self,settings,mode='auto',circuit=None,groq=controlled_call,openrouter=read_number):
        self.settings=settings;self.circuit={} if circuit is None else circuit
        self.order=['groq','openrouter'] if mode=='auto' else [mode] if mode in ('groq','openrouter') else []
        self.groq=groq;self.openrouter=openrouter;self.attempts=[]
    def __call__(self,stage,work,key,**options):
        self.attempts=[]
        document=options.pop('document',None)
        for name in self.order:
            token=self.settings.get(name.upper()+'_API_KEY','')
            if not token:
                self.attempts.append({'provider':name,'status':'missing_key'});continue
            if self.circuit.get(name):
                self.attempts.append({'provider':name,'status':'circuit_open'});continue
            try:
                model=self.settings[name.upper()+'_VISION_MODEL']
                if document is not None:result=read_document(name,document,work,token,model,**options)
                elif name=='groq':result=self.groq('vision',work,token,vision_model=model,**options)
                else:result=self.openrouter(work,token,model=model,**options)
                if not result.get('readable') or normalize(result.get('raw')) is None:
                    self.attempts.append({'provider':name,'status':'partial_reading'});continue
                self.attempts.append({'provider':name,'status':'success','model':result.get('model',model)})
                return {**result,'provider':name}
            except Exception as error:
                message=str(error) if isinstance(error,ProviderError) else 'خطای ارتباط یا پاسخ سرویس'
                for secret in (self.settings.get('GROQ_API_KEY'),self.settings.get('OPENROUTER_API_KEY')):message=safe(message,secret)
                self.attempts.append({'provider':name,'status':'error','http_status':getattr(error,'http_status',None),'message':message})
                self.circuit[name]=True
        raise ProviderError('خواندن از سرویس‌های انتخاب‌شده کامل نشد؛ مسیر محلی استفاده شد.','services_unavailable')
