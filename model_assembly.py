"""Restore the unchanged model from four small download packages, verified by SHA256."""
import hashlib,json,os,tempfile
from pathlib import Path

def ensure_model(root):
    root=Path(root);target=root/'persian_crnn.onnx'
    expected=json.loads((root/'model_info.json').read_text(encoding='utf-8'))['sha256']
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==expected:return target
    chunks=[root/f'persian_crnn.part{i:02d}' for i in range(1,5)]
    missing=[str(p.name) for p in chunks if not p.is_file()]
    if missing:raise RuntimeError('مدل OCR پیدا نشد؛ در به‌روزرسانی، مدل نسخهٔ قبلی را نگه دارید یا بستهٔ FULL را استخراج کنید. در نسخهٔ چهارقسمتی هر چهار بخش لازم‌اند. بخش گمشده: '+', '.join(missing))
    handle=tempfile.NamedTemporaryFile(prefix='assembling_',suffix='.onnx',dir=root,delete=False)
    temp=Path(handle.name)
    try:
        digest=hashlib.sha256()
        with handle:
            for part in chunks:
                block=part.read_bytes();digest.update(block);handle.write(block)
        if digest.hexdigest()!=expected:raise RuntimeError('بخش‌های مدل سالم نیستند؛ فایل‌های ZIP را دوباره استخراج کنید.')
        os.replace(temp,target)
    finally:
        if temp.exists():temp.unlink()
    return target

if __name__=='__main__':
    print(ensure_model(Path(__file__).parent/'models'))
