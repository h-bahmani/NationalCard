"""Image preparation with Pillow only; no model download or GPU dependencies."""
from __future__ import annotations
import base64
import hashlib
import io
import warnings

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

MAX_BYTES = 30 * 1024 * 1024
MAX_PIXELS = 60_000_000


SUPPORTED_EXTENSIONS=['jpg','jpeg','jpe','jfif','png','webp','bmp','dib','tif','tiff','gif','heic','heif','avif']
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

def image_info(data):
    if len(data)>MAX_BYTES:raise ValueError('حجم فایل باید کمتر از ۳۰ مگابایت باشد.')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width*im.height>MAX_PIXELS:raise ValueError('ابعاد تصویر بیشتر از ۶۰ مگاپیکسل است.')
            return {'format':im.format,'size':im.size,'frames':min(getattr(im,'n_frames',1),100)}
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning):
        raise ValueError('این فایل تصویر قابل خواندن نیست. برای HEIC/HEIF نصب requirements.txt باید کامل باشد؛ تغییر پسوند کافی نیست.') from None

def load_image(data:bytes,frame=0)->Image.Image:
    if len(data)>MAX_BYTES:raise ValueError('حجم عکس باید کمتر از ۳۰ مگابایت باشد.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if not 0<=frame<getattr(source,'n_frames',1):raise ValueError('صفحه یا فریم انتخاب‌شده وجود ندارد.')
                source.seek(frame)
                if source.width*source.height>MAX_PIXELS:raise ValueError('عکس را به کمتر از ۶۰ مگاپیکسل کاهش دهید.')
                if min(source.size) < 20:
                    raise ValueError('تصویر آن‌قدر کوچک است که حتی برای بازخوانی آزمایشی پیکسل کافی ندارد.')
                work=ImageOps.exif_transpose(source)
                original_size = list(work.size)
                # Do not reject small cards. Upscale them before detection/OCR so every
                # downstream stage receives a usable canvas. This is interpolation only;
                # it never invents characters and low-confidence OCR is still gated later.
                if min(work.size) < 320:
                    factor = min(8.0, 320.0 / max(1, min(work.size)))
                    work = work.resize((round(work.width * factor), round(work.height * factor)), Image.Resampling.LANCZOS)
                work.thumbnail((3600,3600),Image.Resampling.LANCZOS)
                image=work.convert('RGBA')
                canvas=Image.new('RGBA',image.size,'white');canvas.alpha_composite(image)
                result = canvas.convert('RGB')
                result.info['cardreader_original_size'] = original_size
                return result
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning):
        raise ValueError('تصویر سالم و قابل خواندن نیست. فرمت فایل و نصب کامل وابستگی‌ها را بررسی کنید.') from None


def fingerprint(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.size).encode('ascii'))
    digest.update(image.tobytes())
    return digest.hexdigest()


def crop_percent(image: Image.Image, horizontal: tuple[int, int], vertical: tuple[int, int]):
    left, right = horizontal
    top, bottom = vertical
    if not 0 <= left < right <= 100 or not 0 <= top < bottom <= 100:
        raise ValueError('محدودهٔ برش باید عرض و ارتفاع مثبت داشته باشد.')
    width, height = image.size
    box = (round(width * left / 100), round(height * top / 100),
           round(width * right / 100), round(height * bottom / 100))
    if box[2] - box[0] < 40 or box[3] - box[1] < 20:
        raise ValueError('ناحیهٔ انتخاب‌شده بیش از حد کوچک است.')
    return image.crop(box)


def crop_preview(image, horizontal, vertical):
    preview = image.copy()
    w, h = preview.size
    l, r = horizontal
    t, b = vertical
    ImageDraw.Draw(preview).rectangle((w*l/100, h*t/100, w*r/100, h*b/100),
                                      outline='#dc7c11', width=max(2, w//250))
    return preview


def data_url(image: Image.Image, crop=False) -> str:
    work = image.copy()
    if crop and work.width < 1200:
        factor = min(3, 1200 / work.width)
        work = work.resize((round(work.width * factor), round(work.height * factor)), Image.Resampling.LANCZOS)
    work.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    work.save(out, format='JPEG', quality=95)
    return 'data:image/jpeg;base64,' + base64.b64encode(out.getvalue()).decode('ascii')


def enhance_lighting(image):
    """Conservative luminance CLAHE, no learned reconstruction or new geometry."""
    import cv2
    import numpy as np
    lab=cv2.cvtColor(np.asarray(image.convert('RGB')),cv2.COLOR_RGB2LAB)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(lab[:,:,0])
    adjusted=Image.fromarray(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB))
    return Image.blend(image.convert('RGB'),adjusted,.65)
