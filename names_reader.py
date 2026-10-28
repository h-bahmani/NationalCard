"""PaddleOCR PP-OCRv3 Arabic-script recognizer, used only for names.

ONNX conversion by monkt/paddleocr-onnx; Apache-2.0. No number reading
or digit repair. Model scores are uncalibrated and not comparable to Hezar.
"""
from pathlib import Path
import re
import cv2
import numpy as np

def name_key(text):
    return re.sub(r'[\s\u200c]','',text.translate(str.maketrans({'ي':'ی','ى':'ی','ك':'ک'})))

def valid_name(text):
    return sum(c.isalpha() for c in text)>=2 and all(c.isalpha() or c.isspace() or c in '\u200c-' for c in text)

class NamesReader:
    def __init__(self):
        import onnxruntime as ort
        ort.disable_telemetry_events()
        root=Path(__file__).parent/'models'
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        self.session=ort.InferenceSession(str(root/'paddle_names.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.labels=['blank']+(root/'paddle_names_dict.txt').read_text(encoding='utf-8').splitlines()+[' ']
    def read_scored(self,image):
        rgb=np.asarray(image.convert('RGB'));h,w=rgb.shape[:2]
        width=max(16,min(1024,round(48*w/h)))
        pixels=cv2.resize(rgb[:,:,::-1],(width,48)).astype(np.float32)/127.5-1
        padded=np.zeros((48,max(320,width),3),np.float32);padded[:,:width]=pixels
        out=self.session.run(None,{'x':padded.transpose(2,0,1)[None]})[0][0]
        ids=out.argmax(-1);chars=[];scores=[]
        for i,j in enumerate(ids):
            if j and (i==0 or j!=ids[i-1]):chars.append(self.labels[int(j)]);scores.append(float(out[i,j]))
        # Reverse Arabic code points, preserving complete Latin/number runs,
        # as in upstream Arabic CTC postprocessing. Normalize glyph variants.
        blocks=re.findall(r'[a-zA-Z0-9 :*./%+\-]+|.', ''.join(chars))
        text=''.join(reversed(blocks)).strip().translate(str.maketrans({'ي':'ی','ى':'ی','ك':'ک'}))
        return text.strip(' .،,:;؛()[]'),float(np.mean(scores)) if scores else 0.0

def reconcile_name(primary_views,secondary_views):
    """Keep a strong primary; use stable second-model reading for weak primary.

    Internal scores are used only within their own model, not as cross-model
    accuracy probabilities. Disagreement remains visible in the output.
    """
    from collections import Counter
    primary=primary_views[0] if primary_views else ('',0)
    valid_primary=[(text,score) for text,score in primary_views if valid_name(text) and score>=.7]
    selected=primary if valid_name(primary[0]) and primary[1]>=.9 else max(valid_primary,key=lambda v:v[1],default=('',0))
    second=[(text,score) for text,score in secondary_views if valid_name(text) and score>=.8]
    counts=Counter(name_key(text) for text,_ in second)
    stable=[(text,score) for text,score in second if counts[name_key(text)]>=2]
    stable_second=max(stable,key=lambda v:v[1],default=('',0))
    primary_counts=Counter(name_key(text) for text,score in valid_primary)
    strong_primary=valid_name(primary[0]) and primary[1]>=.9 and primary_counts[name_key(primary[0])]>=2
    if not strong_primary and stable_second[0] and stable_second[1]>=.9:
        selected=stable_second;source='paddle_names'
    else:source='persian_crnn'
    disagreement=bool(valid_name(primary[0]) and stable_second[0] and name_key(primary[0])!=name_key(stable_second[0]))
    models_agree=bool(selected[0] and stable_second[0] and name_key(selected[0])==name_key(stable_second[0]) and any(name_key(text)==name_key(selected[0]) for text,_ in valid_primary))
    ambiguous=not models_agree or disagreement
    return selected[0],ambiguous,source,disagreement
