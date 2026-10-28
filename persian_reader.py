"""CPU inference for the bundled Apache-2.0 Hezar Persian CRNN (ONNX conversion).
Preprocessing follows upstream: grayscale, mirror, resize 384x32, standardize.
"""
import json,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parent/'models'
class PersianReader:
    def __init__(self):
        from model_assembly import ensure_model
        ensure_model(ROOT)
        import onnxruntime as ort
        ort.disable_telemetry_events()
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning","0")
        self.session=ort.InferenceSession(str(ROOT/'persian_crnn.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.labels=json.loads((ROOT/'labels.json').read_text(encoding='utf-8'))
    def read(self,image):
        return self.read_scored(image)[0]
    def read_scored(self,image):
        gray=ImageOps.mirror(image.convert('L')).resize((384,32),Image.Resampling.BICUBIC)
        pixels=((np.asarray(gray,dtype=np.float32)/255-.6595)/.1501)[None,None,:,:]
        logits=self.session.run(None,{'pixels':pixels})[0][:,0,:]
        ids=logits.argmax(axis=-1);chars=[];last=None
        for index in ids:
            if int(index)!=last and int(index)!=0:chars.append(self.labels[str(int(index))])
            last=int(index)
        text=''.join(chars)
        text=re.sub(r'(\d+(?:\D\d+)*)',lambda m:m.group(1)[::-1],text)
        probabilities=np.exp(logits-logits.max(axis=-1,keepdims=True))
        probabilities/=probabilities.sum(axis=-1,keepdims=True)
        chosen=[float(probabilities[i,int(index)]) for i,index in enumerate(ids) if int(index)!=0 and (i==0 or int(index)!=int(ids[i-1]))]
        return text.strip(),float(np.mean(chosen)) if chosen else 0.0
