"""Local Paddle static-graph line OCR without model-registry or network lookups.

Preprocessing and CTC vocabulary follow each model's published inference.yml.
Only literal decoded characters are returned; no spelling or digit correction.
"""
from __future__ import annotations
import math
import re
from pathlib import Path


def logical_persian(text):
    # Preserve ASCII number/date runs while restoring Persian logical order.
    return ''.join(reversed(re.findall(r'[A-Za-z0-9 :*./%+\-]+|.', str(text), re.S)))


def decode_ctc(predictions, characters):
    import numpy as np
    scores=np.asarray(predictions)
    if scores.ndim!=3 or scores.shape[0]!=1 or scores.shape[2]!=len(characters):
        raise ValueError('OCR output shape does not match its vocabulary')
    scores=scores[0]
    indices=scores.argmax(axis=1)
    chosen=[i for i,index in enumerate(indices) if index!=0 and (i==0 or index!=indices[i-1])]
    if not chosen:
        return '',0.0
    return ''.join(characters[int(indices[i])] for i in chosen),float(np.mean([scores[i,indices[i]] for i in chosen]))


class StaticCTCReader:
    def __init__(self, directory, *, device='cpu', min_width=None, logical_mode='persian'):
        import yaml
        from paddle.inference import Config,create_predictor
        directory=Path(directory)
        metadata=yaml.safe_load((directory/'inference.yml').read_text(encoding='utf-8'))
        operations=metadata['PreProcess']['transform_ops']
        self.shape=next(op['RecResizeImg']['image_shape'] for op in operations if 'RecResizeImg' in op)
        self.color=next(op['DecodeImage'].get('img_mode','BGR') for op in operations if 'DecodeImage' in op)
        self.characters=['']+list(metadata['PostProcess']['character_dict'])+[' ']
        self.min_width=min_width or self.shape[2]
        self.logical_mode=logical_mode
        config=Config(str(directory/'inference.json'),str(directory/'inference.pdiparams'))
        if device.startswith('gpu:'):
            config.enable_use_gpu(256,int(device.split(':')[1]))
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(2)
        config.disable_glog_info()
        # Verified on this Windows runtime: oneDNN fusion raises an invalid-shape
        # error on the custom graph. Keep the tested plain CPU execution path.
        if hasattr(config,'disable_mkldnn'):
            config.disable_mkldnn()
        self.predictor=create_predictor(config)
        self.input=self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self.output=self.predictor.get_output_handle(self.predictor.get_output_names()[0])

    def read_scored(self,image):
        import cv2
        import numpy as np
        if min(image.size)<=0:
            raise ValueError('Empty OCR crop')
        channels,height,_=self.shape
        natural=math.ceil(height*image.width/image.height)
        width=min(3200,max(self.min_width,32*math.ceil(natural/32)))
        resized_width=min(width,natural)
        pixels=np.asarray(image.convert('RGB'))
        if self.color=='BGR':
            pixels=pixels[:,:,::-1]
        pixels=cv2.resize(pixels,(resized_width,height)).astype(np.float32).transpose(2,0,1)
        normalized=(pixels/255.0-.5)/.5
        batch=np.zeros((1,channels,height,width),dtype=np.float32)
        batch[0,:,:,:resized_width]=normalized
        self.input.copy_from_cpu(batch)
        self.predictor.run()
        visual,confidence=decode_ctc(self.output.copy_to_cpu(),self.characters)
        if self.logical_mode=='persian':
            return logical_persian(visual),confidence
        if self.logical_mode=='bidi':
            from bidi.algorithm import get_display
            return get_display(visual),confidence
        return visual,confidence
