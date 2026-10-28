from pathlib import Path
import ocr_ensemble


def test_bina_does_not_override_the_model_preprocessing_width(monkeypatch):
    import static_ctc
    called=[]
    def reader(path,**options):
        called.append((Path(path),options))
        return object()
    monkeypatch.setattr(static_ctc,'StaticCTCReader',reader)
    ocr_ensemble.BinaReader()
    assert len(called)==1
    assert 'min_width' not in called[0][1]
    # The source of the default is model metadata, not a supported dynamic
    # tensor bound. Read the shipped configuration so model updates stay valid.
    import yaml
    metadata=yaml.safe_load((called[0][0]/'inference.yml').read_text(encoding='utf-8'))
    shape=next(x['RecResizeImg']['image_shape'] for x in metadata['PreProcess']['transform_ops'] if 'RecResizeImg' in x)
    assert shape[2] >= 320
