import asyncio
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import service_api


@pytest.fixture
def client():
    return TestClient(service_api.app)


def picture():
    stream = io.BytesIO()
    Image.new('RGB', (640, 400), 'white').save(stream, 'PNG')
    return stream.getvalue()


def test_crop_and_thresholds_reach_reader(client, monkeypatch):
    observed = {}
    def analyze(data, **options):
        observed.update(options)
        return {'company': {'status': 'review', 'next_action': {'type': 'review_fields'}}}
    monkeypatch.setattr(service_api.reader, 'analyze', analyze)
    response = client.post('/v1/extract?min_confidence=0.95&selection_threshold=0.88',
        files={'file': ('test.png', picture(), 'image/png')},
        data={'card_box': '[0.1,0.1,0.9,0.9]', 'kind': 'old'})
    assert response.status_code == 200
    assert observed == dict(card_box=[.1,.1,.9,.9], kind='old', min_confidence=.95, selection_threshold=.88)


@pytest.mark.parametrize('box', ['invalid', 'null', '{}', '[0,0,1]', '[0,0,0,1]',
                               '[0,0,2,1]', '[0,0,NaN,1]', '[false,0,1,1]', '["0",0,1,1]'])
def test_invalid_crop_does_not_call_ocr(client, monkeypatch, box):
    monkeypatch.setattr(service_api.reader, 'analyze', lambda *a, **k: pytest.fail('OCR should not run'))
    response = client.post('/v1/extract', files={'file': ('test.png', picture())}, data={'card_box': box})
    assert response.status_code == 422


@pytest.mark.parametrize('threshold', ['nan', 'inf', '0.49', '1.01'])
def test_invalid_threshold_is_422(client, threshold):
    assert client.post('/v1/extract?min_confidence=' + threshold,
        files={'file': ('test.png', picture())}).status_code == 422


def test_empty_image_is_400(client):
    assert client.post('/v1/extract', files={'file': ('test.png', b'')}).status_code == 400


def test_upload_read_is_bounded():
    class Upload:
        async def read(self, size):
            assert size == service_api.MAX_BYTES + 1
            return b'x' * size
    with pytest.raises(service_api.HTTPException) as exc:
        asyncio.run(service_api._read_upload(Upload()))
    assert exc.value.status_code == 413


def test_busy_ocr_returns_retry_without_blocking_health(client):
    service_api._processing_lock.acquire()
    try:
        response = client.post('/v1/extract', files={'file': ('test.png', picture())})
        assert response.status_code == 503
        assert response.headers['retry-after'] == '2'
        assert client.get('/health').status_code == 200
    finally:
        service_api._processing_lock.release()


def test_disabled_primary_is_not_ready(client, monkeypatch):
    monkeypatch.setattr(service_api, 'engine_status', lambda: [
        dict(engine='Bina primary', ready=True, enabled=False),
        dict(engine='PP-OCRv5 primary', ready=True, enabled=True)])
    assert client.get('/ready').status_code == 503


def test_corrupt_image_returns_422(client):
    assert client.post('/v1/extract', files={'file': ('bad.png', b'broken')}).status_code == 422
