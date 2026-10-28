"""One owned, serialized OCR subprocess retaining models, never image results.

The private stdin/stdout protocol has no network listener. Each request uses a
fresh temporary input/output file and a unique response token. A deadline,
crash or code/configuration change discards the child and all its model state.
"""
from __future__ import annotations
import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent


class OCRProcess:
    def __init__(self, script=None):
        self.script = Path(script) if script is not None else ROOT / 'ocr_server.py'
        self.lock = threading.Lock()
        self.process = None
        self.stderr = None
        self.responses = None
        self.signature = None

    def _signature(self):
        files = list(ROOT.glob('*.py')) + [ROOT / '.env']
        stamps = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.is_file())
        settings = tuple(sorted((k, v) for k, v in os.environ.items()
                                if k.startswith(('BINA_', 'PADDLE', 'NAME_CORRECTION', 'OCR_'))))
        return stamps, settings

    def _stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=10)
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe:
                    pipe.close()
        if self.stderr:
            self.stderr.close()
        self.process = self.stderr = self.responses = self.signature = None

    def close(self):
        with self.lock:
            self._stop()

    def _start(self, signature):
        self.stderr = tempfile.TemporaryFile()
        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONIOENCODING': 'utf-8'}
        self.process = subprocess.Popen(
            [sys.executable, '-u', str(self.script)], cwd=str(ROOT), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        responses = self.responses = queue.Queue()
        stdout = self.process.stdout
        def collect():
            try:
                for line in stdout:
                    responses.put(line.strip())
            finally:
                responses.put(None)
        threading.Thread(target=collect, daemon=True, name='ocr-response-reader').start()
        self.signature = signature

    def read(self, source, output, timeout, progress=None):
        with self.lock:
            signature = self._signature()
            reused = self.process is not None and self.process.poll() is None and signature == self.signature
            if not reused:
                self._stop()
                self._start(signature)
            token = uuid.uuid4().hex
            try:
                self.process.stdin.write(json.dumps({'token': token, 'source': str(source), 'output': str(output)}) + '\n')
                self.process.stdin.flush()
                deadline = time.monotonic() + timeout
                while True:
                    response = self.responses.get(timeout=max(0, deadline-time.monotonic()))
                    if response and response.startswith('{'):
                        event = json.loads(response)
                        if event.get('token') != token or 'progress' not in event:
                            raise RuntimeError('Unexpected OCR response')
                        if progress:
                            try:
                                progress('local_field_done', event['progress'])
                            except Exception:
                                pass
                        continue
                    break
                if response != token or not Path(output).is_file():
                    raise RuntimeError('OCR worker ended without a matching result')
                result = json.loads(Path(output).read_text(encoding='utf-8'))
                result.setdefault('models_reused', reused)
                return result
            except queue.Empty:
                self._stop()
                raise subprocess.TimeoutExpired('local OCR', timeout) from None
            except (OSError, ValueError, RuntimeError):
                self._stop()
                raise


worker = OCRProcess()
atexit.register(worker.close)
