"""Private child entrypoint: retain only recognizers between independent requests."""
import contextlib
import json
from pathlib import Path
import sys
import time


def serve():
    readers = None
    for line in sys.stdin:
        request = json.loads(line)
        start = time.monotonic()
        reused = readers is not None
        try:
            # Libraries can log to stdout. Keep that out of the response protocol.
            with contextlib.redirect_stdout(sys.stderr):
                from local_worker import run
                if readers is None:
                    from persian_reader import PersianReader
                    from names_reader import NamesReader
                    from ocr_ensemble import build_optional_readers
                    optional, errors = build_optional_readers()
                    readers = (PersianReader(), NamesReader(), optional, errors)
                def progress(payload):
                    sys.__stdout__.write(json.dumps({'token': request['token'], 'progress': payload}) + '\n')
                    sys.__stdout__.flush()
                fields = run(request['source'], readers=readers, progress=progress)
            result = {'ok': True, 'fields': fields, 'elapsed_seconds': round(time.monotonic() - start, 3), 'models_reused': reused}
        except Exception as exc:
            result = {'ok': False, 'error': type(exc).__name__, 'message': str(exc)[:700]}
            readers = None
        Path(request['output']).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
        print(request['token'], flush=True)


if __name__ == '__main__':
    serve()
