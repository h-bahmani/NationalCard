# AiNationalCard

AiNationalCard is an API-based Iranian National ID Card detection and OCR service designed for local and LAN deployment.

The service receives an image, detects the National ID Card, rectifies its perspective, extracts identity fields, validates the results, and returns a structured response suitable for integration with internal applications and services.

## Features

- Iranian National ID Card detection
- Automatic card localization and perspective correction
- Automatic field extraction
- Persian OCR with multiple recognition paths
- National ID checksum validation
- Field confidence and review status
- Fast and deep processing modes
- Concurrent API request processing
- Unique tracking ID for each request
- Health and readiness endpoints
- Offline-capable runtime after installation
- LAN deployment support
- Structured JSON responses

## Extracted Fields

The service currently focuses on the following fields:

- National ID
- First Name
- Last Name
- Father's Name
- Date of Birth
- Expiry Date

Fields that cannot be read with sufficient confidence are returned for review instead of being silently guessed.

## Architecture

```text
Input Image
    |
    v
Card Detection
    |
    v
Card Verification
    |
    v
Perspective Rectification
    |
    v
Field Localization
    |
    v
OCR Engines
    |
    v
Consensus & Validation
    |
    v
Structured API Response
```

The OCR pipeline keeps literal recognition evidence and uses validation and agreement between recognition paths to reduce incorrect automatic acceptance.

National ID checksum validation is used as a validation signal and is not used to invent or repair digits.

## API

Default local address:

```text
http://127.0.0.1:8510
```

For LAN deployment, the service can listen on:

```text
0.0.0.0:8510
```

### Health Check

```http
GET /health
```

Returns service and OCR engine status.

### Readiness Check

```http
GET /ready
```

Indicates whether the service is ready to accept OCR requests.

### Extract National ID Card

```http
POST /v1/extract
```

The request accepts an image file using `multipart/form-data`.

Example with Python:

```python
import requests

url = "http://127.0.0.1:8510/v1/extract"

headers = {
    "X-API-Key": "YOUR_API_KEY"
}

with open("card.jpg", "rb") as image:
    response = requests.post(
        url,
        headers=headers,
        files={"file": image},
        timeout=120,
    )

print(response.json())
```

## Processing Modes

### Fast Mode

Fast mode is the default processing path and should be used for normal API traffic.

```http
POST /v1/extract?deep=false
```

### Deep Mode

Deep mode is intended for explicit recovery attempts when the normal path cannot obtain sufficient evidence.

```http
POST /v1/extract?deep=true
```

Deep mode may require more processing time.

## Concurrency

AiNationalCard supports concurrent request processing through multiple service workers.

A real integration test with four simultaneous OCR requests successfully completed all four requests with independent tracking IDs and overlapping execution.

The appropriate worker count depends on the CPU and memory available on the target machine.

Example:

```powershell
$env:CARDREADER_WORKERS="4"
.\RUN_COMPANY_SERVICE.bat
```

Higher worker counts increase concurrent capacity but may increase the latency of individual OCR requests.

## Request Tracking

Each request can be associated with a unique tracking identifier.

The tracking ID can be used to correlate API responses with server-side operational logs without storing sensitive card information in normal request logs.

Example response header:

```text
X-Track-ID: <unique-request-id>
```

## Review Handling

OCR is not treated as infallible.

When available evidence is insufficient or recognition engines disagree, the response can indicate that manual review is required.

Typical cases include:

- OCR engine disagreement
- Low-quality field image
- Missing field reading
- Invalid National ID checksum
- Invalid or incomplete date
- Insufficient recognition confidence

Consumers should use the structured review state instead of assuming every returned OCR string is verified.

## Security

Do not commit secrets or personal identity data to the repository.

The following files and data must remain outside Git:

```text
.env
API keys
real National ID Card images
OCR result files containing personal data
runtime logs containing sensitive values
temporary output files
```

Use `.env.example` only for documenting required environment variables.

## Repository Structure

```text
AiNationalCard/
├── service_api.py
├── company_api.py
├── cardreader_module.py
├── detector.py
├── card_verifier.py
├── pipeline.py
├── local_worker.py
├── ocr_process.py
├── ocr_server.py
├── ocr_ensemble.py
├── persian_reader.py
├── names_reader.py
├── regions.py
├── validation.py
├── presentation.py
├── quality.py
├── quality_recovery.py
│
├── models/
├── templates/
├── tests/
├── tools/
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── RUN_COMPANY_SERVICE.bat
```

Development artifacts, temporary repair scripts, generated reports, test outputs, real card images, virtual environments, and runtime caches should not be committed to the repository.

## Running the Service

From the project directory:

```powershell
cd "C:\path\to\AiNationalCard"
```

Start the API:

```powershell
.\RUN_COMPANY_SERVICE.bat
```

Or configure the worker count before startup:

```powershell
$env:CARDREADER_WORKERS="4"
.\RUN_COMPANY_SERVICE.bat
```

Open the API documentation:

```text
http://127.0.0.1:8510/docs
```

Check service health:

```text
http://127.0.0.1:8510/health
```

## Production Notes

Before deployment:

- Keep API credentials outside Git
- Use an appropriate worker count for the target hardware
- Restrict network access to trusted systems
- Configure the required Windows firewall rule for LAN access
- Run regression tests after OCR-related changes
- Keep the validated OCR core unchanged unless a specific regression has been reproduced
- Monitor request latency and service errors under concurrent load

## Project Principle

AiNationalCard follows a conservative identity-processing principle:

> Prefer an explicit review state over silently returning an unsupported identity value.

The service is designed to preserve OCR evidence, validate structured fields, and keep uncertainty visible to downstream systems.