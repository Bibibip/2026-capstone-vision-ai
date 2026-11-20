# Streaming Vision Batch Frames API

`POST /vision/frames` is additive.  `POST /vision/frame` and every existing
start/complete/status/result/callback contract remain valid and unchanged.

## Request

Content type is `multipart/form-data`.

| part | type | required | rule |
|---|---|---:|---|
| `sessionId` | text | yes | Existing started session ID. |
| `manifest` | JSON text | yes | Batch identity and ordered frame metadata. |
| `frame_<index>` | JPEG file | yes | One file for every `manifest.frames[].part`. |

The service accepts 1 to 5 frames by default.  The server-side hard limit is
`STREAMING_POC_MAX_BATCH_FRAMES` (default 5; deployment may lower it).

```json
{
  "batchId": "a0e3a2c7-0b85-4d5b-b296-3df8d207c8d6",
  "frames": [
    {
      "part": "frame_000120",
      "frameIndex": 120,
      "sourceFrameIndex": 120,
      "capturedAt": "2026-11-04T14:00:04.000+09:00",
      "sha256": "lowercase-64-character-sha256-of-the-exact-jpeg-bytes"
    }
  ]
}
```

`batchId` must be a UUID.  `frameIndex` must be non-negative, ascending, and
contiguous within the batch.  `sourceFrameIndex` is optional and defaults to
`frameIndex`; `capturedAt` is optional and may be null.  Every file must declare
`image/jpeg` or `image/jpg`.  The SHA-256 is required: Vision calculates it
after reading the upload and rejects a mismatch.

## Atomic acceptance and retry

Vision validates every multipart part and every declared SHA-256 before it
looks up or changes the session.  Under the session lock it then requires:

1. batch first index equals `nextExpectedFrameIndex`;
2. every relevant worker queue has room for all frames in the batch;
3. the session is still open.

Only then are all frame items inserted in MediaPipe, YOLO, and, for AUDIO,
AUDIO queues.  Workers consume only (they cannot add queue items), so capacity
preflight plus the producer lock prevents an ordinary `queue.Full` race.  The
session index and accepted-batch record are updated only after all inserts.
No inference or callback is awaited by this HTTP request.

The accepted-batch record stores the ordered tuple of frame index,
source-frame index, captured timestamp, and SHA-256 -- never JPEG bytes.  A
retry must reuse the exact `batchId`, manifest, and JPEG bytes.  It returns a
202 duplicate only when all of those values match.  A changed payload with the
same batchId is a 409 conflict.

| case | response | Pi action |
|---|---:|---|
| Whole new batch accepted | 202 `accepted` | Delete acknowledged local spool files; send next batch. |
| Exact retry of accepted batch | 202 `duplicate` | Treat as success; delete acknowledged spool files. |
| Gap, overlap, old unknown range, changed batchId payload, closed input | 409 | Read `nextExpectedFrameIndex`; never skip a frame. |
| Complete batch cannot fit in any worker queue | 429 | Keep all files; retry the same batchId and bytes with bounded backoff. |
| JPEG MIME type invalid | 415 | Do not retry unchanged. |
| Manifest, index, part, empty file, or SHA mismatch invalid | 422 | Do not retry unchanged. |

There is no partial success response.  A 409, 415, 422, or 429 does not advance
the expected frame index or enqueue any member of that batch.

## Responses

Accepted:

```json
{
  "sessionId": "pi-9b604d36",
  "batchId": "a0e3a2c7-0b85-4d5b-b296-3df8d207c8d6",
  "status": "accepted",
  "firstFrameIndex": 120,
  "lastFrameIndex": 122,
  "acceptedFrameCount": 3,
  "duplicateFrameCount": 0,
  "nextExpectedFrameIndex": 123
}
```

Exact retry:

```json
{
  "sessionId": "pi-9b604d36",
  "batchId": "a0e3a2c7-0b85-4d5b-b296-3df8d207c8d6",
  "status": "duplicate",
  "firstFrameIndex": 120,
  "lastFrameIndex": 122,
  "acceptedFrameCount": 0,
  "duplicateFrameCount": 3,
  "nextExpectedFrameIndex": 123
}
```

Error responses are FastAPI `{"detail":"..."}` objects.  For 409 and 429,
Pi should query `GET /vision/status/{sessionId}` to obtain the authoritative
`nextExpectedFrameIndex` if a transport failure makes the prior response
ambiguous.

## Pi reference implementation

```python
import hashlib
import json
import uuid

def build_batch(session_id, spooled_frames):
    # spooled_frames: ordered records with index/source/captured_at/path.
    # Read exact retained bytes; never re-encode a retry.
    batch_id = str(uuid.uuid4())
    manifest_frames, files = [], []
    for item in spooled_frames:
        jpeg = item["path"].read_bytes()
        part = f"frame_{item['frame_index']:06d}"
        manifest_frames.append({
            "part": part,
            "frameIndex": item["frame_index"],
            "sourceFrameIndex": item["source_frame_index"],
            "capturedAt": item["captured_at"],
            "sha256": hashlib.sha256(jpeg).hexdigest(),
        })
        files.append((part, (part + ".jpg", jpeg, "image/jpeg")))
    return batch_id, {
        "sessionId": session_id,
        "manifest": json.dumps({"batchId": batch_id, "frames": manifest_frames}, separators=(",", ":")),
    }, files

# Keep batch_id, data, and files/JPEG bytes (or their spool paths) unchanged
# until a 202 accepted or duplicate response arrives.
batch_id, data, files = build_batch(session_id, next_three_spooled_frames)
response = http_session.post(vision_base + "/vision/frames", data=data, files=files,
                             timeout=(3.05, 10))
```

## Pi storage and completion policy

An in-memory 30-frame queue is not a no-loss strategy.  If a 7-second capture
produces 210 JPEGs faster than the network drains them, it fills after about one
second and either blocks capture or drops frames.  Pi must use a per-session
disk spool:

1. At capture time, JPEG-encode and atomically write
   `spool/<sessionId>/<frameIndex>.jpg`, plus index/source/capture timestamp
   and SHA-256 metadata.
2. Enqueue only lightweight file descriptors in RAM.  The sender reads three
   ordered files, makes one batch, and retains them until 202 accepted or
   duplicate.
3. Delete a spool file only after that successful response.  Retries reuse the
   same file bytes and batchId; they never re-encode a frame.
4. Reserve free storage before capture for at least
   `2 * 210 * measured_p95_jpeg_bytes` plus MP4 recording needs.  Measure this
   on the deployed 1280x720 quality setting; do not substitute the localhost
   960x540 measurement.
5. If spool write, capacity, or bounded retry fails, preserve MP4 recording,
   mark streaming unavailable, and take the existing MP4 fallback exactly once.

Call `/vision/complete` only after every expected frame has a 202 accepted or
duplicate result.  Calling complete after an unrecoverable upload gap can make
an incomplete prefix appear as a normal shorter input; the current API has no
abort endpoint or declared expected-frame count.

## Batch size

Batch 3 is the default recommendation.  At 30 FPS its first image waits at
most about 67 ms for the third image; batch 5 waits at most about 133 ms.  Batch
5 further reduces request count, but increases that early-event delay and the
instantaneous worker queue burst.  Select batch 5 only after actual Pi/tunnel
measurements show batch 3 cannot meet the delivery target without 429/backlog.

## Optional timing diagnostics

Timing is off by default and changes neither acceptance nor worker behavior.
For a diagnostic-only server process, set:

```powershell
$env:STREAMING_POC_BATCH_TIMING_LOG = "true"
```

Each `/vision/frames` request writes one structured `streaming_batch_timing`
record to the Uvicorn error log. It contains only `sessionId`, `batchId`, frame
count, total JPEG byte count, HTTP status, and millisecond intervals. It never
logs JPEG content, multipart headers, callback keys, Authorization values, or
tokens.

Important fields are `asgiBodyReceiveMs` (ASGI receive duration),
`handlerToFormMs`, `formAfterBodyMs`, `jpegReadMs`, `sha256Ms`,
`sessionLockWaitMs`, `enqueueMs`, and `handlerTotalMs`. `response_ready` is
the point immediately before the handler returns its 202/error response; it is
not a client-side receipt timestamp.
