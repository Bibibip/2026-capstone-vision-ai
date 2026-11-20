"""Isolated Streaming API PoC: online Silent Signal EARLY_EVENT only.

Run on a port other than 8000.  This module never imports or alters
api.vision_api.  It is deliberately not a replacement for the complete MP4
Vision API: AUDIO, Rear Approach, and complete legacy final-event assembly are
not included yet.
"""
from __future__ import annotations

import hashlib, json, logging, os, queue, re, sys, threading, time, urllib.error, urllib.parse, urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "hand_motion_desktop/src")]
sys.path.insert(0, str(ROOT / "final_demo/src"))
from detector import _build_detector, MIN_FIST_FRAMES, REQUIRED_REPEATS, MAX_GAP_BETWEEN_REPS_FRAMES
from detector_front import FRONT_MAX_TRANSITION_GAP_FRAMES
from gesture_rules import RepeatedFistToOpenTracker, frame_gesture_from_hands
from gesture_rules_front import classify_gesture_front
from ultralytics import YOLO
import torch
from experiments.streaming_poc.streaming_api_poc.adapters import (
    OnlineOcclusionAdapter, final_audio_events, final_keypad_events, source_is_contiguous,
)

app = FastAPI(title="OmniGuardy Streaming Silent Signal PoC", version="0.1")
END = object()
ALLOWED_CALLBACK_HOSTS = {x.strip() for x in os.getenv("STREAMING_POC_CALLBACK_HOSTS", "127.0.0.1,localhost").split(",") if x.strip()}
CALLBACK_BEARER_TOKEN = os.getenv("STREAMING_POC_CALLBACK_BEARER_TOKEN")
CALLBACK_KEY = os.getenv("STREAMING_POC_CALLBACK_KEY")
QUEUE_CAPACITY = max(1, int(os.getenv("STREAMING_POC_QUEUE_CAPACITY", "300")))
COMPLETED_SESSION_TTL_SECONDS = max(1.0, float(os.getenv("STREAMING_POC_COMPLETED_SESSION_TTL_SECONDS", "3600")))
MAX_BATCH_FRAMES = max(1, min(32, int(os.getenv("STREAMING_POC_MAX_BATCH_FRAMES", "5"))))
BATCH_PART_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BATCH_TIMING_ENABLED = os.getenv("STREAMING_POC_BATCH_TIMING_LOG", "false").strip().lower() in {"1", "true", "yes", "on"}
# Keep normal responses compact.  Turn this on only while investigating a
# camera-occlusion case; it adds one metric object per decoded AUDIO frame.
OCCLUSION_FRAME_METRICS_ENABLED = os.getenv("STREAMING_POC_OCCLUSION_FRAME_METRICS", "false").strip().lower() in {"1", "true", "yes", "on"}
# Uvicorn configures this logger with its standard server-output handler.  Use
# it only when timing is explicitly enabled; do not add a global handler.
BATCH_TIMING_LOGGER = logging.getLogger("uvicorn.error")


class BatchTimingASGIMiddleware:
    """Optionally retain receive() timestamps without reading request content."""
    def __init__(self, app: Any): self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if not BATCH_TIMING_ENABLED or scope.get("type") != "http" or scope.get("path") != "/vision/frames":
            await self.app(scope, receive, send); return

        async def timed_receive() -> Any:
            message = await receive()
            if message.get("type") == "http.request":
                now = time.perf_counter()
                scope.setdefault("_batch_timing_body_first", now)
                if not message.get("more_body", False): scope["_batch_timing_body_last"] = now
            return message

        await self.app(scope, timed_receive, send)


class BatchTiming:
    """Safe per-request timing record; it never keeps JPEG bytes or headers."""
    def __init__(self, request: Request):
        self.enabled = BATCH_TIMING_ENABLED; self.scope = request.scope; self.started = time.perf_counter()
        self.marks: dict[str, float] = {"handler_enter": self.started}; self.session_id: str | None = None
        self.batch_id: str | None = None; self.frame_count: int | None = None; self.jpeg_bytes = 0

    def mark(self, name: str) -> None:
        if self.enabled: self.marks[name] = time.perf_counter()

    def set_identity(self, session_id: str, batch_id: str, frame_count: int) -> None:
        self.session_id, self.batch_id, self.frame_count = session_id, batch_id, frame_count

    def emit(self, status: int) -> None:
        if not self.enabled: return
        self.mark("response_ready")
        def elapsed(end: str, start: str) -> float | None:
            if end not in self.marks or start not in self.marks: return None
            return round((self.marks[end] - self.marks[start]) * 1000, 3)
        first, last = self.scope.get("_batch_timing_body_first"), self.scope.get("_batch_timing_body_last")
        record = {
            "event": "streaming_batch_timing", "httpStatus": status,
            "sessionId": self.session_id, "batchId": self.batch_id,
            "frameCount": self.frame_count, "jpegTotalBytes": self.jpeg_bytes,
            "handlerToFormMs": elapsed("form_done", "handler_enter"),
            "asgiBodyFirstAfterHandlerMs": None if first is None else round((first-self.started)*1000, 3),
            "asgiBodyReceiveMs": None if first is None or last is None else round((last-first)*1000, 3),
            "formAfterBodyMs": None if last is None or "form_done" not in self.marks else round((self.marks["form_done"]-last)*1000, 3),
            "jpegReadMs": elapsed("jpeg_read_done", "form_done"),
            "sha256Ms": elapsed("sha256_done", "jpeg_read_done"),
            "sessionLookupMs": elapsed("session_lookup_done", "sha256_done"),
            "sessionLockWaitMs": elapsed("session_lock_acquired", "batch_push_start"),
            "enqueueMs": elapsed("enqueue_done", "session_lock_acquired"),
            "responsePreparationMs": elapsed("response_ready", "enqueue_done"),
            "handlerTotalMs": elapsed("response_ready", "handler_enter"),
        }
        BATCH_TIMING_LOGGER.info("%s", json.dumps(record, separators=(",", ":"), ensure_ascii=False))


app.add_middleware(BatchTimingASGIMiddleware)


class StartRequest(BaseModel):
    sessionId: str = Field(min_length=1, max_length=128)
    triggerId: str | None = Field(default=None, max_length=128)
    # Legacy PoC alias only. New Pi/Spring Boot callers must use triggerId.
    eventId: str | None = Field(default=None, max_length=128)
    securityEventId: str | None = Field(default=None, max_length=128)
    triggerType: str = Field(default="KEYPAD", max_length=32)
    sourceFps: float = Field(gt=0, le=120)
    callbackUrl: str | None = None
    finalResultCallbackEnabled: bool = False


class CompleteRequest(BaseModel):
    sessionId: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def validate_callback(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in ALLOWED_CALLBACK_HOSTS:
        raise HTTPException(400, "callbackUrl host is not in STREAMING_POC_CALLBACK_HOSTS allowlist")
    return url


class Session:
    def __init__(self, request: StartRequest):
        self.id, self.trigger_id, self.trigger_type, self.fps = request.sessionId, request.triggerId or request.eventId, request.triggerType, request.sourceFps
        self.security_event_id = request.securityEventId
        self.final_callback_enabled = request.finalResultCallbackEnabled
        self.callback_url = validate_callback(request.callbackUrl)
        self.q: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=QUEUE_CAPACITY)
        self.lock = threading.RLock(); self.closed = False; self.expected_frame = 0; self.events: list[dict[str, Any]] = []
        self.created_monotonic = time.monotonic(); self.completed_monotonic: float | None = None
        self.callback_records: list[dict[str, Any]] = []; self.worker_done = threading.Event(); self.final_dispatched = False; self.final_payload: dict[str, Any] | None = None; self.error: str | None = None; self.max_queue_depth = 0
        self.dispatcher = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"callback-{self.id}")
        # Kept for the life of a session so a client can safely retry a batch
        # after losing its HTTP response.  Values contain no JPEG bytes.
        self.accepted_batches: dict[str, tuple[tuple[int, int, str | None, str], ...]] = {}
        self.worker = threading.Thread(target=self._run, daemon=True, name=f"mediapipe-{self.id}"); self.worker.start()

    def push(self, frame_index: int, source_frame_index: int, captured_at: str | None, jpeg: bytes) -> str:
        with self.lock:
            if self.closed: raise HTTPException(409, "session input is already complete")
            if frame_index < self.expected_frame: return "duplicate"
            if frame_index > self.expected_frame: raise HTTPException(409, f"out-of-order frame: expected {self.expected_frame}")
            try: self.q.put_nowait({"frame": frame_index, "sourceFrame": source_frame_index, "capturedAt": captured_at, "jpeg": jpeg, "receivedAt": time.perf_counter()})
            except queue.Full: raise HTTPException(429, "session queue is full; retry this frame")
            self.max_queue_depth = max(self.max_queue_depth, self.q.qsize()); self.expected_frame += 1
            return "accepted"

    def complete(self) -> None:
        with self.lock:
            if not self.closed:
                self.closed = True
                threading.Thread(target=self._enqueue_end, args=(self.q,), daemon=True, name=f"complete-{self.id}").start()

    @staticmethod
    def _enqueue_end(target: queue.Queue[dict[str, Any] | object]) -> None:
        # Blocking here preserves every accepted frame, but it is deliberately
        # outside the HTTP handler so /vision/complete can return 202 promptly.
        target.put(END)

    def release(self) -> None:
        """Release callback resources only after a terminal session TTL expires."""
        self.dispatcher.shutdown(wait=False, cancel_futures=False)

    def _callback(self, payload: dict[str, Any]) -> None:
        if not self.callback_url: return
        # Spring Boot requires the callback key. Keep localhost/legacy Bearer
        # callbacks usable, but do not send an unauthenticated request to the
        # production Spring Boot host when its required key is absent.
        callback_host = urllib.parse.urlparse(self.callback_url).hostname
        if callback_host == "api.omniguardy.cloud" and not CALLBACK_KEY:
            with self.lock:
                self.callback_records.append({"messageType": payload["messageType"], "eventInstanceId": payload.get("eventInstanceId"), "attempt": 0, "status": None, "error": "callback authentication is not configured: STREAMING_POC_CALLBACK_KEY is required for api.omniguardy.cloud", "at": now_iso(), "elapsedMs": 0.0})
            return
        for attempt in (1, 2):
            started = time.perf_counter(); status = None; error = None
            try:
                headers = {"Content-Type": "application/json"}
                if CALLBACK_BEARER_TOKEN: headers["Authorization"] = f"Bearer {CALLBACK_BEARER_TOKEN}"
                if CALLBACK_KEY: headers["X-Vision-Callback-Key"] = CALLBACK_KEY
                req = urllib.request.Request(self.callback_url, data=json.dumps(payload).encode(), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=2) as response: status = response.status
            except urllib.error.HTTPError as exc:
                status = exc.code
                error = (f"callback authentication rejected (HTTP {status}; credentials redacted)"
                         if status in {401, 403} else f"callback returned HTTP {status}")
            except Exception as exc:
                # Exception text can include request headers on malformed
                # credentials, so never retain it in a status/log record.
                error = f"callback transport error ({type(exc).__name__}; credentials redacted)"
            with self.lock: self.callback_records.append({"messageType": payload["messageType"], "eventInstanceId": payload.get("eventInstanceId"), "attempt": attempt, "status": status, "error": error, "at": now_iso(), "elapsedMs": round((time.perf_counter()-started)*1000,3)})
            if status and 200 <= status < 300: return
            time.sleep(.05)

    def _emit_early(self, frame: int, captured_at: str | None) -> None:
        sequence = len(self.events) + 1; iid = f"{self.id}-signal-{sequence:03d}"
        event = {"eventType": "SILENT_SIGNAL", "detectedFrame": frame, "confirmedFrame": frame, "capturedAt": captured_at, "eventInstanceId": iid}
        self.events.append(event)
        payload = {"messageType": "EARLY_EVENT", "sessionId": self.id, "triggerId": self.trigger_id, "securityEventId": self.security_event_id, "triggerType": self.trigger_type, "eventInstanceId": iid, "sequence": sequence, "provisional": True, "event": {k:event[k] for k in ("eventType","detectedFrame","confirmedFrame","capturedAt")}, "emittedAt": now_iso()}
        self.dispatcher.submit(self._callback, payload)

    def _run(self) -> None:
        detector = _build_detector(); tracker = RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS, min_fist_frames=MIN_FIST_FRAMES, max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES, max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES)
        try:
            while True:
                item = self.q.get()
                if item is END: self.q.task_done(); break
                frame = int(item["sourceFrame"]); image = cv2.imdecode(np.frombuffer(item["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None: self.q.task_done(); continue
                rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(frame * 1000 / self.fps))
                gestures = [classify_gesture_front(lm, hand[0].category_name) for lm, hand in zip(result.hand_landmarks, result.handedness)]
                if tracker.update(frame_gesture_from_hands(gestures)):
                    with self.lock: self._emit_early(frame, item["capturedAt"])
                self.q.task_done()
        except Exception as exc:
            with self.lock: self.error = repr(exc)
        finally:
            detector.close()
            with self.lock: self.completed_monotonic = time.monotonic()
            self.worker_done.set(); self._dispatch_final()

    def _dispatch_final(self) -> None:
        with self.lock:
            if self.final_dispatched: return
            self.final_dispatched = True
            if not self.final_callback_enabled: return
            final = self.result_payload()
        self.dispatcher.submit(self._callback, final)

    def result_payload(self) -> dict[str, Any]:
        # Compatibility shape is retained where possible, but this deliberately
        # contains only the online Silent Signal PoC event family.
        if self.final_payload is not None: return self.final_payload
        status = "failed" if self.error else "success"
        payload={"messageType": "FINAL_RESULT", "sessionId": self.id, "trigger": {"triggerId": self.trigger_id, "triggerType": self.trigger_type, "securityEventId": self.security_event_id}, "relatedEarlyEventInstanceIds": [x["eventInstanceId"] for x in self.events], "status": status, "result": {"visionEvents": [{k:v for k,v in e.items() if k != "eventInstanceId"} for e in self.events], "trackedPersonCount": None, "streamingPocScope": "SILENT_SIGNAL only"}, "error": self.error, "emittedAt": now_iso()}
        if self.worker_done.is_set(): self.final_payload=payload
        return payload

    def status(self) -> dict[str, Any]:
        with self.lock:
            analysis_state = "FAILED" if self.error else ("COMPLETED" if self.worker_done.is_set() else ("PROCESSING" if self.closed else "RECEIVING"))
            ttl_remaining = None if self.completed_monotonic is None else max(0.0, COMPLETED_SESSION_TTL_SECONDS - (time.monotonic() - self.completed_monotonic))
            return {"sessionId": self.id, "triggerId": self.trigger_id, "securityEventId": self.security_event_id, "triggerType": self.trigger_type, "analysisState": analysis_state, "inputComplete": self.closed, "nextExpectedFrameIndex": self.expected_frame, "queueDepth": self.q.qsize(), "mediaPipeQueueMaxDepth": self.max_queue_depth, "yoloQueueMaxDepth": None, "finalResultAvailable": self.worker_done.is_set(), "completedSessionTtlSecondsRemaining": None if ttl_remaining is None else round(ttl_remaining, 3), "earlyEventCount": len(self.events), "callbackAttempts": list(self.callback_records), "error": self.error}

    def next_expected_frame_index(self) -> int:
        with self.lock: return self.expected_frame


class PhaseBSession(Session):
    """Session-local MediaPipe, YOLO/BoT-SORT, and optional AUDIO workers."""
    def __init__(self, request: StartRequest):
        self.yolo_q: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=QUEUE_CAPACITY)
        self.audio_q: queue.Queue[dict[str, Any] | object] | None = queue.Queue(maxsize=QUEUE_CAPACITY) if request.triggerType == "AUDIO" else None
        self.done_workers: set[str] = set(); self.expected_workers = {"media", "yolo"} | ({"audio"} if self.audio_q else set())
        self.tracking_rows: list[dict[str, Any]] = []; self.source_indices: list[int] = []; self.frame_width: int | None = None; self.frame_height: int | None = None
        self.occlusion = OnlineOcclusionAdapter() if self.audio_q else None; self.final_events: list[dict[str, Any]] = []; self.tracked_person_count: int | None = None; self.analysis_completeness = "PROCESSING"
        self.yolo_max_queue = 0; self.audio_max_queue = 0
        self.worker_progress: dict[str, dict[str, Any]] = {
            "media": {"enabled": True, "state": "RUNNING", "processedFrames": 0, "decodedFrames": 0, "error": None},
            "yolo": {"enabled": True, "state": "RUNNING", "processedFrames": 0, "decodedFrames": 0, "error": None},
            "audio": {"enabled": self.audio_q is not None, "state": "RUNNING" if self.audio_q else "DISABLED", "processedFrames": 0, "decodedFrames": 0, "error": None},
        }
        super().__init__(request)
        self.yolo_worker = threading.Thread(target=self._run_yolo, daemon=True, name=f"yolo-{self.id}"); self.yolo_worker.start()
        self.audio_worker = None
        if self.audio_q:
            self.audio_worker = threading.Thread(target=self._run_audio, daemon=True, name=f"audio-{self.id}"); self.audio_worker.start()

    def push(self, frame_index: int, source_frame_index: int, captured_at: str | None, jpeg: bytes) -> str:
        with self.lock:
            if self.closed: raise HTTPException(409, "session input is already complete")
            if frame_index < self.expected_frame: return "duplicate"
            if frame_index > self.expected_frame: raise HTTPException(409, f"out-of-order frame: expected {self.expected_frame}")
            queues=[self.q,self.yolo_q]+([self.audio_q] if self.audio_q else [])
            if any(q.qsize() >= q.maxsize for q in queues): raise HTTPException(429, "session queue is full; retry this frame")
            item={"frame":frame_index,"sourceFrame":source_frame_index,"capturedAt":captured_at,"jpeg":jpeg,"receivedAt":time.perf_counter()}
            for q in queues:q.put_nowait(item)
            self.max_queue_depth=max(self.max_queue_depth,self.q.qsize()); self.yolo_max_queue=max(self.yolo_max_queue,self.yolo_q.qsize()); self.audio_max_queue=max(self.audio_max_queue,self.audio_q.qsize() if self.audio_q else 0)
            self.source_indices.append(source_frame_index); self.expected_frame+=1
            return "accepted"

    def push_batch(self, batch_id: str, frames: list[dict[str, Any]], timing: BatchTiming | None = None) -> str:
        """Atomically accept a contiguous batch, or leave session state unchanged.

        The session lock serializes all producers (single-frame and batch
        endpoints).  Workers only dequeue, so after capacity for every frame is
        reserved under this lock, queue.Full cannot occur during the following
        put_nowait calls: consumers can only make additional space available.
        """
        descriptor = tuple((int(x["frameIndex"]), int(x["sourceFrameIndex"]), x["capturedAt"], x["sha256"])
                           for x in frames)
        size = len(frames)
        first, last = descriptor[0][0], descriptor[-1][0]
        if timing: timing.mark("batch_push_start")
        with self.lock:
            if timing: timing.mark("session_lock_acquired")
            if self.closed: raise HTTPException(409, "session input is already complete")
            prior = self.accepted_batches.get(batch_id)
            if prior is not None:
                if prior == descriptor: return "duplicate"
                raise HTTPException(409, "batchId was already accepted with different frame metadata or payload")
            if first != self.expected_frame:
                raise HTTPException(409, f"out-of-order batch: expected {self.expected_frame}")
            # Defence in depth: request parsing validates this before the lock.
            if any(descriptor[i][0] != first + i for i in range(size)):
                raise HTTPException(409, "batch frameIndex values must be contiguous and ordered")
            queues = [self.q, self.yolo_q] + ([self.audio_q] if self.audio_q else [])
            if any(q.qsize() + size > q.maxsize for q in queues):
                raise HTTPException(429, "session queue cannot accept the complete batch; retry this batch")
            items = [{"frame": x["frameIndex"], "sourceFrame": x["sourceFrameIndex"],
                      "capturedAt": x["capturedAt"], "jpeg": x["jpeg"],
                      "receivedAt": time.perf_counter()} for x in frames]
            # No blocking or inference happens here.  The preflight above is a
            # capacity reservation because this is the sole producer under lock.
            for item in items:
                for target in queues: target.put_nowait(item)
            if timing: timing.mark("enqueue_done")
            self.max_queue_depth = max(self.max_queue_depth, self.q.qsize())
            self.yolo_max_queue = max(self.yolo_max_queue, self.yolo_q.qsize())
            self.audio_max_queue = max(self.audio_max_queue, self.audio_q.qsize() if self.audio_q else 0)
            self.source_indices.extend(x["sourceFrameIndex"] for x in frames)
            self.expected_frame = last + 1
            self.accepted_batches[batch_id] = descriptor
            return "accepted"

    def complete(self) -> None:
        with self.lock:
            if self.closed:return
            self.closed=True
            for name in self.expected_workers:
                self.worker_progress[name]["state"] = "DRAINING"
            targets=[self.q,self.yolo_q]+([self.audio_q] if self.audio_q else [])
            for target in targets:
                threading.Thread(target=self._enqueue_end,args=(target,),daemon=True,name=f"complete-{self.id}").start()

    def _record_worker_frame(self, name: str, decoded: bool) -> None:
        with self.lock:
            worker = self.worker_progress[name]
            worker["processedFrames"] += 1
            if decoded:
                worker["decodedFrames"] += 1

    def _record_worker_error(self, name: str, error: Exception) -> None:
        with self.lock:
            text = repr(error)
            self.error = text
            self.worker_progress[name]["error"] = text

    def _analysis_diagnostics(self) -> dict[str, Any]:
        """Expose lifecycle/occlusion evidence without retaining image bytes."""
        with self.lock:
            contiguous = source_is_contiguous(self.source_indices)
            if self.error:
                final_reason = "analysis_failed"
            elif not self.closed:
                final_reason = "input_not_complete"
            elif not self.worker_done.is_set():
                final_reason = "waiting_for_workers"
            elif not contiguous:
                final_reason = "partial_source_time_axis"
            else:
                final_reason = "final_result_generated"
            diagnostics: dict[str, Any] = {
                "inputComplete": self.closed,
                "analysisState": "FAILED" if self.error else ("COMPLETED" if self.worker_done.is_set() else ("PROCESSING" if self.closed else "RECEIVING")),
                "analysisCompleteness": self.analysis_completeness,
                "frames": {
                    "received": self.expected_frame,
                    "sourceFrameIndexContiguous": contiguous,
                    "mediaProcessed": self.worker_progress["media"]["processedFrames"],
                    "yoloProcessed": self.worker_progress["yolo"]["processedFrames"],
                    "audioProcessed": self.worker_progress["audio"]["processedFrames"],
                    "audioDecoded": self.worker_progress["audio"]["decodedFrames"],
                },
                "workers": {name: dict(progress) for name, progress in self.worker_progress.items()},
                "finalResult": {"available": self.worker_done.is_set(), "reason": final_reason},
            }
            if self.occlusion is not None:
                occlusion = self.occlusion.diagnostics(
                    self.fps, include_frame_metrics=OCCLUSION_FRAME_METRICS_ENABLED,
                )
                occlusion["eventEligibility"] = (
                    "eligible" if contiguous and not self.error
                    else "not_evaluated_partial_source_time_axis" if not contiguous
                    else "not_evaluated_worker_failed"
                )
                diagnostics["occlusion"] = occlusion
            return diagnostics

    def _run(self) -> None:
        detector=_build_detector(); tracker=RepeatedFistToOpenTracker(required_repeats=REQUIRED_REPEATS,min_fist_frames=MIN_FIST_FRAMES,max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES)
        try:
            while True:
                item=self.q.get()
                if item is END:self.q.task_done();break
                frame=int(item["sourceFrame"]); image=cv2.imdecode(np.frombuffer(item["jpeg"],dtype=np.uint8),cv2.IMREAD_COLOR)
                self._record_worker_frame("media", image is not None)
                if image is not None:
                    rgb=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB); result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),int(frame*1000/self.fps)); gestures=[classify_gesture_front(lm,h[0].category_name) for lm,h in zip(result.hand_landmarks,result.handedness)]
                    if tracker.update(frame_gesture_from_hands(gestures)):
                        with self.lock:self._emit_early(frame,item["capturedAt"])
                self.q.task_done()
        except Exception as exc:
            self._record_worker_error("media", exc)
        finally:
            detector.close(); self._mark_done("media")

    def _run_yolo(self) -> None:
        model=None
        try:
            model=YOLO(str(ROOT/"tracking/yolov8n.pt")); device=0 if torch.cuda.is_available() else "cpu"; imgsz=max(160,int(os.getenv("VISION_YOLO_IMGSZ","640")))
            while True:
                item=self.yolo_q.get()
                if item is END:self.yolo_q.task_done();break
                image=cv2.imdecode(np.frombuffer(item["jpeg"],dtype=np.uint8),cv2.IMREAD_COLOR)
                self._record_worker_frame("yolo", image is not None)
                if image is not None:
                    if self.frame_width is None:self.frame_height,self.frame_width=image.shape[:2]
                    result=model.track(image,tracker="botsort.yaml",persist=True,save=False,conf=.3,classes=[0],stream=False,device=device,verbose=False,imgsz=imgsz)[0]
                    if result.boxes is not None:
                        for box in result.boxes:
                            if box.id is None:continue
                            x1,y1,x2,y2=box.xyxy[0].cpu().numpy(); self.tracking_rows.append({"frame":int(item["sourceFrame"]),"track_id":int(box.id[0]),"x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),"confidence":float(box.conf[0])})
                self.yolo_q.task_done()
        except Exception as exc:
            self._record_worker_error("yolo", exc)
        finally:self._mark_done("yolo")

    def _run_audio(self) -> None:
        try:
            assert self.audio_q is not None and self.occlusion is not None
            while True:
                item=self.audio_q.get()
                if item is END:self.audio_q.task_done();break
                image=cv2.imdecode(np.frombuffer(item["jpeg"],dtype=np.uint8),cv2.IMREAD_COLOR)
                self._record_worker_frame("audio", image is not None)
                if image is not None:self.occlusion.observe(int(item["sourceFrame"]),image)
                self.audio_q.task_done()
        except Exception as exc:
            self._record_worker_error("audio", exc)
        finally:self._mark_done("audio")

    def _mark_done(self,name:str)->None:
        with self.lock:
            self.worker_progress[name]["state"] = "FAILED" if self.worker_progress[name]["error"] else "COMPLETED"
            self.done_workers.add(name)
            if self.done_workers != self.expected_workers:return
            contiguous=source_is_contiguous(self.source_indices)
            if self.error:self.analysis_completeness="FAILED"
            elif not contiguous:self.analysis_completeness="PARTIAL_SAMPLED_TIME_AXIS"
            elif self.trigger_type=="KEYPAD":
                extras,count=final_keypad_events(self.tracking_rows,self.fps,self.frame_width,self.frame_height,True); self.final_events=[*self.events,*extras]; self.tracked_person_count=count; self.analysis_completeness="COMPLETE_KEYPAD"
            elif self.trigger_type=="AUDIO":
                assert self.occlusion is not None
                extras,count=final_audio_events(self.tracking_rows,self.occlusion,self.fps,self.frame_width,self.frame_height,True); self.final_events=[*extras]; self.tracked_person_count=count; self.analysis_completeness="COMPLETE_AUDIO"
            else:self.analysis_completeness="FAILED"; self.error=f"unsupported triggerType={self.trigger_type}"
            if not self.final_events and self.trigger_type=="KEYPAD":self.final_events=list(self.events)
            self.completed_monotonic=time.monotonic(); self.worker_done.set(); self._dispatch_final()

    def result_payload(self)->dict[str,Any]:
        if self.final_payload is not None:return self.final_payload
        status="failed" if self.error else ("partial" if self.analysis_completeness.startswith("PARTIAL") else "success")
        payload={"messageType":"FINAL_RESULT","sessionId":self.id,"trigger":{"triggerId":self.trigger_id,"triggerType":self.trigger_type,"securityEventId":self.security_event_id},"analysisState":"FAILED" if self.error else "COMPLETED","analysisCompleteness":self.analysis_completeness,"relatedEarlyEventInstanceIds":[x["eventInstanceId"] for x in self.events],"status":status,"result":{"visionEvents":[{k:v for k,v in e.items() if k!="eventInstanceId"} for e in self.final_events],"trackedPersonCount":self.tracked_person_count,"streamingPocScope":"KEYPAD Silent+Rear or AUDIO Occlusion+Dwell when COMPLETE","diagnostics":self._analysis_diagnostics()},"error":self.error,"emittedAt":now_iso()}
        if self.worker_done.is_set():self.final_payload=payload
        return payload

    def status(self)->dict[str,Any]:
        base=super().status(); base.update({"analysisState":"FAILED" if self.error else ("COMPLETED" if self.worker_done.is_set() else ("PROCESSING" if self.closed else "RECEIVING")),"analysisCompleteness":self.analysis_completeness,"mediaPipeQueueMaxDepth":self.max_queue_depth,"yoloQueueMaxDepth":self.yolo_max_queue,"audioQueueMaxDepth":self.audio_max_queue,"trackingRecordCount":len(self.tracking_rows),"diagnostics":self._analysis_diagnostics()}); return base


sessions: dict[str, Session] = {}; sessions_lock = threading.RLock()

def cleanup_expired_sessions() -> int:
    """Remove only terminal sessions after their result-retention TTL.

    Active RECEIVING/PROCESSING sessions are never removed by this PoC helper;
    production needs a separately agreed input-idle timeout policy.
    """
    now=time.monotonic()
    with sessions_lock:
        expired=[sid for sid,session in sessions.items() if session.worker_done.is_set() and session.completed_monotonic is not None and now-session.completed_monotonic>=COMPLETED_SESSION_TTL_SECONDS]
        removed=[sessions.pop(sid) for sid in expired]
    for session in removed: session.release()
    return len(removed)


async def parse_batch_form(request: Request, timing: BatchTiming | None = None) -> tuple[str, str, list[dict[str, Any]]]:
    """Validate a batch completely before it can mutate a session queue."""
    form = await request.form()
    if timing: timing.mark("form_done")
    session_id, manifest_text = form.get("sessionId"), form.get("manifest")
    if not isinstance(session_id, str) or not session_id:
        raise HTTPException(422, "sessionId is required")
    if not isinstance(manifest_text, str):
        raise HTTPException(422, "manifest is required and must be JSON text")
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError:
        raise HTTPException(422, "manifest is not valid JSON")
    if not isinstance(manifest, dict):
        raise HTTPException(422, "manifest must be a JSON object")
    batch_id, raw_frames = manifest.get("batchId"), manifest.get("frames")
    if not isinstance(batch_id, str):
        raise HTTPException(422, "manifest.batchId is required")
    try: uuid.UUID(batch_id)
    except (ValueError, AttributeError, TypeError): raise HTTPException(422, "manifest.batchId must be a UUID")
    if not isinstance(raw_frames, list) or not 1 <= len(raw_frames) <= MAX_BATCH_FRAMES:
        raise HTTPException(422, f"manifest.frames must contain 1 to {MAX_BATCH_FRAMES} frames")
    if timing: timing.set_identity(session_id, batch_id, len(raw_frames))

    prepared: list[dict[str, Any]] = []; expected_parts: set[str] = set(); prior_index: int | None = None
    for position, raw in enumerate(raw_frames):
        if not isinstance(raw, dict): raise HTTPException(422, f"manifest.frames[{position}] must be an object")
        part, index, declared_hash = raw.get("part"), raw.get("frameIndex"), raw.get("sha256")
        if not isinstance(part, str) or not BATCH_PART_RE.fullmatch(part) or part in expected_parts:
            raise HTTPException(422, f"manifest.frames[{position}].part must be a unique safe part name")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise HTTPException(422, f"manifest.frames[{position}].frameIndex must be a non-negative integer")
        if prior_index is not None and index != prior_index + 1:
            raise HTTPException(409, "batch frameIndex values must be contiguous and ordered")
        source = raw.get("sourceFrameIndex", index)
        captured_at = raw.get("capturedAt")
        if isinstance(source, bool) or not isinstance(source, int) or source < 0:
            raise HTTPException(422, f"manifest.frames[{position}].sourceFrameIndex must be a non-negative integer")
        if captured_at is not None and not isinstance(captured_at, str):
            raise HTTPException(422, f"manifest.frames[{position}].capturedAt must be a string or null")
        if not isinstance(declared_hash, str) or not SHA256_RE.fullmatch(declared_hash):
            raise HTTPException(422, f"manifest.frames[{position}].sha256 must be lowercase SHA-256 hex")
        uploads = form.getlist(part)
        if len(uploads) != 1 or not hasattr(uploads[0], "read"):
            raise HTTPException(422, f"exactly one JPEG file part named {part} is required")
        upload = uploads[0]
        if upload.content_type not in {"image/jpeg", "image/jpg"}:
            raise HTTPException(415, f"{part} must have content type image/jpeg")
        jpeg = await upload.read()
        if not jpeg:
            raise HTTPException(422, f"{part} must not be empty")
        expected_parts.add(part); prior_index = index
        prepared.append({"frameIndex": index, "sourceFrameIndex": source, "capturedAt": captured_at,
                         "sha256": declared_hash, "jpeg": jpeg})
    if timing:
        timing.jpeg_bytes = sum(len(x["jpeg"]) for x in prepared)
        timing.mark("jpeg_read_done")
    for raw in prepared:
        actual_hash = hashlib.sha256(raw["jpeg"]).hexdigest()
        if actual_hash != raw["sha256"]:
            raise HTTPException(422, f"{raw['frameIndex']} JPEG SHA-256 does not match manifest")
        raw["sha256"] = actual_hash
    if timing: timing.mark("sha256_done")
    # Reject an accidental extra file rather than silently accepting an
    # ambiguous multipart body.
    for key, value in form.multi_items():
        if hasattr(value, "filename") and key not in expected_parts:
            raise HTTPException(422, f"unexpected file part: {key}")
    return session_id, batch_id, prepared

@app.post("/vision/start", status_code=201)
def start(request: StartRequest):
    cleanup_expired_sessions()
    if not (request.triggerId or request.eventId): raise HTTPException(422, "triggerId is required (eventId is accepted only as a legacy alias)")
    if request.triggerType == "AUDIO" and not request.securityEventId: raise HTTPException(422, "securityEventId is required for AUDIO")
    with sessions_lock:
        if request.sessionId in sessions: raise HTTPException(409, "sessionId already exists")
        sessions[request.sessionId] = PhaseBSession(request)
    return {"sessionId": request.sessionId, "triggerId": request.triggerId or request.eventId, "status": "started", "nextExpectedFrameIndex": 0, "scope": "online Silent Signal EARLY_EVENT PoC"}

@app.post("/vision/frame", status_code=202)
async def frame(sessionId: str = Form(...), frameIndex: int = Form(...), sourceFrameIndex: int | None = Form(None), capturedAt: str | None = Form(None), frame: UploadFile = File(...)):
    cleanup_expired_sessions()
    if frame.content_type not in {"image/jpeg", "image/jpg"}: raise HTTPException(415, "frame must be image/jpeg")
    jpeg = await frame.read()
    with sessions_lock: session = sessions.get(sessionId)
    if not session: raise HTTPException(404, "unknown sessionId")
    source_index = frameIndex if sourceFrameIndex is None else sourceFrameIndex
    state = session.push(frameIndex, source_index, capturedAt, jpeg)
    return {"sessionId": sessionId, "frameIndex": frameIndex, "sourceFrameIndex": source_index, "status": state}

@app.post("/vision/frames", status_code=202)
async def frames(request: Request):
    timing = BatchTiming(request)
    try:
        cleanup_expired_sessions()
        session_id, batch_id, prepared = await parse_batch_form(request, timing)
        timing.set_identity(session_id, batch_id, len(prepared))
        with sessions_lock: session = sessions.get(session_id)
        timing.mark("session_lookup_done")
        if not session: raise HTTPException(404, "unknown sessionId")
        # All validation is complete before this call.  push_batch does the
        # session state/capacity preflight and mutates nothing on 409 or 429.
        state = session.push_batch(batch_id, prepared, timing)
        first, last = prepared[0]["frameIndex"], prepared[-1]["frameIndex"]
        next_index = session.next_expected_frame_index()
        timing.emit(202)
        return {"sessionId": session_id, "batchId": batch_id, "status": state,
                "firstFrameIndex": first, "lastFrameIndex": last,
                "acceptedFrameCount": len(prepared) if state == "accepted" else 0,
                "duplicateFrameCount": len(prepared) if state == "duplicate" else 0,
                "nextExpectedFrameIndex": next_index}
    except HTTPException as exc:
        timing.emit(exc.status_code)
        raise
    except Exception:
        timing.emit(500)
        raise

@app.post("/vision/complete", status_code=202)
def complete(request: CompleteRequest):
    cleanup_expired_sessions()
    with sessions_lock: session = sessions.get(request.sessionId)
    if not session: raise HTTPException(404, "unknown sessionId")
    session.complete()
    return {"sessionId": request.sessionId, "status": "input_complete", "resultUrl": f"/vision/result/{request.sessionId}"}

@app.get("/vision/status/{session_id}")
def status(session_id: str):
    cleanup_expired_sessions()
    with sessions_lock: session = sessions.get(session_id)
    if not session: raise HTTPException(404, "unknown sessionId")
    return session.status()

@app.get("/vision/result/{session_id}")
def result(session_id: str):
    cleanup_expired_sessions()
    with sessions_lock: session = sessions.get(session_id)
    if not session: raise HTTPException(404, "unknown sessionId")
    if not session.worker_done.is_set(): raise HTTPException(202, "analysis is still draining")
    return session.result_payload()
