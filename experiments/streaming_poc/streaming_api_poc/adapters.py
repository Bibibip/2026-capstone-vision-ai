"""Read-only adapters from JPEG streaming frames to existing final rules."""
from __future__ import annotations
from collections import defaultdict
from typing import Any
import cv2, numpy as np, pandas as pd

from exhibition_audio_service import (
    EDGE_DENSITY_MAX, LAPLACIAN_VARIANCE_MAX, OCCLUSION_MIN_SUSTAINED_SECONDS,
    ANALYSIS_SECONDS, detect_person_dwell,
)
from tracking.keypad_rules_front import detect_front_rear_approach

TRACKING_COLUMNS=["frame","track_id","x1","y1","x2","y2","confidence"]

class OnlineOcclusionAdapter:
    """Same metric/threshold calculation as detect_camera_occlusion, on frames."""

    def __init__(self) -> None:
        # (source frame, both thresholds pass, edge density, Laplacian variance)
        # Kept in memory only for the life of one streaming session. JPEG data is
        # never retained here.
        self.rows: list[tuple[int, bool, float, float]] = []

    def observe(self, source_frame: int, image: np.ndarray) -> None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge = float(cv2.countNonZero(cv2.Canny(gray, 100, 200))) / float(gray.size)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.rows.append((source_frame, edge < EDGE_DENSITY_MAX and lap < LAPLACIAN_VARIANCE_MAX, edge, lap))

    @staticmethod
    def _required_frames(fps: float) -> int:
        return max(1, round(OCCLUSION_MIN_SUSTAINED_SECONDS * fps))

    def _candidate_runs(self) -> list[tuple[int, int]]:
        """Return row-index ranges for all simultaneous low-structure runs."""
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for position, (_, candidate, _, _) in enumerate([*self.rows, (None, False, 0.0, 0.0)]):
            if candidate and start is None:
                start = position
            elif not candidate and start is not None:
                runs.append((start, position - 1))
                start = None
        return runs

    def diagnostics(self, fps: float, *, include_frame_metrics: bool = False) -> dict[str, Any]:
        """Return safe metric summaries without changing the detection rule."""
        required = self._required_frames(fps)
        runs = self._candidate_runs()
        longest = max((end - start + 1 for start, end in runs), default=0)
        edge_values = [row[2] for row in self.rows]
        lap_values = [row[3] for row in self.rows]
        edge_pass = sum(edge < EDGE_DENSITY_MAX for edge in edge_values)
        lap_pass = sum(lap < LAPLACIAN_VARIANCE_MAX for lap in lap_values)
        both_pass = sum(candidate for _, candidate, _, _ in self.rows)
        intervals = [
            {
                "startSourceFrameIndex": self.rows[start][0],
                "endSourceFrameIndex": self.rows[end][0],
                "frameCount": end - start + 1,
                "durationSeconds": round((end - start + 1) / fps, 3) if fps > 0 else None,
                "meetsSustainedThreshold": end - start + 1 >= required,
            }
            for start, end in runs
        ]
        if not self.rows:
            reason = "no_decodable_audio_frames"
        elif longest >= required:
            reason = "sustained_occlusion_candidate_detected"
        elif both_pass == 0:
            reason = "no_frame_met_both_structure_thresholds"
        else:
            reason = "qualified_runs_shorter_than_sustained_threshold"
        result: dict[str, Any] = {
            "framesAnalyzed": len(self.rows),
            "edgeDensityThreshold": EDGE_DENSITY_MAX,
            "laplacianVarianceThreshold": LAPLACIAN_VARIANCE_MAX,
            "minimumSustainedSeconds": OCCLUSION_MIN_SUSTAINED_SECONDS,
            "minimumSustainedFrames": required,
            "edgeDensityPassFrameCount": edge_pass,
            "laplacianVariancePassFrameCount": lap_pass,
            "bothThresholdsPassFrameCount": both_pass,
            "longestBothThresholdsRunFrames": longest,
            "candidateIntervals": intervals,
            "eventDetectionReason": reason,
            "frameMetricsIncluded": include_frame_metrics,
        }
        if edge_values:
            result["edgeDensityRange"] = {"min": min(edge_values), "max": max(edge_values)}
            result["laplacianVarianceRange"] = {"min": min(lap_values), "max": max(lap_values)}
        if include_frame_metrics:
            result["frameMetrics"] = [
                {
                    "sourceFrameIndex": source_frame,
                    "edgeDensity": edge,
                    "laplacianVariance": lap,
                    "edgeDensityPass": edge < EDGE_DENSITY_MAX,
                    "laplacianVariancePass": lap < LAPLACIAN_VARIANCE_MAX,
                    "bothThresholdsPass": candidate,
                }
                for source_frame, candidate, edge, lap in self.rows
            ]
        return result

    def final_events(self, fps: float) -> list[dict[str, Any]]:
        required = self._required_frames(fps)
        events: list[dict[str, Any]] = []
        for start, end in self._candidate_runs():
            if end - start + 1 < required:
                continue
            first = self.rows[start][0]
            last = self.rows[end][0]
            events.append({"eventType": "CAMERA_OCCLUSION_SUSPECTED", "confidence": None,
                           "detectedFrame": self.rows[start + required - 1][0],
                           "details": {"startSeconds": round(first / fps, 3),
                                       "endSeconds": round(last / fps, 3),
                                       "durationSeconds": round((end - start + 1) / fps, 3),
                                       "edgeDensityThreshold": EDGE_DENSITY_MAX,
                                       "laplacianThreshold": LAPLACIAN_VARIANCE_MAX,
                                       "minSustainedSeconds": OCCLUSION_MIN_SUSTAINED_SECONDS,
                                       "windowSeconds": round(len(self.rows) / fps, 3),
                                       "experimental": True}})
        return events

def tracking_dataframe(rows:list[dict[str,Any]],width:int|None,height:int|None)->pd.DataFrame:
    df=pd.DataFrame(rows,columns=TRACKING_COLUMNS)
    if width:df.attrs['frame_width']=width
    if height:df.attrs['frame_height']=height
    return df

def source_is_contiguous(indices:list[int])->bool:
    return bool(indices) and all(b==a+1 for a,b in zip(indices,indices[1:]))

def final_keypad_events(rows:list[dict[str,Any]],fps:float,width:int|None,height:int|None,contiguous:bool)->tuple[list[dict[str,Any]],int|None]:
    df=tracking_dataframe(rows,width,height);count=0 if df.empty else int(df.track_id.nunique())
    if not contiguous:return [],count
    event=detect_front_rear_approach(df,fps)
    return ([] if event is None else [event]),count

def final_audio_events(rows:list[dict[str,Any]],occlusion:OnlineOcclusionAdapter,fps:float,width:int|None,height:int|None,contiguous:bool)->tuple[list[dict[str,Any]],int|None]:
    df=tracking_dataframe(rows,width,height);count=0 if df.empty else int(df.track_id.nunique())
    if not contiguous:return [],count
    window=min(ANALYSIS_SECONDS,max(0.,(max([r['frame'] for r in rows],default=-1)+1)/fps))
    dwell=detect_person_dwell(df,fps,window_seconds=window)
    return [*occlusion.final_events(fps),*dwell],count
