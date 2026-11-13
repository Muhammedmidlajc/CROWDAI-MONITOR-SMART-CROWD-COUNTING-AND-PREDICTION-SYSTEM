import os
import tempfile
import base64
import uuid
import cv2

from django.shortcuts import render, get_object_or_404
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile
from django.http import FileResponse, JsonResponse, HttpResponseBadRequest
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from ai.hybrid_predict import hybrid_predict
from ai.pdf_report import generate_pdf, generate_stream_pdf
from ai.temporal_forecasting import forecast_crowd_series, get_occupancy_status
from .models import MonitoringSession, CrowdTimeSeriesEntry


def get_density_level(count):
    if count < 50:
        return "Low"
    elif count < 200:
        return "Medium"
    elif count < 500:
        return "High"
    else:
        return "Very High"


def home(request):

    context = {}

    if request.method == "POST":

        print("\n==============================")
        print("FORM SUBMITTED")
        print("==============================")

        print("POST DATA :", request.POST)
        print("FILES :", request.FILES)

        # --------------------------------
        # Get uploaded image
        # --------------------------------

        image = request.FILES.get("image")

        if not image:
            print("ERROR: No image received")
            return render(request, "detector/home.html", context)

        # --------------------------------
        # Get maximum capacity
        # --------------------------------

        capacity_value = request.POST.get("max_capacity")

        print("RAW CAPACITY :", capacity_value)

        try:
            max_capacity = int(capacity_value)

        except (TypeError, ValueError):
            max_capacity = 0

        print("MAX CAPACITY :", max_capacity)

        # --------------------------------
        # Save image
        # --------------------------------

        fs = FileSystemStorage()

        filename = fs.save(image.name, image)

        image_url = fs.url(filename)

        image_path = fs.path(filename)

        # --------------------------------
        # Hybrid AI Prediction
        # --------------------------------

        (
            people_count,
            heatmap_path,
            model_used,
            yolo_count,
            mccnn_count
        ) = hybrid_predict(image_path)

        heatmap_url = fs.url(
            os.path.basename(heatmap_path)
        )

        # --------------------------------
        # Density Level
        # --------------------------------

        density_level = get_density_level(
            people_count
        )

        # --------------------------------
        # Occupancy Calculation
        # --------------------------------

        if max_capacity > 0:

            occupancy_percentage = (
                people_count / max_capacity
            ) * 100

        else:

            occupancy_percentage = 0

        # --------------------------------
        # Occupancy Status
        # --------------------------------

        if occupancy_percentage < 50:

            occupancy_status = "Low Occupancy"

        elif occupancy_percentage < 75:

            occupancy_status = "Moderate Occupancy"

        elif occupancy_percentage < 90:

            occupancy_status = "High Occupancy"

        elif occupancy_percentage <= 100:

            occupancy_status = "Near Capacity"

        else:

            occupancy_status = "Over Capacity"

        # --------------------------------
        # Debug
        # --------------------------------

        print("==============================")
        print("OCCUPANCY RESULT")
        print("==============================")
        print("People Count :", people_count)
        print("Maximum Capacity :", max_capacity)
        print("Occupancy % :", occupancy_percentage)
        print("Occupancy Status :", occupancy_status)
        print("==============================")

        # --------------------------------
        # Save Session
        # --------------------------------

        request.session["image_path"] = image_path
        request.session["heatmap_path"] = heatmap_path

        request.session["people_count"] = people_count
        request.session["density_level"] = density_level

        request.session["yolo_count"] = yolo_count
        request.session["mccnn_count"] = mccnn_count
        request.session["model_used"] = model_used

        request.session["max_capacity"] = max_capacity
        request.session["occupancy_percentage"] = occupancy_percentage
        request.session["occupancy_status"] = occupancy_status

        # --------------------------------
        # Send to HTML
        # --------------------------------

        context = {

            "image_url": image_url,

            "heatmap_url": heatmap_url,

            "people_count": people_count,

            "density_level": density_level,

            "model_used": model_used,

            "yolo_count": yolo_count,

            "mccnn_count": mccnn_count,

            "max_capacity": max_capacity,

            "occupancy_percentage":
                occupancy_percentage,

            "occupancy_status":
                occupancy_status,
        }

    return render(
        request,
        "detector/home.html",
        context
    )

def download_report(request):
    image_path = request.session.get("image_path")
    heatmap_path = request.session.get("heatmap_path")

    people_count = request.session.get("people_count")
    density_level = request.session.get("density_level")

    yolo_count = request.session.get("yolo_count")
    mccnn_count = request.session.get("mccnn_count")
    model_used = request.session.get("model_used")
    max_capacity = request.session.get("max_capacity")
    occupancy_percentage = request.session.get("occupancy_percentage")
    occupancy_status = request.session.get("occupancy_status")
    if not image_path:
        return render(request, "detector/home.html")

    pdf_path = os.path.join(
        tempfile.gettempdir(),
        "Crowd_Density_Report.pdf"
    )

    generate_pdf(
        pdf_path,
        image_path,
        heatmap_path,

        people_count,

        yolo_count,

        mccnn_count,

        model_used,

        density_level,

        max_capacity,

        occupancy_percentage,

        occupancy_status,
    )

    return FileResponse(
        open(pdf_path, "rb"),
        as_attachment=True,
        filename="Crowd_Density_Report.pdf"
    )


# ==============================================================================
# LIVE VIDEO STREAM & TEMPORAL FORECASTING VIEWS
# ==============================================================================

def live_monitoring(request):
    """
    Renders the live video stream monitoring dashboard with real-time
    spatial crowd estimation (Stage 1) and temporal time-series forecasting (Stage 2).
    """
    # Fetch recent active sessions for quick inspection if any
    recent_sessions = MonitoringSession.objects.all()[:5]

    context = {
        "recent_sessions": recent_sessions,
        "default_capacity": 100,
        "default_interval": 3,
    }
    return render(request, "detector/live_monitoring.html", context)


@csrf_exempt
def api_create_session(request):
    """
    Initializes a new live video monitoring session in the database.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"}, status=405)

    title = request.POST.get("title", "Live Crowd Monitoring")
    source_type = request.POST.get("source_type", "webcam")
    source_name = request.POST.get("source_name", "Webcam Feed")

    try:
        max_capacity = int(request.POST.get("max_capacity", 100))
    except (ValueError, TypeError):
        max_capacity = 100

    try:
        interval_seconds = float(request.POST.get("interval_seconds", 3.0))
    except (ValueError, TypeError):
        interval_seconds = 3.0

    session_id = f"stream_{uuid.uuid4().hex[:10]}"

    session = MonitoringSession.objects.create(
        session_id=session_id,
        title=title,
        source_type=source_type,
        source_name=source_name,
        max_capacity=max_capacity,
        interval_seconds=interval_seconds,
        is_active=True,
    )

    return JsonResponse({
        "status": "success",
        "session_id": session.session_id,
        "max_capacity": session.max_capacity,
        "interval_seconds": session.interval_seconds,
        "created_at": session.created_at.strftime("%Y-%m-%d %H:%M:%S"),
    })


@csrf_exempt
def api_process_frame(request):
    """
    Stage 1 & Stage 2 Live Pipeline Endpoint:
    Called at fixed intervals (every N seconds) with the latest captured video frame.
    1. Saves captured frame to media/stream_frames/
    2. Executes STAGE 1: hybrid_predict(frame_path) (YOLOv8 + Enhanced MC-CNN)
    3. Computes instantaneous occupancy metrics
    4. Records entry in CrowdTimeSeriesEntry
    5. Executes STAGE 2: forecast_crowd_series(...) over session time series
    6. Returns updated live monitoring JSON with future scene forecasts.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"}, status=405)

    session_id = request.POST.get("session_id")
    if not session_id:
        return JsonResponse({"error": "session_id is required"}, status=400)

    try:
        session = MonitoringSession.objects.get(session_id=session_id)
    except MonitoringSession.DoesNotExist:
        # Create session on the fly if not found
        try:
            max_capacity = int(request.POST.get("max_capacity", 100))
        except (ValueError, TypeError):
            max_capacity = 100
        try:
            interval_seconds = float(request.POST.get("interval_seconds", 3.0))
        except (ValueError, TypeError):
            interval_seconds = 3.0

        session = MonitoringSession.objects.create(
            session_id=session_id,
            title="Auto-Created Stream Session",
            max_capacity=max_capacity,
            interval_seconds=interval_seconds,
        )

    # Allow dynamic capacity updates from client
    client_capacity = request.POST.get("max_capacity")
    if client_capacity:
        try:
            session.max_capacity = int(client_capacity)
            session.save(update_fields=["max_capacity"])
        except ValueError:
            pass

    # Directory for session frames
    session_media_dir = os.path.join(settings.MEDIA_ROOT, "stream_frames", session.session_id)
    os.makedirs(session_media_dir, exist_ok=True)

    frame_file = request.FILES.get("frame")
    frame_base64 = request.POST.get("frame_base64")

    # Get frame number
    existing_count = session.entries.count()
    frame_number = existing_count + 1

    filename = f"frame_{frame_number:04d}.jpg"
    frame_disk_path = os.path.join(session_media_dir, filename)

    if frame_file:
        with open(frame_disk_path, "wb+") as destination:
            for chunk in frame_file.chunks():
                destination.write(chunk)
    elif frame_base64:
        # Handle data:image/jpeg;base64,...
        if "," in frame_base64:
            frame_base64 = frame_base64.split(",", 1)[1]
        try:
            img_data = base64.b64decode(frame_base64)
            with open(frame_disk_path, "wb") as f:
                f.write(img_data)
        except Exception as e:
            return JsonResponse({"error": f"Failed to decode base64 frame: {str(e)}"}, status=400)
    else:
        return JsonResponse({"error": "No frame received (file or base64)"}, status=400)

    # --------------------------------------------------------------------------
    # STAGE 1: SPATIAL CROWD ESTIMATION (Existing YOLOv8 + MC-CNN Hybrid Model)
    # --------------------------------------------------------------------------
    try:
        (
            people_count,
            result_image_path,
            model_used,
            yolo_count,
            mccnn_count
        ) = hybrid_predict(frame_disk_path)
    except Exception as e:
        print(f"Error during hybrid_predict on frame {frame_number}: {e}")
        return JsonResponse({"error": f"AI Prediction Error: {str(e)}"}, status=500)

    # Metrics
    density_level = get_density_level(people_count)
    if session.max_capacity > 0:
        occupancy_pct = (people_count / session.max_capacity) * 100.0
    else:
        occupancy_pct = 0.0

    occupancy_status = get_occupancy_status(occupancy_pct)

    # Compute URL paths
    rel_frame_path = os.path.relpath(frame_disk_path, settings.MEDIA_ROOT).replace("\\", "/")
    frame_url = f"{settings.MEDIA_URL}{rel_frame_path}"

    rel_result_path = os.path.relpath(result_image_path, settings.MEDIA_ROOT).replace("\\", "/")
    result_url = f"{settings.MEDIA_URL}{rel_result_path}"

    # --------------------------------------------------------------------------
    # STORE CROWD COUNTS WITH TIMESTAMPS
    # --------------------------------------------------------------------------
    entry = CrowdTimeSeriesEntry.objects.create(
        session=session,
        frame_number=frame_number,
        timestamp=timezone.now(),
        frame_image_url=frame_url,
        result_image_url=result_url,
        people_count=people_count,
        yolo_count=yolo_count,
        mccnn_count=mccnn_count,
        model_used=model_used,
        density_level=density_level,
        occupancy_percentage=occupancy_pct,
        occupancy_status=occupancy_status,
    )

    # --------------------------------------------------------------------------
    # STAGE 2: TEMPORAL CROWD FORECASTING
    # --------------------------------------------------------------------------
    all_entries = list(session.entries.order_by("timestamp"))
    historical_counts = [e.people_count for e in all_entries]

    forecasting_result = forecast_crowd_series(
        counts=historical_counts,
        max_capacity=session.max_capacity,
        interval_seconds=session.interval_seconds,
        future_steps=4,
    )

    # Build lightweight history log for frontend chart and timeline
    history_records = [
        {
            "frame_number": e.frame_number,
            "timestamp": e.timestamp.strftime("%H:%M:%S"),
            "count": e.people_count,
            "model_used": e.model_used,
            "density_level": e.density_level,
            "occupancy_pct": round(e.occupancy_percentage, 1),
            "occupancy_status": e.occupancy_status,
            "result_url": e.result_image_url,
        }
        for e in all_entries
    ]

    return JsonResponse({
        "status": "success",
        "session_id": session.session_id,
        "frame_number": frame_number,
        "timestamp": entry.timestamp.strftime("%H:%M:%S"),
        "people_count": people_count,
        "model_used": model_used,
        "yolo_count": yolo_count,
        "mccnn_count": round(mccnn_count) if mccnn_count is not None else None,
        "density_level": density_level,
        "occupancy_percentage": round(occupancy_pct, 1),
        "occupancy_status": occupancy_status,
        "frame_url": frame_url,
        "result_url": result_url,
        "max_capacity": session.max_capacity,
        "interval_seconds": session.interval_seconds,
        "total_frames": len(history_records),
        "history": history_records,
        "forecasting": forecasting_result,
    })


def api_get_session_data(request, session_id):
    """
    Returns full time-series history and latest temporal forecast for a session.
    """
    session = get_object_or_404(MonitoringSession, session_id=session_id)
    all_entries = list(session.entries.order_by("timestamp"))
    counts = [e.people_count for e in all_entries]

    forecasting_result = forecast_crowd_series(
        counts=counts,
        max_capacity=session.max_capacity,
        interval_seconds=session.interval_seconds,
        future_steps=4,
    )

    history_records = [
        {
            "frame_number": e.frame_number,
            "timestamp": e.timestamp.strftime("%H:%M:%S"),
            "count": e.people_count,
            "yolo_count": e.yolo_count,
            "mccnn_count": round(e.mccnn_count) if e.mccnn_count is not None else None,
            "model_used": e.model_used,
            "density_level": e.density_level,
            "occupancy_pct": round(e.occupancy_percentage, 1),
            "occupancy_status": e.occupancy_status,
            "result_url": e.result_image_url,
            "frame_url": e.frame_image_url,
        }
        for e in all_entries
    ]

    return JsonResponse({
        "status": "success",
        "session_id": session.session_id,
        "max_capacity": session.max_capacity,
        "interval_seconds": session.interval_seconds,
        "total_frames": len(history_records),
        "history": history_records,
        "forecasting": forecasting_result,
    })


@csrf_exempt
def api_reset_session(request, session_id):
    """
    Clears all entries for a session to restart monitoring.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"}, status=405)

    session = get_object_or_404(MonitoringSession, session_id=session_id)
    deleted_count, _ = session.entries.all().delete()

    return JsonResponse({
        "status": "success",
        "message": f"Cleared {deleted_count} records for session {session_id}.",
    })


@csrf_exempt
def api_process_video_file(request):
    """
    Analyzes an uploaded video file (e.g. CCTV recording):
    Extracts frames at interval N seconds, passes through hybrid pipeline,
    stores timestamped time-series records, and generates temporal forecasts.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"}, status=405)

    video_file = request.FILES.get("video")
    if not video_file:
        return JsonResponse({"error": "No video file provided"}, status=400)

    try:
        max_capacity = int(request.POST.get("max_capacity", 100))
    except (ValueError, TypeError):
        max_capacity = 100

    try:
        interval_seconds = float(request.POST.get("interval_seconds", 3.0))
    except (ValueError, TypeError):
        interval_seconds = 3.0

    session_id = f"video_{uuid.uuid4().hex[:10]}"
    session = MonitoringSession.objects.create(
        session_id=session_id,
        title=f"Video Analysis: {video_file.name}",
        source_type="video_file",
        source_name=video_file.name,
        max_capacity=max_capacity,
        interval_seconds=interval_seconds,
    )

    # Save video file
    videos_dir = os.path.join(settings.MEDIA_ROOT, "uploaded_videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"{session_id}_{video_file.name}")
    with open(video_path, "wb+") as dest:
        for chunk in video_file.chunks():
            dest.write(chunk)

    # Frame extraction directory
    frames_dir = os.path.join(settings.MEDIA_ROOT, "stream_frames", session_id)
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return JsonResponse({"error": "Could not open uploaded video file"}, status=400)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(round(fps * interval_seconds)))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    sampled_frame_num = 0
    processed_entries = []

    # Limit to reasonable number of frames (e.g. 50 max) for instant response
    max_samples = 40

    while cap.isOpened() and sampled_frame_num < max_samples:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            sampled_frame_num += 1
            frame_filename = f"frame_{sampled_frame_num:04d}.jpg"
            frame_disk_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_disk_path, frame)

            # Spatial Prediction
            (
                people_count,
                result_image_path,
                model_used,
                yolo_count,
                mccnn_count
            ) = hybrid_predict(frame_disk_path)

            density_level = get_density_level(people_count)
            occ_pct = (people_count / max_capacity * 100.0) if max_capacity > 0 else 0.0
            occ_status = get_occupancy_status(occ_pct)

            rel_frame_path = os.path.relpath(frame_disk_path, settings.MEDIA_ROOT).replace("\\", "/")
            frame_url = f"{settings.MEDIA_URL}{rel_frame_path}"

            rel_result_path = os.path.relpath(result_image_path, settings.MEDIA_ROOT).replace("\\", "/")
            result_url = f"{settings.MEDIA_URL}{rel_result_path}"

            entry = CrowdTimeSeriesEntry.objects.create(
                session=session,
                frame_number=sampled_frame_num,
                timestamp=timezone.now(),
                frame_image_url=frame_url,
                result_image_url=result_url,
                people_count=people_count,
                yolo_count=yolo_count,
                mccnn_count=mccnn_count,
                model_used=model_used,
                density_level=density_level,
                occupancy_percentage=occ_pct,
                occupancy_status=occ_status,
            )
            processed_entries.append(entry)

        frame_idx += 1

    cap.release()

    counts = [e.people_count for e in processed_entries]
    forecasting_result = forecast_crowd_series(
        counts=counts,
        max_capacity=max_capacity,
        interval_seconds=interval_seconds,
        future_steps=4,
    )

    history_records = [
        {
            "frame_number": e.frame_number,
            "timestamp": e.timestamp.strftime("%H:%M:%S"),
            "count": e.people_count,
            "yolo_count": e.yolo_count,
            "mccnn_count": round(e.mccnn_count) if e.mccnn_count is not None else None,
            "model_used": e.model_used,
            "density_level": e.density_level,
            "occupancy_pct": round(e.occupancy_percentage, 1),
            "occupancy_status": e.occupancy_status,
            "result_url": e.result_image_url,
            "frame_url": e.frame_image_url,
        }
        for e in processed_entries
    ]

    return JsonResponse({
        "status": "success",
        "session_id": session.session_id,
        "max_capacity": max_capacity,
        "interval_seconds": interval_seconds,
        "total_frames": len(history_records),
        "history": history_records,
        "forecasting": forecasting_result,
        "latest_result_url": history_records[-1]["result_url"] if history_records else "",
        "latest_frame_url": history_records[-1]["frame_url"] if history_records else "",
    })


def download_stream_report(request, session_id):
    """
    Generates and returns an official PDF report for a live stream monitoring session,
    incorporating time series history and Stage 2 temporal forecasts.
    """
    session = get_object_or_404(MonitoringSession, session_id=session_id)
    entries = list(session.entries.order_by("timestamp"))

    if not entries:
        return render(request, "detector/live_monitoring.html", {
            "error_message": "No frames recorded for this session yet."
        })

    counts = [e.people_count for e in entries]
    forecast_info = forecast_crowd_series(
        counts=counts,
        max_capacity=session.max_capacity,
        interval_seconds=session.interval_seconds,
        future_steps=4,
    )

    session_info = {
        "session_id": session.session_id,
        "title": session.title,
        "source_type": session.source_type,
        "max_capacity": session.max_capacity,
        "interval_seconds": session.interval_seconds,
    }

    # Locate latest original image and latest result image on disk
    latest_entry = entries[-1]
    latest_orig_image = None
    latest_result_image = None

    if latest_entry.frame_image_url:
        rel_path = latest_entry.frame_image_url.replace(settings.MEDIA_URL, "").replace("/", os.sep)
        cand = os.path.join(settings.MEDIA_ROOT, rel_path)
        if os.path.exists(cand):
            latest_orig_image = cand

    if latest_entry.result_image_url:
        rel_path = latest_entry.result_image_url.replace(settings.MEDIA_URL, "").replace("/", os.sep)
        cand = os.path.join(settings.MEDIA_ROOT, rel_path)
        if os.path.exists(cand):
            latest_result_image = cand

    pdf_path = os.path.join(
        tempfile.gettempdir(),
        f"Crowd_Monitoring_Report_{session.session_id}.pdf"
    )

    generate_stream_pdf(
        pdf_path=pdf_path,
        session_info=session_info,
        forecast_info=forecast_info,
        entries_list=entries,
        latest_orig_image=latest_orig_image,
        latest_result_image=latest_result_image,
    )

    return FileResponse(
        open(pdf_path, "rb"),
        as_attachment=True,
        filename=f"Crowd_Monitoring_Report_{session.session_id}.pdf",
    )
