from django.db import models
from django.utils import timezone


class MonitoringSession(models.Model):
    """
    Represents a live video crowd monitoring session.
    Stores session metadata, maximum space capacity, and capture interval.
    """
    session_id = models.CharField(max_length=64, unique=True, db_index=True)
    title = models.CharField(max_length=150, default="Live Crowd Monitoring")
    source_type = models.CharField(max_length=50, default="webcam")  # webcam, video_file, rtsp
    source_name = models.CharField(max_length=255, blank=True, null=True)
    max_capacity = models.PositiveIntegerField(default=100)
    interval_seconds = models.FloatField(default=3.0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} ({self.session_id}) - Cap: {self.max_capacity}"


class CrowdTimeSeriesEntry(models.Model):
    """
    Stores individual sampled frames with timestamps and AI crowd estimates.
    Forms the historical time series for Stage 2 temporal forecasting.
    """
    session = models.ForeignKey(
        MonitoringSession,
        on_delete=models.CASCADE,
        related_name="entries"
    )
    frame_number = models.PositiveIntegerField(default=1)
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    frame_image_url = models.CharField(max_length=500, blank=True, default="")
    result_image_url = models.CharField(max_length=500, blank=True, default="")
    people_count = models.IntegerField(default=0)
    yolo_count = models.IntegerField(null=True, blank=True)
    mccnn_count = models.FloatField(null=True, blank=True)
    model_used = models.CharField(max_length=60, default="YOLOv8")
    density_level = models.CharField(max_length=60, default="Low")
    occupancy_percentage = models.FloatField(default=0.0)
    occupancy_status = models.CharField(max_length=60, default="Low Occupancy")

    class Meta:
        ordering = ["timestamp"]

    def __str__(self):
        return f"Frame #{self.frame_number} @ {self.timestamp.strftime('%H:%M:%S')} - Count: {self.people_count}"
