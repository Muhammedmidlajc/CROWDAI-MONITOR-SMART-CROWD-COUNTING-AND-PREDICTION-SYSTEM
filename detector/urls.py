from django.urls import path
from . import views

urlpatterns = [
    # Existing Single Image Analysis
    path("", views.home, name="home"),
    path("download-report/", views.download_report, name="download_report"),

    # New Live Video Stream & Temporal Forecasting Suite
    path("live/", views.live_monitoring, name="live_monitoring"),
    path("api/create-session/", views.api_create_session, name="api_create_session"),
    path("api/process-frame/", views.api_process_frame, name="api_process_frame"),
    path("api/session-data/<str:session_id>/", views.api_get_session_data, name="api_get_session_data"),
    path("api/reset-session/<str:session_id>/", views.api_reset_session, name="api_reset_session"),
    path("api/process-video-file/", views.api_process_video_file, name="api_process_video_file"),
    path("download-stream-report/<str:session_id>/", views.download_stream_report, name="download_stream_report"),
]