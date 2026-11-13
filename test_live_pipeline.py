"""
End-to-End Test for Live Stream Pipeline and Stage 2 Temporal Forecasting
"""
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "crowd_ai.settings")
django.setup()

from detector.models import MonitoringSession, CrowdTimeSeriesEntry
from ai.hybrid_predict import hybrid_predict
from ai.temporal_forecasting import forecast_crowd_series
from ai.pdf_report import generate_stream_pdf
from django.utils import timezone

def run_test():
    print("==================================================")
    print("STARTING LIVE STREAM & FORECASTING PIPELINE TEST")
    print("==================================================")

    # 1. Create a Monitoring Session
    session_id = "test_stream_001"
    MonitoringSession.objects.filter(session_id=session_id).delete()
    session = MonitoringSession.objects.create(
        session_id=session_id,
        title="Test Live Crowd Monitoring",
        source_type="webcam",
        max_capacity=50,
        interval_seconds=3.0,
        is_active=True
    )
    print(f"Created Session: {session}")

    # 2. Stage 1: Run spatial prediction on sample test image
    test_image = "media/download144.jpg"
    print(f"\nRunning Stage 1 Spatial Hybrid Estimation on {test_image}...")
    people_count, result_path, model_used, yolo_count, mccnn_count = hybrid_predict(test_image)
    print(f"-> People Count: {people_count}")
    print(f"-> Model Used: {model_used}")
    print(f"-> Result Path: {result_path}")

    # 3. Simulate sequential time-series entries
    simulated_counts = [12, 18, 25, 32, people_count]
    print(f"\nStoring {len(simulated_counts)} Timestamped Frames in Time-Series...")
    for idx, count in enumerate(simulated_counts):
        CrowdTimeSeriesEntry.objects.create(
            session=session,
            frame_number=idx + 1,
            timestamp=timezone.now(),
            frame_image_url="/media/test.jpg",
            result_image_url="/media/test_result.png",
            people_count=count,
            yolo_count=count,
            mccnn_count=None,
            model_used=model_used,
            density_level="Low" if count < 50 else "Medium",
            occupancy_percentage=(count / session.max_capacity) * 100.0,
            occupancy_status="Low Occupancy" if count < 25 else "Moderate Occupancy"
        )
    print(f"-> Total Entries in DB: {session.entries.count()}")

    # 4. Stage 2: Temporal Crowd Forecasting
    print("\nRunning Stage 2 Temporal Forecasting...")
    db_counts = list(session.entries.order_by("timestamp").values_list("people_count", flat=True))
    forecast_results = forecast_crowd_series(
        counts=db_counts,
        max_capacity=session.max_capacity,
        interval_seconds=session.interval_seconds,
        future_steps=3
    )

    print("--------------------------------------------------")
    print("STAGE 2 FORECASTING OUTPUT:")
    print("--------------------------------------------------")
    print(f"Current Count              : {forecast_results['current_count']}")
    print(f"Future Predicted Count     : {forecast_results['future_crowd_count']}")
    print(f"Future Occupancy %         : {forecast_results['future_occupancy_percentage']}%")
    print(f"Future Density Level       : {forecast_results['future_density_level']}")
    print(f"Future Occupancy Status    : {forecast_results['future_occupancy_status']}")
    print(f"Trend Direction            : {forecast_results['trend_direction']}")
    print(f"Growth Velocity (rate/min) : {forecast_results['slope_per_min']} people/min")
    print(f"Time to Capacity (ETA)     : {forecast_results['time_to_capacity_text']}")
    print(f"Crowd Risk Alert           : {forecast_results['risk_alert']}")
    print("Multi-Step Forecasts:")
    for sf in forecast_results["step_forecasts"]:
        print(f"  Horizon {sf['time_label']}: {sf['predicted_count']} people ({sf['occupancy_percentage']}%, {sf['occupancy_status']})")

    # 5. Test Live Stream PDF Report Generation
    print("\nTesting Stream PDF Report Generation...")
    pdf_out = os.path.join(os.path.dirname(__file__), "reports", f"Test_Report_{session_id}.pdf")
    os.makedirs(os.path.dirname(pdf_out), exist_ok=True)
    generate_stream_pdf(
        pdf_path=pdf_out,
        session_info={
            "session_id": session.session_id,
            "title": session.title,
            "source_type": session.source_type,
            "max_capacity": session.max_capacity,
            "interval_seconds": session.interval_seconds,
        },
        forecast_info=forecast_results,
        entries_list=list(session.entries.order_by("timestamp")),
        latest_orig_image=test_image,
        latest_result_image=result_path,
    )
    assert os.path.exists(pdf_out), "PDF was not created!"
    print(f"-> Generated PDF Report successfully: {pdf_out} (Size: {os.path.getsize(pdf_out)} bytes)")

    print("\n==================================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("==================================================")

if __name__ == "__main__":
    run_test()
