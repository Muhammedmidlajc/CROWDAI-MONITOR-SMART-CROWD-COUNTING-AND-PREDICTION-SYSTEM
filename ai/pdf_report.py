import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)


def generate_pdf(
    pdf_path,
    original_image,
    result_image,
    final_count,
    yolo_count,
    mccnn_count,
    model_used,
    density_level,
    max_capacity,
    occupancy_percentage,
    occupancy_status,
):

    # --------------------------------
    # PDF Document
    # --------------------------------

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    elements = []


    # --------------------------------
    # Title
    # --------------------------------

    elements.append(
        Paragraph(
            "<font size=20><b>"
            "AI Crowd Density & Occupancy Report"
            "</b></font>",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 8))

    elements.append(
        Paragraph(
            "<font size=11>"
            "Hybrid YOLOv8 + Enhanced MC-CNN Analysis"
            "</font>",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))


    # --------------------------------
    # Date & Time
    # --------------------------------

    now = datetime.now()

    elements.append(
        Paragraph(
            f"<b>Date :</b> {now.strftime('%d-%m-%Y')}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Time :</b> {now.strftime('%I:%M %p')}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))


    # --------------------------------
    # Crowd & Occupancy Summary
    # --------------------------------

    occupancy_rate = f"{occupancy_percentage:.1f}%"

    mccnn_value = (
        "--"
        if mccnn_count is None
        else str(round(mccnn_count))
    )

    table_data = [

        ["Final Crowd Count", str(final_count)],

        ["Maximum Space Capacity", str(max_capacity)],

        ["Occupancy Rate", occupancy_rate],

        ["Occupancy Status", occupancy_status],

        ["Density Level", density_level],

        ["YOLO Count", str(yolo_count)],

        ["MC-CNN Estimate", mccnn_value],

        ["Selected Model", model_used],

    ]


    summary_table = Table(
        table_data,
        colWidths=[220, 230]
    )


    summary_table.setStyle(

        TableStyle(

            [

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.black
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#E3F2FD")
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold"
                ),

                (
                    "FONTNAME",
                    (1, 0),
                    (1, -1),
                    "Helvetica"
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    9
                ),

                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),

            ]

        )

    )

    elements.append(summary_table)

    elements.append(Spacer(1, 25))


    # --------------------------------
    # Occupancy Analysis
    # --------------------------------

    elements.append(

        Paragraph(
            "<b>Occupancy Analysis</b>",
            styles["Heading2"]
        )

    )

    elements.append(Spacer(1, 8))


    occupancy_text = (

        f"The system estimated <b>{final_count}</b> "
        f"people in a space with a maximum capacity of "
        f"<b>{max_capacity}</b>. "

        f"The calculated occupancy rate is "
        f"<b>{occupancy_rate}</b>, resulting in an "
        f"occupancy status of <b>{occupancy_status}</b>."
    )


    elements.append(

        Paragraph(
            occupancy_text,
            styles["Normal"]
        )

    )

    elements.append(Spacer(1, 20))


    # --------------------------------
    # Original Image
    # --------------------------------

    elements.append(

        Paragraph(
            "<b>Original Image</b>",
            styles["Heading2"]
        )

    )

    elements.append(Spacer(1, 8))


    elements.append(

        Image(
            original_image,
            width=300,
            height=220
        )

    )

    elements.append(Spacer(1, 20))


    # --------------------------------
    # Result Image
    # --------------------------------

    if model_used == "YOLOv8":

        heading = "YOLO Detection"

    else:

        heading = "Density Heatmap"


    elements.append(

        Paragraph(
            f"<b>{heading}</b>",
            styles["Heading2"]
        )

    )

    elements.append(Spacer(1, 8))


    elements.append(

        Image(
            result_image,
            width=300,
            height=220
        )

    )

    elements.append(Spacer(1, 20))


    # --------------------------------
    # Technology Used
    # --------------------------------

    elements.append(

        Paragraph(
            "<b>Technology Used</b>",
            styles["Heading2"]
        )

    )

    elements.append(Spacer(1, 8))


    tech_table = [

        ["Framework", "Django"],

        ["Programming Language", "Python"],

        ["Deep Learning Framework", "PyTorch"],

        ["Model 1", "YOLOv8 Person Detection"],

        ["Model 2", "Enhanced MC-CNN"],

        ["Backbone", "MobileNetV2"],

        ["Image Processing", "OpenCV"],

    ]


    technology_table = Table(

        tech_table,

        colWidths=[220, 230]

    )


    technology_table.setStyle(

        TableStyle(

            [

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.black
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, -1),
                    "Helvetica"
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

            ]

        )

    )

    elements.append(technology_table)

    elements.append(Spacer(1, 25))


    # --------------------------------
    # Footer
    # --------------------------------

    elements.append(

        Paragraph(
            "<b>Generated By :</b> "
            "AI Crowd Density Estimation System",
            styles["Italic"]
        )

    )

    elements.append(

        Paragraph(
            "MSc Computer Science Mini Project",
            styles["Italic"]
        )

    )


    # --------------------------------
    # Build PDF
    # --------------------------------

    doc.build(elements)


def generate_stream_pdf(
    pdf_path,
    session_info,
    forecast_info,
    entries_list,
    latest_orig_image=None,
    latest_result_image=None,
):
    """
    Generates a comprehensive PDF report for a Live Video Stream
    Crowd Monitoring and Temporal Forecasting session.
    """
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=35,
        bottomMargin=35,
    )

    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(
        Paragraph(
            "<font size=18><b>AI Crowd Monitoring & Temporal Forecasting Report</b></font>",
            styles["Title"],
        )
    )
    elements.append(Spacer(1, 6))
    elements.append(
        Paragraph(
            "<font size=10 color='#555'>Two-Stage AI Pipeline: Spatial Hybrid Estimation (YOLOv8 + MC-CNN) & Temporal Time-Series Forecasting</font>",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 14))

    # Session Metadata
    now = datetime.now()
    session_id = session_info.get("session_id", "N/A")
    source_type = session_info.get("source_type", "Live Video Stream").title()
    max_cap = session_info.get("max_capacity", 100)
    interval = session_info.get("interval_seconds", 3.0)
    total_frames = len(entries_list)

    meta_table_data = [
        ["Report Generated:", now.strftime("%d-%m-%Y %I:%M %p"), "Session ID:", session_id[:16]],
        ["Video Source:", source_type, "Sampling Interval:", f"Every {interval}s"],
        ["Maximum Space Capacity:", f"{max_cap} persons", "Total Analyzed Frames:", str(total_frames)],
    ]
    meta_table = Table(meta_table_data, colWidths=[130, 130, 120, 110])
    meta_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F4F8")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8DC")),
        ])
    )
    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    # Executive Summary / Stage 2 Future Scene Prediction
    elements.append(Paragraph("<b>Stage 2: Future Crowd Condition & Temporal Forecasting</b>", styles["Heading2"]))
    elements.append(Spacer(1, 6))

    curr_count = forecast_info.get("current_count", 0)
    fut_count = forecast_info.get("future_crowd_count", curr_count)
    fut_occ = forecast_info.get("future_occupancy_percentage", 0.0)
    fut_status = forecast_info.get("future_occupancy_status", "Low Occupancy")
    trend_dir = forecast_info.get("trend_direction", "Stable")
    velocity = forecast_info.get("slope_per_min", 0.0)
    risk_alert = forecast_info.get("risk_alert", "Normal")
    eta_text = forecast_info.get("time_to_capacity_text", "N/A")

    forecast_table_data = [
        ["Current Observed Count", str(curr_count), "Stage 2 Forecast Method", "Holt's Linear Trend"],
        ["Predicted Future Count", str(fut_count), "Future Occupancy Rate", f"{fut_occ:.1f}%"],
        ["Future Occupancy Status", fut_status, "Crowd Trend Direction", trend_dir],
        ["Crowd Growth Velocity", f"{velocity:+.1f} people/min", "Time-To-Capacity ETA", eta_text],
        ["Crowd Risk Alert", risk_alert, "Model Confidence (R²)", f"{forecast_info.get('r_squared', 1.0):.2f}"],
    ]
    f_table = Table(forecast_table_data, colWidths=[140, 110, 130, 110])
    f_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#B0BEC5")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E1F5FE")),
            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#E1F5FE")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("ALIGN", (3, 0), (3, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    elements.append(f_table)
    elements.append(Spacer(1, 14))

    # 1. Captured Frame Time-Series Log (Recent Entries)
    if entries_list:
        elements.append(Paragraph("<b>Captured Frame Time-Series Log (Recent Entries)</b>", styles["Heading3"]))
        elements.append(Spacer(1, 4))
        log_data = [["Frame #", "Timestamp", "Model Selected", "Count", "Density", "Occupancy %", "Status"]]
        sample_entries = entries_list[-12:]
        for ent in sample_entries:
            ts_str = ent.get("timestamp_str", "") if isinstance(ent, dict) else ent.timestamp.strftime("%H:%M:%S")
            p_cnt = ent.get("people_count", 0) if isinstance(ent, dict) else ent.people_count
            m_used = ent.get("model_used", "") if isinstance(ent, dict) else ent.model_used
            d_lvl = ent.get("density_level", "") if isinstance(ent, dict) else ent.density_level
            o_pct = ent.get("occupancy_percentage", 0.0) if isinstance(ent, dict) else ent.occupancy_percentage
            o_stat = ent.get("occupancy_status", "") if isinstance(ent, dict) else ent.occupancy_status
            f_num = ent.get("frame_number", 1) if isinstance(ent, dict) else ent.frame_number
            log_data.append([
                f"#{f_num}",
                ts_str,
                m_used,
                str(p_cnt),
                d_lvl,
                f"{o_pct:.1f}%",
                o_stat,
            ])
        log_table = Table(log_data, colWidths=[50, 70, 95, 50, 70, 75, 85])
        log_table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8DC")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        elements.append(log_table)
        elements.append(Spacer(1, 14))

    # 2. Multi-Step Future Crowd Density Forecast Horizon Table (After Captured Frames)
    step_forecasts = forecast_info.get("step_forecasts", [])
    if step_forecasts:
        elements.append(Paragraph("<b>Future Crowd Density Projection Schedule</b>", styles["Heading3"]))
        elements.append(Spacer(1, 4))
        h_data = [["Horizon", "Predicted Count", "Projected Occupancy %", "Density Level", "Future Status"]]
        for sf in step_forecasts:
            h_data.append([
                sf.get("time_label", f"+{sf.get('horizon_seconds', 0)}s"),
                str(sf.get("predicted_count", 0)),
                f"{sf.get('occupancy_percentage', 0.0):.1f}%",
                sf.get("density_level", "Low"),
                sf.get("occupancy_status", "Low Occupancy"),
            ])
        h_table = Table(h_data, colWidths=[80, 95, 125, 95, 100])
        h_table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B0BEC5")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0288D1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(h_table)
        elements.append(Spacer(1, 14))

    # Frame Images (if present)
    images_to_show = []
    if latest_orig_image and os.path.exists(latest_orig_image):
        images_to_show.append(("Latest Captured Video Frame", latest_orig_image))
    if latest_result_image and os.path.exists(latest_result_image):
        images_to_show.append(("Latest AI Prediction Output", latest_result_image))

    if images_to_show:
        elements.append(Paragraph("<b>Visual Crowd Evidence</b>", styles["Heading3"]))
        elements.append(Spacer(1, 6))
        img_row = []
        for title, img_path in images_to_show:
            img_row.append([
                Paragraph(f"<b>{title}</b>", styles["Normal"]),
                Image(img_path, width=220, height=150)
            ])
        for title, img_path in images_to_show:
            elements.append(Paragraph(f"<b>{title}</b>", styles["Normal"]))
            elements.append(Spacer(1, 3))
            elements.append(Image(img_path, width=240, height=160))
            elements.append(Spacer(1, 8))

    # Footer
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<b>MSc Computer Science Mini Project — Live Crowd Monitoring & Forecasting</b>", styles["Italic"]))
    doc.build(elements)