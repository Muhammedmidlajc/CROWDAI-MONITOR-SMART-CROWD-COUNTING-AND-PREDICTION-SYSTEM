import os
import tempfile

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse

from ai.predict import predict_image
from ai.pdf_report import generate_pdf


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

    if request.method == "POST" and request.FILES.get("image"):

        image = request.FILES["image"]

        fs = FileSystemStorage()

        filename = fs.save(image.name, image)

        image_url = fs.url(filename)

        image_path = fs.path(filename)

        people_count, heatmap_path = predict_image(image_path)

        heatmap_url = fs.url(os.path.basename(heatmap_path))

        density_level = get_density_level(people_count)

        # Save values for PDF download
        request.session["image_path"] = image_path
        request.session["heatmap_path"] = heatmap_path
        request.session["people_count"] = people_count
        request.session["density_level"] = density_level

        context = {
            "image_url": image_url,
            "heatmap_url": heatmap_url,
            "people_count": people_count,
            "density_level": density_level,
        }

    return render(request, "detector/home.html", context)


def download_report(request):

    image_path = request.session.get("image_path")
    heatmap_path = request.session.get("heatmap_path")
    people_count = request.session.get("people_count")
    density_level = request.session.get("density_level")

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
        density_level
    )

    return FileResponse(
        open(pdf_path, "rb"),
        as_attachment=True,
        filename="Crowd_Density_Report.pdf"
    )