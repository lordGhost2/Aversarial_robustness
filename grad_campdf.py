import os
from fpdf import FPDF
from PIL import Image

# Path where Grad-CAM images are stored
output_dir = "gradcam_outputs"
image_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])

# Create PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

for img_file in image_files:
    img_path = os.path.join(output_dir, img_file)

    # Open image and resize to 256x256 for clarity
    image = Image.open(img_path).convert("RGB")
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    temp_path = "temp_highres.png"
    image.save(temp_path)

    # Add to PDF
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=img_file, ln=1, align='C')
    pdf.image(temp_path, x=25, y=30, w=160)  # Centered and clearer

# Save the PDF
output_pdf = "GradCAM_Report_HighRes.pdf"
pdf.output(output_pdf)
print(f"✅ Grad-CAM PDF report saved as: {output_pdf}")
