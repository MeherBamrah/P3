# ── Cell 1 — Install ──────────────────────────────────────────────────────────
!pip install azure-ai-vision-imageanalysis --quiet
# ── Cell 2 — Credentials (Colab Secrets) ─────────────────────────────────────
from google.colab import userdata

ENDPOINT = userdata.get("AZURE_VISION_ENDPOINT")   # https://<your-resource>.cognitiveservices.azure.com/
KEY      = userdata.get("AZURE_VISION_KEY")

# Fallback — paste directly for quick testing (never commit to GitHub)
# ENDPOINT = "https://<your-resource>.cognitiveservices.azure.com/"
# KEY      = "<your-key>"
# ── Cell 3 — Client setup ─────────────────────────────────────────────────────
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

client = ImageAnalysisClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(KEY)
)
print("✓ Vision client ready")
# ── Cell 4 — Upload YOUR image ───────────────────────────────────────────────
from google.colab import files
import io

print("Upload any image from your device:")
uploaded = files.upload()
filename  = list(uploaded.keys())[0]
img_bytes = uploaded[filename]
print(f"✓ Loaded: {filename}  ({len(img_bytes):,} bytes)")
# ── Cell 5 — Analyse: captions + tags + objects + OCR ────────────────────────
from IPython.display import display, Image as IPImage
import json

result = client.analyze(
    image_data=io.BytesIO(img_bytes),
    visual_features=[
        VisualFeatures.CAPTION,
        VisualFeatures.TAGS,
        VisualFeatures.OBJECTS,
        VisualFeatures.READ,          # OCR
    ],
    gender_neutral_caption=True
)

# Show the image inline
display(IPImage(data=img_bytes, width=480))
# ── Cell 6 — Caption ─────────────────────────────────────────────────────────
cap = result.caption
if cap:
    print(f"📝  Caption   : {cap.text}")
    print(f"    Confidence: {cap.confidence:.0%}")
  # ── Cell 7 — Tags ─────────────────────────────────────────────────────────────
print(f"\n🏷️  Tags  ({len(result.tags)} found)")
print(f"{'Tag':<25} {'Confidence':>10}")
print("-" * 37)
for tag in sorted(result.tags, key=lambda t: t.confidence, reverse=True):
    bar = "█" * int(tag.confidence * 20)
    print(f"{tag.name:<25} {tag.confidence:>9.0%}  {bar}")
  # ── Cell 8 — Object detection ────────────────────────────────────────────────
print(f"\n📦  Objects detected  ({len(result.objects)})")
for obj in result.objects:
    bb = obj.bounding_box
    print(f"  {obj.tags[0].name:<20}  confidence {obj.tags[0].confidence:.0%}  "
          f"bbox: x={bb.x} y={bb.y} w={bb.width} h={bb.height}")
  # ── Cell 9 — OCR: read text from image ───────────────────────────────────────
if result.read and result.read.blocks:
    print(f"\n📄  Text extracted from image:")
    for block in result.read.blocks:
        for line in block.lines:
            print(f"  \"{line.text}\"")
else:
    print("\n📄  No text detected in this image.")
  # ── Cell 10 — Batch helper: analyse multiple images at once ──────────────────
#
# Great for the hands-on section — participants drop images here
# and compare what the model sees in each.

from pathlib import Path
import time

sample_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/320px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
]

for url in sample_urls:
    r = client.analyze_from_url(
        image_url=url,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.TAGS]
    )
    print(f"\n🔗 {url.split('/')[-1]}")
    print(f"   Caption : {r.caption.text}  ({r.caption.confidence:.0%})")
    print(f"   Top tags: {', '.join(t.name for t in sorted(r.tags, key=lambda x: x.confidence, reverse=True)[:5])}")
    time.sleep(0.5)  # stay comfortably under rate limits
