import sys
import json
from PIL import Image
from ai_service import analyze_billboard
import os

def main():
    if len(sys.argv) < 4 or sys.argv[1] in ["-h", "--help"]:
        print("Usage: python test.py <image_path> <state> <area_type> [visualize]")
        sys.exit(0)

    image_path = sys.argv[1]
    state = sys.argv[2]
    area_type = sys.argv[3]
    visualize = False

    if len(sys.argv) >= 5:
        visualize_arg = sys.argv[4].lower()
        visualize = visualize_arg in ["true", "1", "yes"]

    if not os.path.exists(image_path):
        print(json.dumps({"error": "Image path does not exist"}))
        sys.exit(1)

    try:
        img = Image.open(image_path).convert("RGB")
        result = analyze_billboard(img, state=state, area_type=area_type, visualize=visualize)

        analysis = result.get("analysis", {})

        
        output = {
            "oversized": bool(result.get("oversized", False)),
            "billboard_area_percentage": float(result.get("billboard_area_percentage", 0.0)),
            "billboard_angle_deg": analysis.get("billboard_angle_deg", None),
            "croppedImageUrl": result.get("croppedImageUrl", None),
            "analysis": analysis,
            "content_text": result.get("content_text", None),             
            "content_category": result.get("content_category", "Unknown") 
        }
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
