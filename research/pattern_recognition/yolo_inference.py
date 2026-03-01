"""
YOLOv8 Pattern Detection Inference Module.

Uses the pre-trained model from HuggingFace:
https://huggingface.co/foduucom/stockmarket-pattern-detection-yolov8

Detected patterns:
- Head and shoulders bottom
- Head and shoulders top
- M_Head (double top variant)
- StockLine
- Triangle
- W_Bottom (double bottom)
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class YOLOPatternDetector:
    """
    YOLOv8 pattern detection wrapper.

    Loads the pre-trained stock pattern detection model and runs inference
    on chart images.
    """

    # Class names from the HuggingFace model
    CLASSES = [
        "Head and shoulders bottom",
        "Head and shoulders top",
        "M_Head",
        "StockLine",
        "Triangle",
        "W_Bottom"
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = 0.25,
        device: Optional[str] = None
    ):
        """
        Initialize the YOLO pattern detector.

        Args:
            model_path: Path to local model file. If None, downloads from HuggingFace.
            confidence: Confidence threshold for detections (0-1).
            device: Device to run on ('cuda', 'cpu', or None for auto).
        """
        self.confidence = confidence
        self.device = device
        self.model = None
        self.model_path = model_path

        self._load_model()

    def _load_model(self):
        """Load the YOLO model from local path or HuggingFace."""
        try:
            from ultralytics import YOLO

            if self.model_path is None:
                # Download from HuggingFace
                logger.info("Downloading model from HuggingFace...")
                from huggingface_hub import hf_hub_download

                self.model_path = hf_hub_download(
                    repo_id="foduucom/stockmarket-pattern-detection-yolov8",
                    filename="model.pt"
                )
                logger.info(f"Model downloaded to: {self.model_path}")

            self.model = YOLO(self.model_path)

            if self.device:
                self.model.to(self.device)

            logger.info(f"YOLO model loaded: {self.model_path}")

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            )
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Run detection on a chart image.

        Args:
            image_path: Path to the chart image file.

        Returns:
            List of detection dictionaries with keys:
            - class_id: int
            - class_name: str
            - confidence: float
            - bbox: [x1, y1, x2, y2] pixel coordinates
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        results = self.model(image_path, conf=self.confidence, verbose=False)

        detections = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                detections.append({
                    'class_id': class_id,
                    'class_name': self.model.names[class_id] if class_id in self.model.names else f"class_{class_id}",
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                })

        return detections

    def detect_and_save(
        self,
        image_path: str,
        output_path: str,
        line_width: int = 2,
        font_scale: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns and save annotated image.

        Args:
            image_path: Path to input chart image.
            output_path: Path to save annotated image.
            line_width: Bounding box line width.
            font_scale: Label font scale.

        Returns:
            List of detections (same format as detect()).
        """
        import cv2

        if self.model is None:
            raise RuntimeError("Model not loaded")

        results = self.model(image_path, conf=self.confidence, verbose=False)

        # Get annotated image from YOLO
        annotated = results[0].plot(line_width=line_width, font_size=font_scale)

        # Save annotated image
        cv2.imwrite(output_path, annotated)
        logger.info(f"Saved annotated image: {output_path}")

        # Parse detections
        detections = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                detections.append({
                    'class_id': class_id,
                    'class_name': self.model.names[class_id] if class_id in self.model.names else f"class_{class_id}",
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                })

        return detections

    def detect_batch(
        self,
        image_paths: List[str],
        save_annotated: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run detection on multiple images.

        Args:
            image_paths: List of image file paths.
            save_annotated: Whether to save annotated images.
            output_dir: Directory for annotated images (required if save_annotated=True).

        Returns:
            Dictionary mapping image path to list of detections.
        """
        results = {}

        for img_path in image_paths:
            img_path = str(img_path)

            if save_annotated and output_dir:
                output_path = Path(output_dir) / f"{Path(img_path).stem}_yolo.png"
                detections = self.detect_and_save(img_path, str(output_path))
            else:
                detections = self.detect(img_path)

            results[img_path] = detections

            if detections:
                logger.info(
                    f"{Path(img_path).name}: Found {len(detections)} pattern(s) - "
                    f"{', '.join(d['class_name'] for d in detections)}"
                )
            else:
                logger.debug(f"{Path(img_path).name}: No patterns detected")

        return results


def run_yolo_detection(
    image_paths: List[str],
    output_dir: str,
    model_path: Optional[str] = None,
    confidence: float = 0.25,
    save_annotated: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run YOLO detection on multiple images.

    Args:
        image_paths: List of chart image paths.
        output_dir: Directory for output files.
        model_path: Optional path to local model.
        confidence: Detection confidence threshold.
        save_annotated: Whether to save annotated images.

    Returns:
        Dictionary with detection results and statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    detector = YOLOPatternDetector(
        model_path=model_path,
        confidence=confidence
    )

    # Run detection
    results = detector.detect_batch(
        image_paths=image_paths,
        save_annotated=save_annotated,
        output_dir=str(output_dir)
    )

    # Compile statistics
    total_detections = sum(len(dets) for dets in results.values())
    images_with_patterns = sum(1 for dets in results.values() if dets)

    # Count by class
    class_counts = {}
    for dets in results.values():
        for det in dets:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return {
        'detections': results,
        'stats': {
            'total_images': len(image_paths),
            'images_with_patterns': images_with_patterns,
            'total_detections': total_detections,
            'class_counts': class_counts
        }
    }
