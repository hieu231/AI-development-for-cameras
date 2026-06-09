"""
Face Recognition Engine
Core module for face detection and embedding generation.

Detection backend is pluggable:
  - "retinaface" (default): insightface RetinaFace ONNX detector — much
    better on small / occluded / off-axis faces than MTCNN, and runs on
    Jetson via onnxruntime. Returns 5-point landmarks which are used
    for proper similarity-transform alignment before FaceNet embedding.
  - "mtcnn": legacy facenet_pytorch MTCNN — kept as a fallback so the
    pipeline keeps working if insightface fails to install.

Embedding model (InceptionResnetV1 / FaceNet) is unchanged so existing
profiles (averaged embeddings stored in the DB) stay compatible after
the detector swap.
"""
import os
import sys
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import torch
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)


# ── alignment helper (used by RetinaFace path) ─────────────────────────
# Standard ArcFace 5-point reference landmarks for a 112x112 aligned crop.
# Re-scaled below to whatever output size FaceNet expects (160x160).
_ARCFACE_REF_5PTS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _align_face_by_landmarks(
    bgr_image: np.ndarray,
    landmarks_5: np.ndarray,
    output_size: int = 160,
) -> np.ndarray:
    """Similarity-transform a face crop to ArcFace reference landmarks.

    Uses cv2.estimateAffinePartial2D — same math as insightface's
    `face_align.norm_crop` but without the dependency on skimage.

    Args:
        bgr_image: source image (BGR, uint8).
        landmarks_5: (5,2) array of (x,y) landmarks in image coords —
            order is left-eye, right-eye, nose, left-mouth, right-mouth.
        output_size: target square size (160 for FaceNet).
    Returns:
        Aligned face crop (output_size x output_size x 3, BGR).
    """
    ref = _ARCFACE_REF_5PTS_112 * (output_size / 112.0)
    src = np.asarray(landmarks_5, dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(
        src, ref, method=cv2.LMEDS,
    )
    if matrix is None:
        # Degenerate landmarks — fall back to a centered resize so the
        # caller still gets a usable tensor instead of crashing.
        h, w = bgr_image.shape[:2]
        side = min(h, w)
        cy, cx = h // 2, w // 2
        crop = bgr_image[
            cy - side // 2 : cy + side // 2,
            cx - side // 2 : cx + side // 2,
        ]
        return cv2.resize(crop, (output_size, output_size))
    return cv2.warpAffine(
        bgr_image, matrix, (output_size, output_size), borderValue=0.0,
    )


def _bgr_face_to_facenet_tensor(face_bgr: np.ndarray) -> torch.Tensor:
    """Convert an aligned BGR face crop into the [3,160,160] tensor
    FaceNet (InceptionResnetV1) expects, normalized to [-1, 1].
    """
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    # facenet_pytorch's standard prewhitening — match what MTCNN's
    # post_process=True did for the legacy path so existing profile
    # embeddings stay comparable to fresh detections.
    rgb = (rgb - 127.5) / 128.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return tensor


class _MTCNNBackend:
    """Legacy MTCNN detection path, kept as a safe fallback."""

    name = "mtcnn"

    def __init__(self, device: str):
        from facenet_pytorch import MTCNN

        self.device = device
        self.detector = MTCNN(
            image_size=160,
            margin=0,
            keep_all=True,
            device=device,
            post_process=True,
        )

    def detect(
        self, image: np.ndarray
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[float]], Optional[List[Tuple[int, int, int, int]]]]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        boxes, probs = self.detector.detect(pil_image)
        if boxes is None or len(boxes) == 0:
            return None, None, None
        face_tensors = self.detector.extract(pil_image, boxes, save_path=None)
        if face_tensors is None:
            return None, None, None
        face_list: List[torch.Tensor] = []
        confidences: List[float] = []
        bbox_list: List[Tuple[int, int, int, int]] = []
        if face_tensors.dim() == 3:
            face_tensors = face_tensors.unsqueeze(0)
        img_h, img_w = image.shape[:2]
        for i in range(face_tensors.shape[0]):
            face_list.append(face_tensors[i])
            conf = float(probs[i]) if probs is not None and i < len(probs) else 0.99
            confidences.append(conf)
            box = boxes[i]
            try:
                x1 = int(max(0, min(img_w - 1, float(box[0]))))
                y1 = int(max(0, min(img_h - 1, float(box[1]))))
                x2 = int(max(0, min(img_w - 1, float(box[2]))))
                y2 = int(max(0, min(img_h - 1, float(box[3]))))
            except (TypeError, ValueError, IndexError):
                x1, y1, x2, y2 = 0, 0, 1, 1
            bbox_list.append((x1, y1, x2, y2))
        return face_list, confidences, bbox_list


class _RetinaFaceBackend:
    """RetinaFace ONNX detector via insightface.

    insightface auto-downloads the buffalo_sc / buffalo_l ONNX bundle on
    first init (cached under ~/.insightface/models). On Jetson aarch64
    we run with onnxruntime's CPUExecutionProvider — for the small
    RetinaFace-mnet0.25 model in `buffalo_sc` that's still real-time at
    typical face-cam resolutions and avoids the GPU EP wheel hassle.
    """

    name = "retinaface"

    def __init__(self, device: str, det_size: int = 640):
        import insightface
        import onnxruntime as ort

        self.device = device
        available = ort.get_available_providers()

        # ── Provider selection ─────────────────────────────────────────
        # On Jetson with CUDA 13 host + onnxruntime-gpu 1.23 (CUDA 12.9
        # build), the CUDA EP loads but crashes at kernel launch with
        # `cudaErrorSymbolNotFound` — classic CUDA ABI mismatch. TensorRT
        # EP uses libnvinfer directly and sidesteps the CUDA 12/13
        # mismatch entirely, so we prefer it and only fall back to CUDA
        # EP on x86 hosts where it actually works.
        providers: List[str] = []
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if device == "cuda" and "CUDAExecutionProvider" in available and os.getenv(
            "FACE_ALLOW_CUDA_EP", "false"
        ).lower() == "true":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        # ctx_id is still required by insightface.prepare(), but it
        # doesn't actually pick the EP — we inject providers below.
        ctx_id = 0 if "TensorrtExecutionProvider" in providers or "CUDAExecutionProvider" in providers else -1

        # `buffalo_sc` is the lightweight bundle (RetinaFace mnet0.25 +
        # ArcFace mbf). We only need detection here.
        self.app = insightface.app.FaceAnalysis(
            name=os.getenv("INSIGHTFACE_MODEL", "buffalo_sc"),
            allowed_modules=["detection"],
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        logger.info(
            "RetinaFace providers request: %s | available: %s",
            providers,
            available,
        )

        # Quality filters to stop backpack-mesh / shirt-logo / background
        # patterns from being misclassified as faces. Raising det_score
        # cuts most false positives; min_size kills tiny specks;
        # aspect_ratio kills long rectangular blobs.
        self.min_det_score = float(
            os.getenv("FACE_DET_SCORE_MIN", "0.65")
        )
        self.min_face_side = int(os.getenv("FACE_MIN_SIDE_PX", "40"))
        self.max_aspect_ratio = float(
            os.getenv("FACE_MAX_ASPECT_RATIO", "2.2")
        )

        logger.info(
            "RetinaFace backend initialised (insightface=%s, ctx_id=%d, "
            "det_size=%d, min_det_score=%.2f, min_side=%dpx, "
            "max_aspect=%.1f)",
            os.getenv("INSIGHTFACE_MODEL", "buffalo_sc"),
            ctx_id,
            det_size,
            self.min_det_score,
            self.min_face_side,
            self.max_aspect_ratio,
        )

    def detect(
        self, image: np.ndarray
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[float]], Optional[List[Tuple[int, int, int, int]]]]:
        # insightface expects BGR uint8 — same as our pipeline.
        faces = self.app.get(image)
        if not faces:
            return None, None, None

        face_list: List[torch.Tensor] = []
        confidences: List[float] = []
        bbox_list: List[Tuple[int, int, int, int]] = []
        img_h, img_w = image.shape[:2]
        for face in faces:
            try:
                x1, y1, x2, y2 = (int(v) for v in face.bbox.astype(int))
                x1 = max(0, min(img_w - 1, x1))
                y1 = max(0, min(img_h - 1, y1))
                x2 = max(0, min(img_w - 1, x2))
                y2 = max(0, min(img_h - 1, y2))
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue

            # ── Quality gate ────────────────────────────────────────
            # Reject non-face detections before they waste an embedding
            # inference + clutter the overlay. RetinaFace is
            # well-calibrated so det_score is a reliable signal — 0.65
            # empirically cuts virtually all backpack/clothing patterns
            # while keeping genuine profile + side-lit faces.
            det_score = float(getattr(face, "det_score", 1.0) or 1.0)
            if det_score < self.min_det_score:
                continue
            bw = x2 - x1
            bh = y2 - y1
            if bw < self.min_face_side or bh < self.min_face_side:
                continue
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > self.max_aspect_ratio:
                continue

            # 5 facial landmarks — required for similarity-transform
            # alignment. insightface's RetinaFace returns these as
            # `face.kps` (shape (5, 2), order matches ArcFace ref).
            kps = getattr(face, "kps", None)
            if kps is None:
                # Fallback: just centre-crop the bbox + resize.
                crop = image[y1:y2, x1:x2]
                aligned = cv2.resize(crop, (160, 160))
            else:
                aligned = _align_face_by_landmarks(
                    image, np.asarray(kps, dtype=np.float32), output_size=160,
                )

            face_list.append(_bgr_face_to_facenet_tensor(aligned))
            confidences.append(float(getattr(face, "det_score", 0.99) or 0.99))
            bbox_list.append((x1, y1, x2, y2))

        if not face_list:
            return None, None, None
        return face_list, confidences, bbox_list


class FaceDetector:
    """Face detection front-end. Picks backend by env / kwarg."""

    def __init__(self, device: str = "cuda", backend: Optional[str] = None):
        self.device = device
        backend = (
            backend
            or os.getenv("FACE_DETECTOR_BACKEND", "retinaface")
        ).strip().lower()

        impl = None
        if backend == "retinaface":
            try:
                impl = _RetinaFaceBackend(device=device)
            except Exception as exc:
                logger.warning(
                    "RetinaFace backend failed to initialise (%s) — "
                    "falling back to MTCNN. Install `insightface` to "
                    "enable RetinaFace.",
                    exc,
                )
                impl = None

        if impl is None:
            impl = _MTCNNBackend(device=device)

        self.backend = impl
        logger.info(
            "Face detector initialised on %s using backend=%s",
            device,
            self.backend.name,
        )

    def detect(
        self, image: np.ndarray
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[List[float]], Optional[List[Tuple[int, int, int, int]]]]:
        """Detect and align faces.

        Returns:
            (face_tensors, confidences, bboxes) where face_tensors are
            [3,160,160] FaceNet-ready tensors, confidences are detection
            scores, and bboxes are (x1,y1,x2,y2) ints in image coordinates.
            Each list is the same length and ordered consistently. Returns
            (None, None, None) when nothing was detected.
        """
        return self.backend.detect(image)


class FaceEmbedder:
    """
    Face embedding generation using InceptionResnetV1
    Can be easily replaced with other embedding models
    """

    def __init__(self, device: str = 'cuda', use_fp16: bool = True):
        """
        Initialize face embedder

        Args:
            device: Device to run on ('cuda', 'cpu', 'mps')
            use_fp16: Whether to use FP16 precision (only on CUDA)
        """
        from facenet_pytorch import InceptionResnetV1

        self.device = device
        self.use_fp16 = use_fp16 and device == 'cuda'

        # Load pretrained model (VGGFace2)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.model = self.model.to(device)

        # Convert to FP16 if requested
        if self.use_fp16:
            self.model = self.model.half()
            logger.info(f"Face embedder initialized on {device} with FP16")
        else:
            logger.info(f"Face embedder initialized on {device} with FP32")

    def embed(self, face_tensors: List[torch.Tensor]) -> np.ndarray:
        """
        Generate embeddings for face tensors from MTCNN

        Args:
            face_tensors: List of face tensors [3, 160, 160] from MTCNN (already aligned and normalized)

        Returns:
            Numpy array of embeddings (N x 512)
        """
        if not face_tensors:
            return np.array([])

        # Stack tensors into batch [N, 3, 160, 160]
        batch = torch.stack(face_tensors).to(self.device)

        # Convert to FP16 if needed
        if self.use_fp16:
            batch = batch.half()

        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model(batch)

        # Convert to numpy
        embeddings = embeddings.cpu().float().numpy()

        # L2 normalize embeddings (important for cosine similarity)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings


class FaceRecognitionEngine:
    """
    Complete face recognition engine
    Combines detection and embedding for full pipeline
    """

    def __init__(self, device: Optional[str] = None, use_fp16: bool = True):
        """
        Initialize face recognition engine

        Args:
            device: Device to run on (auto-detect if None)
            use_fp16: Whether to use FP16 for embedder (only on CUDA)
        """
        if device is None:
            device = self._auto_detect_device()

        self.device = device

        # Initialize detector (always FP32)
        self.detector = FaceDetector(device=device)

        # Initialize embedder
        self.embedder = FaceEmbedder(device=device, use_fp16=use_fp16)

        logger.info(f"Face recognition engine initialized on {device}")

    def _auto_detect_device(self) -> str:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def process_image(
        self,
        image: np.ndarray,
        min_resolution: int = 100
    ) -> Dict[str, Any]:
        """
        Process image for face detection and embedding

        Args:
            image: Input image (BGR format from OpenCV)
            min_resolution: Minimum face resolution (width and height)

        Returns:
            Dictionary with:
            - num_faces: Number of faces detected
            - faces: List of face data (each with 'embedding', 'confidence', 'bbox')
            - success: Whether processing was successful
            - error: Error message if failed
        """
        try:
            # Detect + align faces. Detector now returns a 3-tuple
            # (tensors, confidences, bboxes) — the bboxes are unused
            # here (registration only needs embeddings) but the unpack
            # has to match or we get "too many values to unpack".
            detect_out = self.detector.detect(image)
            if isinstance(detect_out, tuple) and len(detect_out) == 3:
                face_tensors, confidences, _bboxes = detect_out
            else:
                # Defensive: tolerate the legacy 2-tuple shape so a
                # downgrade or third-party detector still works.
                face_tensors, confidences = detect_out

            if face_tensors is None:
                return {
                    'success': True,
                    'num_faces': 0,
                    'faces': [],
                    'error': None
                }

            # Note: min_resolution parameter is ignored since MTCNN always outputs 160x160 aligned faces
            # This is actually beneficial as it ensures consistent input to the embedding model

            # Generate embeddings
            embeddings = self.embedder.embed(face_tensors)

            # Prepare result
            faces_data = []
            for i, (embedding, conf) in enumerate(zip(embeddings, confidences)):
                faces_data.append({
                    'embedding': embedding.tolist(),
                    'confidence': conf,
                    'face_index': i
                })

            return {
                'success': True,
                'num_faces': len(faces_data),
                'faces': faces_data,
                'error': None
            }

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return {
                'success': False,
                'num_faces': 0,
                'faces': [],
                'error': str(e)
            }

    def validate_same_person(
        self,
        embeddings: List[np.ndarray],
        threshold: float = 0.6
    ) -> Tuple[bool, float]:
        """
        Check if all embeddings belong to the same person

        Args:
            embeddings: List of face embeddings
            threshold: Similarity threshold (higher = stricter)

        Returns:
            Tuple of (is_same_person, min_similarity)
        """
        if len(embeddings) < 2:
            return True, 1.0

        # Convert to numpy array
        embeddings_array = np.array(embeddings)

        # Compute pairwise cosine similarities
        similarities = np.dot(embeddings_array, embeddings_array.T)

        # Get minimum similarity (excluding diagonal)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        min_similarity = similarities[mask].min()

        is_same = min_similarity >= threshold

        return is_same, float(min_similarity)

    def compute_average_embedding(
        self,
        embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute average embedding from multiple embeddings

        Args:
            embeddings: List of face embeddings

        Returns:
            Average embedding (L2 normalized)
        """
        embeddings_array = np.array(embeddings)
        avg_embedding = embeddings_array.mean(axis=0)

        # L2 normalize
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        return avg_embedding

    def compare_embeddings(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compare two face embeddings using cosine similarity

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity (0-1, higher = more similar)
        """
        # Ensure both are numpy arrays
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Compute cosine similarity
        similarity = np.dot(emb1, emb2)

        return float(similarity)

    def cleanup(self):
        """Clean up resources"""
        try:
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        except Exception as e:
            if sys.meta_path is None:
                return
            active_logger = globals().get("logger")
            if active_logger is not None:
                active_logger.warning(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except Exception:
            pass


# Global singleton instance
_engine: Optional[FaceRecognitionEngine] = None


def get_face_engine(force_reload: bool = False) -> Optional[FaceRecognitionEngine]:
    """
    Get global face recognition engine instance

    Args:
        force_reload: Force reload the engine

    Returns:
        FaceRecognitionEngine instance or None if face recognition is disabled
    """
    global _engine

    # Check if face recognition is enabled
    enable_face_recognition = os.getenv('ENABLE_FACE_RECOGNITION', 'true').lower() == 'true'

    if not enable_face_recognition:
        logger.info("Face recognition is disabled in configuration")
        return None

    if _engine is None or force_reload:
        logger.info("Initializing global face recognition engine...")
        _engine = FaceRecognitionEngine()

    return _engine
