import os
import hashlib
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import imagehash
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
from tinydb import TinyDB, Query
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
import numpy as np
from scipy.ndimage import laplace
from imagehash import hex_to_hash
from fastapi.responses import StreamingResponse
from fastapi import Query
# ensure values are native Python types before JSON encoding
import numpy as np

def as_native(x):
    """Convert numpy scalars/arrays to native Python types where appropriate."""
    if x is None:
        return None
    # numpy scalar floats and ints
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    # numpy arrays - convert to list
    if isinstance(x, np.ndarray):
        return x.tolist()
    # fallback: leave as-is
    return x
def compute_authenticity_score(sig_ok: bool, forensic_confidence: float, phash_distance_val, ssim_val):
    """
    Combine signals into a final authenticity score (0..100).
    - sig_ok: bool (True if stored signature verifies the stored id)
    - forensic_confidence: float 0..1 (higher => more suspicious)
    - phash_distance_val: int or None
    - ssim_val: float 0..1
    Returns: (score_int_0_100, combined_float_0_1)
    """
    # normalize signature
    sig_score = 1.0 if sig_ok else 0.0

    # forensic_inv: 1.0 for no suspicion, 0.0 for max suspicion
    forensic_inv = 1.0 - float(forensic_confidence or 0.0)

    # phash_score mapping (tunable)
    pd = phash_distance_val
    if pd is None:
        phash_score = 0.5
    elif pd <= 2:
        phash_score = 1.0
    elif pd <= 6:
        phash_score = 0.8
    elif pd <= 12:
        phash_score = 0.5
    else:
        phash_score = 0.0

    # ssim fallback
    ssim_score = float(ssim_val or 0.0)

    # weights
    w_sig = 0.40
    w_for = 0.35
    w_ph  = 0.15
    w_ss  = 0.10

    combined = (w_sig * sig_score) + (w_for * forensic_inv) + (w_ph * phash_score) + (w_ss * ssim_score)
    combined = max(0.0, min(1.0, combined))
    return int(round(combined * 100)), combined



# ----------------------------------------------------------
# Folders and DB setup
# ----------------------------------------------------------
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

DB = TinyDB("metadata.json")

KEY_DIR = Path("keys")
KEY_DIR.mkdir(exist_ok=True)

PRIVATE_KEY_PATH = KEY_DIR / "ed25519_private.pem"
PUBLIC_KEY_PATH = KEY_DIR / "ed25519_public.pem"

# ----------------------------------------------------------
# Key Loading / Creation
# ----------------------------------------------------------
def load_or_create_keypair():
    if PRIVATE_KEY_PATH.exists() and PUBLIC_KEY_PATH.exists():
        # Load from disk
        with open(PRIVATE_KEY_PATH, "rb") as f:
            sk = serialization.load_pem_private_key(f.read(), password=None)
        with open(PUBLIC_KEY_PATH, "rb") as f:
            pk = serialization.load_pem_public_key(f.read())
    else:
        # Create fresh keys
        sk = Ed25519PrivateKey.generate()
        pk = sk.public_key()

        with open(PRIVATE_KEY_PATH, "wb") as f:
            f.write(sk.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        with open(PUBLIC_KEY_PATH, "wb") as f:
            f.write(pk.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))

    return sk, pk

SK, PK = load_or_create_keypair()

# ----------------------------------------------------------
# Utility: SHA-256
# ----------------------------------------------------------
def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ----------------------------------------------------------
# Utility: Perceptual Hash (image tamper check)
# ----------------------------------------------------------
def compute_phash(data: bytes):
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
        return str(imagehash.phash(img))
    except:
        return None

# ----------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------
app = FastAPI(title="QAMIS - Phase 1")

# ----------------------------------------------------------
# /ingest endpoint
# ----------------------------------------------------------
@app.post("/ingest")
async def ingest(file: UploadFile = File(...), owner: str = "unknown"):
    content = await file.read()

    if not content:
        raise HTTPException(400, "Empty file")

    sha = file_sha256(content)
    phash = compute_phash(content)

    saved_filename = f"{sha}_{file.filename}"
    saved_path = STORAGE_DIR / saved_filename

    # Save file
    with open(saved_path, "wb") as f:
        f.write(content)

    # Sign SHA-256 using Ed25519 private key
    signature = SK.sign(bytes.fromhex(sha)).hex()

    # Store metadata
    DB.insert({
        "id": sha,
        "filename": file.filename,
        "saved_name": saved_filename,
        "owner": owner,
        "phash": phash,
        "signature": signature
    })

    return {
        "message": "file ingested",
        "id": sha,
        "phash": phash,
        "signature": signature
    }

# ----------------------------------------------------------
# /verify endpoint
# ----------------------------------------------------------
@app.post("/verify")
async def verify(file: UploadFile = File(...), known_id: str = None):
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    sha = file_sha256(content)
    phash_uploaded = compute_phash(content)

    Record = Query()

    # Case 1: verify against known ID
    if known_id:
        result = DB.search(Record.id == known_id)
        if not result:
            return {"verified": False, "reason": "ID not found"}

        entry = result[0]

        signature_hex = entry.get("signature")
        # Verify signature (signature is over stored id at ingest time)
        signature_ok = False
        try:
            if signature_hex:
                PK.verify(bytes.fromhex(signature_hex), bytes.fromhex(entry["id"]))
                signature_ok = True
        except Exception:
            signature_ok = False

        phash_stored = entry.get("phash")
        phash_match = (phash_uploaded == phash_stored) if phash_uploaded and phash_stored else None

        if sha == entry["id"]:
            decision = "Authentic (exact match)"
        elif signature_ok and phash_match:
            decision = "Probably authentic"
        elif signature_ok and phash_match is False:
            decision = "Suspicious (image changed)"
        else:
            decision = "Tampered or invalid"

        # compute phash_distance if possible
        try:
            phash_distance_val = None
            if phash_uploaded and phash_stored:
                phash_distance_val = phash_distance(phash_stored, phash_uploaded)
        except Exception:
            phash_distance_val = None

        # compute SSIM by loading the stored file if present
        ssim_val = None
        try:
            stored_path = Path("storage") / entry["saved_name"]
            if stored_path.exists():
                with open(stored_path, "rb") as f:
                    stored_bytes = f.read()
                ssim_val = compute_ssim_from_bytes(stored_bytes, content)
        except Exception:
            ssim_val = None

        # heuristic forensic_confidence for verify: exact byte match -> 0, otherwise 0.5
        forensic_confidence_val = 0.0 if sha == entry["id"] else 0.5

        # compute final authenticity score
        auth_score_int, auth_combined = compute_authenticity_score(signature_ok, forensic_confidence_val, phash_distance_val, ssim_val)

        # ensure native types for JSON
        phash_distance_val = as_native(phash_distance_val)
        ssim_val = as_native(ssim_val)
        forensic_confidence_val = as_native(forensic_confidence_val)

        return {
            "uploaded_sha256": sha,
            "signature_ok": signature_ok,
            "phash_uploaded": phash_uploaded,
            "phash_stored": phash_stored,
            "phash_match": phash_match,
            "phash_distance": phash_distance_val,
            "ssim": ssim_val,
            "decision": decision,
            "authenticity_score": auth_score_int,
            "authenticity_combined": round(auth_combined, 3)
        }

    # Case 2: no known_id provided
    match = DB.search(Record.id == sha)
    if match:
        return {"verified": True, "result": "Exact match found"}

    # No known_id and no exact match: return basic info + a lightweight authenticity estimate
    # forensic_confidence default 0.0 (no comparison) -> authenticity score will be modest
    forensic_confidence_val = 0.0
    auth_score_int, auth_combined = compute_authenticity_score(False, forensic_confidence_val, None, None)

    return {
        "verified": False,
        "uploaded_sha256": sha,
        "phash_uploaded": phash_uploaded,
        "authenticity_score": auth_score_int,
        "authenticity_combined": round(auth_combined, 3)
    }

# ---- Phase 2: lightweight forensic checks ----

def phash_distance(h1: str, h2: str) -> int:
    """Return Hamming distance between two perceptual hashes (strings)."""
    try:
        return hex_to_hash(h1) - hex_to_hash(h2)
    except Exception:
        return None

def compute_ssim_from_bytes(a_bytes: bytes, b_bytes: bytes):
    """
    Robust SSIM: always returns a float between 0.0 and 1.0.
    Resizes both images to the same size (min width/min height) before computing SSIM.
    """
    try:
        from io import BytesIO
        a_img = Image.open(BytesIO(a_bytes)).convert("L")
        b_img = Image.open(BytesIO(b_bytes)).convert("L")

        # choose common size (min width/min height) to avoid distortions
        size = (min(a_img.width, b_img.width), min(a_img.height, b_img.height))
        if size[0] <= 0 or size[1] <= 0:
            return 0.0

        a_resized = a_img.resize(size)
        b_resized = b_img.resize(size)

        a_arr = np.asarray(a_resized, dtype=np.float32)
        b_arr = np.asarray(b_resized, dtype=np.float32)

        # compute SSIM; ensure a valid data_range for float images
        # skimage ssim expects images with same dtype and range; for floats assume 0..255
        score, _ = ssim(a_arr, b_arr, full=True, data_range=255.0)
        # clamp and return Python float
        return float(max(0.0, min(1.0, score)))
    except Exception:
        # never return None; return 0.0 if anything goes wrong
        return 0.0

def noise_metric_from_bytes(data: bytes):
    """
    Simple noise / tamper indicator: variance of Laplacian (high-pass).
    Lower values => smooth / possibly heavily filtered; higher => noisy.
    Return normalized float (0..1) using a practical scale.
    """
    try:
        from io import BytesIO
        img = Image.open(BytesIO(data)).convert("L").resize((512,512))
        arr = np.array(img, dtype=np.float32)
        lap = laplace(arr)
        v = float(np.var(lap))
        # Normalize using a heuristic scale (tweakable)
        # Typical natural images: variance ~ [5, 200]; map to 0..1.
        norm = (v - 5.0) / (200.0 - 5.0)
        return float(max(0.0, min(1.0, norm)))
    except Exception:
        return None

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), known_id: str = None):
    """
    Robust forensic analysis with final authenticity score.
    Returns forensic signals plus:
      - signature_ok (bool)
      - authenticity_score (0..100 int)
      - authenticity_combined (0..1 float)
    """
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")

    uploaded_phash = compute_phash(content)
    uploaded_sha = file_sha256(content)

    # Baseline signals
    s = {
        "uploaded_sha256": uploaded_sha,
        "phash_uploaded": uploaded_phash,
        "phash_distance": None,
        "ssim": None,
        "noise_metric": None,
        "forensic_confidence": None,
        "explanation": None
    }

    # compute noise metric
    s["noise_metric"] = noise_metric_from_bytes(content)

    # If known_id provided, compare with stored file
    if known_id:
        Record = Query()
        res = DB.search(Record.id == known_id)
        if not res:
            s["explanation"] = "known_id not found in database"
            s["forensic_confidence"] = 0.5
            # ensure native types before return
            for k in ("phash_distance", "ssim", "noise_metric", "forensic_confidence"):
                s[k] = as_native(s[k])
            s["signature_ok"] = False
            s["authenticity_score"], s["authenticity_combined"] = compute_authenticity_score(False, s["forensic_confidence"], s["phash_distance"], s["ssim"])
            s["authenticity_combined"] = round(s["authenticity_combined"], 3)
            return s

        entry = res[0]
        stored_path = Path("storage") / entry["saved_name"]
        if not stored_path.exists():
            s["explanation"] = "stored file missing"
            s["forensic_confidence"] = 0.6
            for k in ("phash_distance", "ssim", "noise_metric", "forensic_confidence"):
                s[k] = as_native(s[k])
            s["signature_ok"] = False
            s["authenticity_score"], s["authenticity_combined"] = compute_authenticity_score(False, s["forensic_confidence"], s["phash_distance"], s["ssim"])
            s["authenticity_combined"] = round(s["authenticity_combined"], 3)
            return s

        # load stored file bytes
        with open(stored_path, "rb") as f:
            stored_bytes = f.read()

        s["phash_stored"] = entry.get("phash")

        # phash distance
        if s["phash_stored"] and s["phash_uploaded"]:
            try:
                s["phash_distance"] = phash_distance(s["phash_stored"], s["phash_uploaded"])
            except Exception:
                s["phash_distance"] = None

        # ssim (robust)
        s["ssim"] = compute_ssim_from_bytes(stored_bytes, content)

        # Decide forensic_confidence (simple rule-based score)
        score = 0.0
        reasons = []

        if s["phash_distance"] is not None:
            if s["phash_distance"] > 12:
                score += 0.6
                reasons.append(f"Large pHash distance ({s['phash_distance']})")
            elif s["phash_distance"] > 6:
                score += 0.25
                reasons.append(f"Moderate pHash distance ({s['phash_distance']})")
            else:
                reasons.append(f"Small pHash distance ({s['phash_distance']})")

        if s["ssim"] is not None:
            if s["ssim"] < 0.90:
                score += 0.5
                reasons.append(f"Low SSIM ({s['ssim']:.3f})")
            elif s["ssim"] < 0.98:
                score += 0.15
                reasons.append(f"Moderate SSIM ({s['ssim']:.3f})")
            else:
                reasons.append(f"High SSIM ({s['ssim']:.3f})")

        if s["noise_metric"] is not None:
            try:
                stored_noise = noise_metric_from_bytes(stored_bytes)
                if stored_noise is not None:
                    delta = abs((s["noise_metric"] or 0.0) - stored_noise)
                    if delta > 0.25:
                        score += 0.25
                        reasons.append(f"Noise level changed (delta {delta:.2f})")
                    else:
                        reasons.append(f"Noise similar (delta {delta:.2f})")
            except Exception:
                pass

        final = max(0.0, min(1.0, score))
        s["forensic_confidence"] = round(final, 3)
        s["explanation"] = "; ".join(reasons) if reasons else "No strong indicators"

        # signature check (signature stored at ingest time)
        sig_ok = False
        try:
            sig_hex = entry.get("signature")
            if sig_hex:
                PK.verify(bytes.fromhex(sig_hex), bytes.fromhex(entry["id"]))
                sig_ok = True
        except Exception:
            sig_ok = False

        # compute final authenticity score
        auth_score_int, auth_combined = compute_authenticity_score(
            sig_ok,
            s["forensic_confidence"],
            s.get("phash_distance"),
            s.get("ssim")
        )

        # convert all numeric values to native Python types
        for k in ("phash_distance", "ssim", "noise_metric", "forensic_confidence"):
            s[k] = as_native(s[k])

        s["signature_ok"] = sig_ok
        s["authenticity_score"] = auth_score_int
        s["authenticity_combined"] = round(auth_combined, 3)

        return s

    # If no known_id provided, just return local signals and a lightweight score
    s["forensic_confidence"] = 0.0
    s["explanation"] = "No comparison file provided; supply known_id to compare with stored original."

    # compute lightweight authenticity score (no signature)
    auth_score_int, auth_combined = compute_authenticity_score(
        False,
        s["forensic_confidence"],
        s.get("phash_distance"),
        s.get("ssim")
    )

    # convert numeric types
    for k in ("phash_distance", "ssim", "noise_metric", "forensic_confidence"):
        s[k] = as_native(s[k])

    s["signature_ok"] = False
    s["authenticity_score"] = auth_score_int
    s["authenticity_combined"] = round(auth_combined, 3)

    return s
@app.get("/storage/{sha}")
def get_stored(sha: str):
    """
    Return the stored file bytes for a previously ingested file id (sha).
    Useful for dashboards that want to preview the original image.
    """
    Record = Query()
    res = DB.search(Record.id == sha)
    if not res:
        raise HTTPException(404, "ID not found")
    entry = res[0]
    path = Path("storage") / entry["saved_name"]
    if not path.exists():
        raise HTTPException(404, "stored file missing")
    f = open(path, "rb")
    # try infer content type by extension (basic)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif ext == ".png":
        media_type = "image/png"
    else:
        media_type = "application/octet-stream"
    return StreamingResponse(f, media_type=media_type)
@app.get("/ids")
def list_ids(limit: int = 50):
    """
    Return the most recent `limit` ingested IDs (most recent last).
    Useful for dashboards to present quick selection of known ids.
    """
    all_records = DB.all()
    # return last 'limit' ids (newest last)
    recent = all_records[-limit:]
    # present newest first
    recent_ids = [r["id"] for r in reversed(recent)]
    return {"count": len(recent_ids), "ids": recent_ids}
