# streamlit_app.py
import streamlit as st
import requests
from PIL import Image, ImageChops, ImageOps, ImageStat
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -----------------------
# Configuration
# -----------------------
API_BASE = "http://127.0.0.1:8000"   # adjust if your FastAPI runs elsewhere

st.set_page_config(page_title="QAMIS Dashboard", layout="wide")

st.title("QAMIS — Authenticity Dashboard")

# Session state for last ingested id and last JSON results
if "last_id" not in st.session_state:
    st.session_state.last_id = ""
if "last_ingest_resp" not in st.session_state:
    st.session_state.last_ingest_resp = None
# store known_id input in session state to allow programmatic updates
if "known_id_input" not in st.session_state:
    st.session_state.known_id_input = st.session_state.last_id

# -----------------------
# Sidebar: controls
# -----------------------
st.sidebar.header("Actions")

with st.sidebar.expander("Ingest (store original)"):
    ingest_file = st.file_uploader("Choose image to ingest", type=["png", "jpg", "jpeg"])
    owner = st.text_input("Owner (optional)", value="me")
    if st.button("Ingest file"):
        if ingest_file is None:
            st.sidebar.warning("Choose a file first.")
        else:
            files = {"file": (ingest_file.name, ingest_file.getvalue())}
            data = {"owner": owner}
            try:
                r = requests.post(f"{API_BASE}/ingest", files=files, data=data, timeout=30)
                r.raise_for_status()
                resp = r.json()
                st.session_state.last_ingest_resp = resp
                st.session_state.last_id = resp.get("id", "")
                # update known_id_input default if empty
                if not st.session_state.known_id_input:
                    st.session_state.known_id_input = st.session_state.last_id
                st.sidebar.success("Ingested ✓")
                st.sidebar.json(resp)
            except Exception as e:
                st.sidebar.error(f"Ingest failed: {e}")

st.sidebar.markdown("---")
st.sidebar.header("Quick IDs")

# Try fetch recent IDs from backend to let user pick quickly
try:
    r = requests.get(f"{API_BASE}/ids", timeout=5)
    if r.status_code == 200:
        data = r.json()
        recent_ids = data.get("ids", [])
        if recent_ids:
            st.sidebar.write("Recent ingested IDs (click to use):")
            # create a button per id (compact label). clicking sets known_id_input in session_state
            for rid in recent_ids:
                label = rid[:12] + "…" if len(rid) > 12 else rid
                # create unique key for each button so Streamlit doesn't reuse
                btn_key = f"pick_{rid}"
                if st.sidebar.button(label, key=btn_key):
                    st.session_state.known_id_input = rid
        else:
            st.sidebar.write("No ids available yet.")
    else:
        st.sidebar.write("Could not fetch ids (status {})".format(r.status_code))
except Exception:
    st.sidebar.write("Could not contact API /ids")

st.sidebar.write("Last ingested id:")
st.sidebar.code(st.session_state.last_id or "No id yet")
st.sidebar.markdown("---")

# Add a slider to tune difference threshold
st.sidebar.header("Diff sensitivity")
thresh = st.sidebar.slider(
    "Diff threshold (pixel intensity)",
    min_value=1, max_value=80, value=15,
    help="Lower = more sensitive (detect more small changes)."
)

# Add display width control (optional)
img_display_width = st.sidebar.selectbox("Image display width", options=[320, 420, 512, 640], index=1)

st.sidebar.markdown("---")
st.sidebar.caption("QAMIS Dashboard — tweak threshold and choose a recent ID.")

# -----------------------
# Main layout
# -----------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload / Analyze")
    uploaded_file = st.file_uploader("Choose image to analyze/verify", key="analyze_upload", type=["png", "jpg", "jpeg"])
    # text input bound to session state so sidebar buttons can set it
    known_id_input = st.text_input("known_id (paste or use quick IDs)", key="known_id_input", value=st.session_state.known_id_input)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Analyze image"):
            if uploaded_file is None:
                st.warning("Select an image to analyze.")
            else:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                params = {"known_id": known_id_input} if known_id_input else {}
                try:
                    r = requests.post(f"{API_BASE}/analyze", params=params, files=files, timeout=30)
                    r.raise_for_status()
                    resp = r.json()
                    st.session_state.analyze_resp = resp
                    st.success("Analyze completed")
                except Exception as e:
                    st.error(f"Analyze failed: {e}")

    with c2:
        if st.button("Verify image"):
            if uploaded_file is None:
                st.warning("Select an image to verify.")
            else:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                params = {"known_id": known_id_input} if known_id_input else {}
                try:
                    r = requests.post(f"{API_BASE}/verify", params=params, files=files, timeout=30)
                    r.raise_for_status()
                    resp = r.json()
                    st.session_state.verify_resp = resp
                    st.success("Verify completed")
                except Exception as e:
                    st.error(f"Verify failed: {e}")

    with c3:
        if st.button("Show last analyze result"):
            if "analyze_resp" in st.session_state:
                st.json(st.session_state.analyze_resp)
            else:
                st.info("No analyze run yet.")

with col2:
    st.subheader("Visual Comparison / Results")

    if uploaded_file is None:
        st.info("Upload an image on the left to begin.")
        uploaded_img = None
    else:
        # show uploaded image
        img_bytes = uploaded_file.getvalue()
        try:
            uploaded_img = Image.open(BytesIO(img_bytes)).convert("RGB")
            st.image(uploaded_img, caption="Uploaded image", width=img_display_width)
        except Exception as e:
            uploaded_img = None
            st.error(f"Could not open uploaded image: {e}")

    # show JSON outputs if present
    if "analyze_resp" in st.session_state:
        st.markdown("### Analyze Result")
        resp = st.session_state.analyze_resp
        st.json(resp)

        # inform about comparison
        stored_id = resp.get("phash_stored") and known_id_input
        if stored_id and "phash_stored" in resp:
            st.markdown("#### Comparison (uploaded vs original stored in your DB)")
            st.info("If you want the original image preview here, ensure GET /storage/{id} is available in the API.")

    if "verify_resp" in st.session_state:
        st.markdown("### Verify Result")
        st.json(st.session_state.verify_resp)

    # --- START: Stored original fetch + visual comparison ---
    # If analyze response exists we may have phash_stored and known_id_input
    if "analyze_resp" in st.session_state:
        resp = st.session_state.analyze_resp
    else:
        resp = None

    if uploaded_file is not None and uploaded_img is not None:
        # Show uploaded image already handled above.
        # Now try to fetch stored original from backend if known_id provided
        if known_id_input:
            try:
                r = requests.get(f"{API_BASE}/storage/{known_id_input}", timeout=10)
                if r.status_code == 200:
                    orig_bytes = r.content
                    orig_img = Image.open(BytesIO(orig_bytes)).convert("RGB")
                    # Resize both images to same preview size for display and diff
                    preview_size = (512, 512)
                    uploaded_preview = uploaded_img.copy().resize(preview_size)
                    orig_preview = orig_img.copy().resize(preview_size)

                    st.markdown("### Stored original vs Uploaded")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(orig_preview, caption="Stored original", width=img_display_width)
                    with c2:
                        st.image(uploaded_preview, caption="Uploaded (analyzed)", width=img_display_width)

                    # Compute difference image (grayscale) and show heatmap
                    diff = ImageChops.difference(ImageOps.grayscale(orig_preview), ImageOps.grayscale(uploaded_preview))
                    diff_arr = np.asarray(diff, dtype=np.float32)

                    # Amplify differences for visibility
                    diff_vis = np.clip(diff_arr * 6.0, 0, 255).astype(np.uint8)

                    fig, ax = plt.subplots(figsize=(6,3))
                    ax.imshow(diff_vis, cmap="inferno")
                    ax.axis("off")
                    st.markdown("#### Difference heatmap (amplified)")
                    st.pyplot(fig)

                    # Simple numeric diff metric: percent of pixels that changed above the slider threshold
                    changed = int((diff_arr > thresh).sum())
                    total = int(diff_arr.size)
                    percent_changed = 100.0 * changed / total if total > 0 else 0.0
                    st.markdown(f"**Changed pixels:** {changed} / {total} ({percent_changed:.2f}%)")

                    # Show basic image stats (rms) to complement noise_metric
                    stat = ImageStat.Stat(diff)
                    rms = float(np.sqrt(np.mean(np.array(stat.rms) ** 2)))
                    st.markdown(f"**RMS difference:** {rms:.2f}")

                    # If backend produced authenticity_score in analyze_resp, show a gauge
                    if resp and "authenticity_score" in resp:
                        score_val = resp["authenticity_score"]
                        st.markdown("### Authenticity Score")
                        gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score_val,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={'axis': {'range': [0, 100]},
                                   'bar': {'color': "darkgreen" if score_val >= 70 else "orange" if score_val >= 40 else "red"},
                                   'threshold': {'line': {'color': "black", 'width': 4}, 'value': 70}}
                        ))
                        gauge.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
                        st.plotly_chart(gauge, use_container_width=True)

                else:
                    st.info("Stored original not found on server (GET /storage/<id> returned status " + str(r.status_code) + ").")
            except Exception as e:
                st.info("Could not fetch stored original: " + str(e))
        else:
            st.info("Provide a known_id (stored id) to fetch the original for comparison.")
    # --- END: Stored original fetch + visual comparison ---

# -----------------------
# Lower area: Image diff heatmap for quick visual diff (optional)
# -----------------------
st.markdown("---")
st.subheader("Image difference (visual)")

if uploaded_file is not None and st.session_state.last_ingest_resp:
    try:
        # attempt to open the stored image file path from the ingest response using API local file path
        # We assume ingest saved the file in server storage and returned the id; because Streamlit runs locally,
        # we can try to read the storage file directly if the dashboard runs on the same machine.
        # For the demo, show uploaded vs itself if the actual stored original is not available.
        uploaded = Image.open(BytesIO(uploaded_file.getvalue())).convert("L").resize((512,512))
        orig = Image.open(BytesIO(uploaded_file.getvalue())).convert("L").resize((512,512))
        diff = ImageChops.difference(orig, uploaded)
        diff_arr = np.asarray(diff, dtype=np.float32)
        # amplify differences
        diff_vis = np.clip(diff_arr * 4.0, 0, 255).astype(np.uint8)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.imshow(diff_vis, cmap="inferno")
        ax.axis("off")
        st.pyplot(fig)
    except Exception as e:
        st.info("Diff could not be computed: " + str(e))
else:
    st.write("Upload an image and ingest one first to use visual diff here.")

st.markdown("---")
st.caption("QAMIS dashboard — calls local API endpoints at " + API_BASE)
