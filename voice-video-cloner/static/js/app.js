/**
 * Voice & Video Cloner - Frontend Application
 * Handles file uploads, drag & drop, progress polling, and UI state.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ─── Elements ─────────────────────────────────────────
    const form = document.getElementById("cloneForm");
    const submitBtn = document.getElementById("submitBtn");
    const progressSection = document.getElementById("progressSection");
    const resultSection = document.getElementById("resultSection");
    const errorSection = document.getElementById("errorSection");
    const progressBar = document.getElementById("progressBar");
    const progressPercent = document.getElementById("progressPercent");
    const progressMessage = document.getElementById("progressMessage");
    const newCloneBtn = document.getElementById("newCloneBtn");
    const retryBtn = document.getElementById("retryBtn");

    // Upload zones
    const uploadZones = document.querySelectorAll(".upload-zone");

    let currentJobId = null;
    let pollInterval = null;
    let selectedCartoonFace = null;

    // ─── Load Cartoon Faces ──────────────────────────────
    async function loadCartoonFaces() {
        try {
            const response = await fetch("/api/cartoon-faces");
            const data = await response.json();
            const grid = document.getElementById("cartoonGrid");
            grid.innerHTML = "";

            if (data.faces && data.faces.length > 0) {
                data.faces.forEach((face) => {
                    const card = document.createElement("div");
                    card.className = "cartoon-card";
                    card.dataset.filename = face.filename;
                    card.innerHTML = `
                        <img src="${face.url}" alt="${face.name}" loading="lazy">
                        <span class="cartoon-name">${face.name}</span>
                    `;
                    card.addEventListener("click", () => selectCartoonFace(card, face));
                    grid.appendChild(card);
                });
            } else {
                grid.innerHTML = '<p style="color: var(--text-muted); font-size: 13px; grid-column: 1/-1; text-align: center; padding: 20px;">No cartoon faces found. Add images to static/cartoon_faces/ folder.</p>';
            }
        } catch (err) {
            console.error("Failed to load cartoon faces:", err);
        }
    }

    function selectCartoonFace(card, face) {
        // Deselect all
        document.querySelectorAll(".cartoon-card").forEach((c) => c.classList.remove("selected"));

        // If clicking the already selected one, deselect it
        if (selectedCartoonFace === face.filename) {
            selectedCartoonFace = null;
            document.getElementById("cartoon_face").value = "";
            return;
        }

        // Select this one
        card.classList.add("selected");
        selectedCartoonFace = face.filename;
        document.getElementById("cartoon_face").value = face.filename;

        // Clear any manually uploaded target face
        const targetFaceInput = document.getElementById("target_face");
        const zone = document.querySelector('[data-input="target_face"]');
        targetFaceInput.value = "";
        zone.classList.remove("has-file");
        const uploadIcon = zone.querySelector(".upload-icon");
        const uploadText = zone.querySelector(".upload-text");
        const uploadHint = zone.querySelector(".upload-hint");
        if (uploadIcon) uploadIcon.style.display = "";
        if (uploadText) uploadText.style.display = "";
        if (uploadHint) uploadHint.style.display = "";
        const preview = zone.querySelector(".file-preview");
        if (preview) preview.style.display = "none";
    }

    // Load cartoon faces on page load
    loadCartoonFaces();

    // ─── Drag & Drop + Click Upload ──────────────────────
    uploadZones.forEach((zone) => {
        const inputId = zone.dataset.input;
        const input = document.getElementById(inputId);

        // Click to browse
        zone.addEventListener("click", (e) => {
            if (e.target.closest(".remove-btn") || e.target.closest(".file-preview")) return;
            input.click();
        });

        // Drag over
        zone.addEventListener("dragover", (e) => {
            e.preventDefault();
            zone.classList.add("drag-over");
        });

        zone.addEventListener("dragleave", () => {
            zone.classList.remove("drag-over");
        });

        // Drop
        zone.addEventListener("drop", (e) => {
            e.preventDefault();
            zone.classList.remove("drag-over");
            if (e.dataTransfer.files.length > 0) {
                input.files = e.dataTransfer.files;
                handleFileSelect(inputId, e.dataTransfer.files[0]);
            }
        });

        // File input change
        input.addEventListener("change", () => {
            if (input.files.length > 0) {
                handleFileSelect(inputId, input.files[0]);
            }
        });
    });

    // ─── Handle File Selection ───────────────────────────
    function handleFileSelect(inputId, file) {
        const zone = document.querySelector(`[data-input="${inputId}"]`);
        const uploadIcon = zone.querySelector(".upload-icon");
        const uploadText = zone.querySelector(".upload-text");
        const uploadHint = zone.querySelector(".upload-hint");

        // Hide upload prompt
        if (uploadIcon) uploadIcon.style.display = "none";
        if (uploadText) uploadText.style.display = "none";
        if (uploadHint) uploadHint.style.display = "none";

        zone.classList.add("has-file");

        // Show preview based on input type
        if (inputId === "source_video") {
            showVideoPreview("sourceVideoPreview", "sourceVideoPlayer", file);
        } else if (inputId === "target_face") {
            showFacePreview(file);
            // Deselect cartoon face when user uploads their own
            selectedCartoonFace = null;
            document.getElementById("cartoon_face").value = "";
            document.querySelectorAll(".cartoon-card").forEach((c) => c.classList.remove("selected"));
        } else if (inputId === "target_voice") {
            showVoicePreview(file);
        }
    }

    function showVideoPreview(previewId, playerId, file) {
        const preview = document.getElementById(previewId);
        const player = document.getElementById(playerId);
        const fileName = preview.querySelector(".file-name");

        player.src = URL.createObjectURL(file);
        fileName.textContent = file.name;
        preview.style.display = "block";
    }

    function showFacePreview(file) {
        const preview = document.getElementById("targetFacePreview");
        const img = document.getElementById("targetFaceImage");
        const video = document.getElementById("targetFaceVideo");
        const fileName = preview.querySelector(".file-name");

        const isVideo = file.type.startsWith("video/");
        const url = URL.createObjectURL(file);

        if (isVideo) {
            video.src = url;
            video.style.display = "block";
            img.style.display = "none";
        } else {
            img.src = url;
            img.style.display = "block";
            video.style.display = "none";
        }

        fileName.textContent = file.name;
        preview.style.display = "block";
    }

    function showVoicePreview(file) {
        const preview = document.getElementById("targetVoicePreview");
        const audio = document.getElementById("targetVoiceAudio");
        const video = document.getElementById("targetVoiceVideo");
        const fileName = preview.querySelector(".file-name");

        const isVideo = file.type.startsWith("video/");
        const url = URL.createObjectURL(file);

        if (isVideo) {
            video.src = url;
            video.style.display = "block";
            audio.style.display = "none";
        } else {
            audio.src = url;
            audio.style.display = "block";
            video.style.display = "none";
        }

        fileName.textContent = file.name;
        preview.style.display = "block";
    }

    // ─── Remove File ─────────────────────────────────────
    document.querySelectorAll(".remove-btn").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const inputId = btn.dataset.target;
            const input = document.getElementById(inputId);
            const zone = document.querySelector(`[data-input="${inputId}"]`);

            // Reset input
            input.value = "";
            zone.classList.remove("has-file");

            // Show upload prompt again
            const uploadIcon = zone.querySelector(".upload-icon");
            const uploadText = zone.querySelector(".upload-text");
            const uploadHint = zone.querySelector(".upload-hint");
            if (uploadIcon) uploadIcon.style.display = "";
            if (uploadText) uploadText.style.display = "";
            if (uploadHint) uploadHint.style.display = "";

            // Hide preview
            const preview = zone.querySelector(".file-preview");
            if (preview) preview.style.display = "none";
        });
    });

    // ─── Form Submit ─────────────────────────────────────
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Validate files
        const sourceVideo = document.getElementById("source_video");
        const targetFace = document.getElementById("target_face");
        const targetVoice = document.getElementById("target_voice");

        const cartoonFaceValue = document.getElementById("cartoon_face").value;

        if (!sourceVideo.files.length) {
            alert("Please upload your source video.");
            return;
        }
        if (!targetFace.files.length && !cartoonFaceValue) {
            alert("Please select a cartoon character or upload a target face image.");
            return;
        }
        if (!targetVoice.files.length) {
            alert("Please upload target persona voice audio or video.");
            return;
        }

        // Build form data
        const formData = new FormData();
        formData.append("source_video", sourceVideo.files[0]);
        if (targetFace.files.length) {
            formData.append("target_face", targetFace.files[0]);
        }
        if (cartoonFaceValue) {
            formData.append("cartoon_face", cartoonFaceValue);
        }
        formData.append("target_voice", targetVoice.files[0]);
        formData.append("language", document.getElementById("language").value);
        formData.append("voice_name", document.getElementById("voice_name").value);

        // Show progress, hide form
        form.style.display = "none";
        resultSection.style.display = "none";
        errorSection.style.display = "none";
        progressSection.style.display = "block";
        resetProgress();

        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner"></span> Uploading...';

        try {
            const response = await fetch("/api/clone", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Upload failed.");
            }

            currentJobId = data.job_id;
            submitBtn.innerHTML = '<span class="spinner"></span> Processing...';

            // Start polling
            startPolling(currentJobId);
        } catch (err) {
            showError(err.message);
        }
    });

    // ─── Progress Polling ────────────────────────────────
    function startPolling(jobId) {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/status/${jobId}`);
                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || "Failed to get status.");
                }

                updateProgress(data);

                if (data.status === "complete") {
                    clearInterval(pollInterval);
                    showResult(jobId);
                } else if (data.status === "error") {
                    clearInterval(pollInterval);
                    showError(data.message);
                }
            } catch (err) {
                clearInterval(pollInterval);
                showError(err.message);
            }
        }, 1500);
    }

    function updateProgress(data) {
        const percent = data.progress || 0;
        progressBar.style.width = `${percent}%`;
        progressPercent.textContent = `${percent}%`;
        progressMessage.textContent = data.message || "Processing...";

        // Update stage indicators
        const stageOrder = ["extract_audio", "voice_clone", "face_swap", "combine"];
        const currentStage = data.stage;
        const currentIndex = stageOrder.indexOf(currentStage);

        stageOrder.forEach((stage, i) => {
            const el = document.getElementById(`stage-${stage}`);
            if (!el) return;

            el.classList.remove("active", "complete");
            if (i < currentIndex) {
                el.classList.add("complete");
            } else if (i === currentIndex) {
                el.classList.add("active");
                if (data.progress === 100 && data.status === "processing") {
                    el.classList.add("complete");
                }
            }
        });

        // Overall progress estimation
        const stageWeights = { extract_audio: 10, voice_clone: 35, face_swap: 45, combine: 10 };
        let overallProgress = 0;
        stageOrder.forEach((stage, i) => {
            if (i < currentIndex) {
                overallProgress += stageWeights[stage];
            } else if (i === currentIndex) {
                overallProgress += (stageWeights[stage] * percent) / 100;
            }
        });
        progressBar.style.width = `${Math.round(overallProgress)}%`;
        progressPercent.textContent = `${Math.round(overallProgress)}%`;
    }

    function resetProgress() {
        progressBar.style.width = "0%";
        progressPercent.textContent = "0%";
        progressMessage.textContent = "Initializing...";
        document.querySelectorAll(".stage").forEach((el) => {
            el.classList.remove("active", "complete");
        });
    }

    // ─── Show Result ─────────────────────────────────────
    function showResult(jobId) {
        progressSection.style.display = "none";
        resultSection.style.display = "block";

        const resultVideo = document.getElementById("resultVideo");
        const downloadBtn = document.getElementById("downloadBtn");

        resultVideo.src = `/api/download/${jobId}`;
        downloadBtn.href = `/api/download/${jobId}`;

        submitBtn.disabled = false;
        submitBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
            </svg>
            Start Cloning
        `;
    }

    // ─── Show Error ──────────────────────────────────────
    function showError(message) {
        progressSection.style.display = "none";
        resultSection.style.display = "none";
        errorSection.style.display = "block";
        document.getElementById("errorMessage").textContent = message;

        submitBtn.disabled = false;
        submitBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
            </svg>
            Start Cloning
        `;
    }

    // ─── Reset / New Clone ───────────────────────────────
    function resetUI() {
        form.style.display = "";
        progressSection.style.display = "none";
        resultSection.style.display = "none";
        errorSection.style.display = "none";
        resetProgress();

        // Reset file inputs
        ["source_video", "target_face", "target_voice"].forEach((id) => {
            const input = document.getElementById(id);
            const zone = document.querySelector(`[data-input="${id}"]`);
            input.value = "";
            zone.classList.remove("has-file");

            const uploadIcon = zone.querySelector(".upload-icon");
            const uploadText = zone.querySelector(".upload-text");
            const uploadHint = zone.querySelector(".upload-hint");
            if (uploadIcon) uploadIcon.style.display = "";
            if (uploadText) uploadText.style.display = "";
            if (uploadHint) uploadHint.style.display = "";

            const preview = zone.querySelector(".file-preview");
            if (preview) preview.style.display = "none";
        });

        // Reset cartoon face selection
        selectedCartoonFace = null;
        document.getElementById("cartoon_face").value = "";
        document.querySelectorAll(".cartoon-card").forEach((c) => c.classList.remove("selected"));
    }

    newCloneBtn.addEventListener("click", resetUI);
    retryBtn.addEventListener("click", resetUI);
});
