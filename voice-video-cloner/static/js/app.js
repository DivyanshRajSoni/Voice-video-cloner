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
    let bgEnabled = false;
    let bgMode = "prompt"; // "prompt" or "upload"

    // ─── Load Cartoon Faces ──────────────────────────────
    async function loadCartoonFaces() {
        try {
            // Load both human and cartoon faces
            const [humanRes, cartoonRes] = await Promise.all([
                fetch("/api/human-faces"),
                fetch("/api/cartoon-faces")
            ]);
            const humanData = await humanRes.json();
            const cartoonData = await cartoonRes.json();

            // Populate human grid
            const humanGrid = document.getElementById("humanGrid");
            humanGrid.innerHTML = "";
            if (humanData.faces && humanData.faces.length > 0) {
                humanData.faces.forEach((face) => {
                    const card = document.createElement("div");
                    card.className = "cartoon-card";
                    card.dataset.filename = face.filename;
                    card.innerHTML = `
                        <img src="${face.url}" alt="${face.name}" loading="lazy">
                        <span class="cartoon-name">${face.name}</span>
                    `;
                    card.addEventListener("click", () => selectCartoonFace(card, face));
                    humanGrid.appendChild(card);
                });
            }

            // Populate cartoon grid
            const grid = document.getElementById("cartoonGrid");
            grid.innerHTML = "";

            if (cartoonData.faces && cartoonData.faces.length > 0) {
                cartoonData.faces.forEach((face) => {
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
            }

            if ((!humanData.faces || humanData.faces.length === 0) && (!cartoonData.faces || cartoonData.faces.length === 0)) {
                grid.innerHTML = '<p style="color: var(--text-muted); font-size: 13px; grid-column: 1/-1; text-align: center; padding: 20px;">No preset faces found.</p>';
            }
        } catch (err) {
            console.error("Failed to load faces:", err);
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

    // ─── Background Change UI ────────────────────────────
    const bgToggle = document.getElementById("bgToggle");
    const bgOptions = document.getElementById("bgOptions");
    const bgTabs = document.querySelectorAll(".bg-tab");
    const bgPromptMode = document.getElementById("bgPromptMode");
    const bgUploadMode = document.getElementById("bgUploadMode");
    const bgPromptInput = document.getElementById("bgPromptInput");
    const bgPreviewBtn = document.getElementById("bgPreviewBtn");
    const bgPreviewContainer = document.getElementById("bgPreviewContainer");
    const bgPreviewImg = document.getElementById("bgPreviewImg");
    const bgStage = document.getElementById("stage-bg_change");

    // Toggle background feature
    if (bgToggle) {
        bgToggle.addEventListener("change", () => {
            bgEnabled = bgToggle.checked;
            bgOptions.style.display = bgEnabled ? "block" : "none";
            if (bgStage) bgStage.style.display = bgEnabled ? "" : "none";
            // Update progress grid columns
            const stagesGrid = document.querySelector(".progress-stages");
            if (stagesGrid) {
                stagesGrid.style.gridTemplateColumns = bgEnabled ? "repeat(5, 1fr)" : "repeat(4, 1fr)";
            }
        });
    }

    // Tab switching
    bgTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            bgTabs.forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            bgMode = tab.dataset.mode;
            bgPromptMode.style.display = bgMode === "prompt" ? "block" : "none";
            bgUploadMode.style.display = bgMode === "upload" ? "block" : "none";
        });
    });

    // Preset prompt buttons
    document.querySelectorAll(".bg-preset-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".bg-preset-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            bgPromptInput.value = btn.dataset.prompt;
        });
    });

    // Preview generation
    if (bgPreviewBtn) {
        bgPreviewBtn.addEventListener("click", async () => {
            const prompt = bgPromptInput.value.trim();
            if (!prompt) {
                alert("Enter a background description first.");
                return;
            }
            bgPreviewBtn.disabled = true;
            bgPreviewBtn.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span>';
            bgPreviewContainer.style.display = "none";

            try {
                const resp = await fetch("/api/generate-bg", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt }),
                });
                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.error || "Generation failed");
                }
                const blob = await resp.blob();
                bgPreviewImg.src = URL.createObjectURL(blob);
                bgPreviewContainer.style.display = "block";
            } catch (err) {
                alert("Background generation failed: " + err.message);
            } finally {
                bgPreviewBtn.disabled = false;
                bgPreviewBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
            }
        });
    }

    // Background image upload zone
    const bgUploadZone = document.getElementById("bgUploadZone");
    const bgImageInput = document.getElementById("bg_image");
    if (bgUploadZone && bgImageInput) {
        bgUploadZone.addEventListener("click", (e) => {
            if (e.target.closest(".remove-btn") || e.target.closest(".file-preview")) return;
            bgImageInput.click();
        });
        bgUploadZone.addEventListener("dragover", (e) => { e.preventDefault(); bgUploadZone.classList.add("drag-over"); });
        bgUploadZone.addEventListener("dragleave", () => { bgUploadZone.classList.remove("drag-over"); });
        bgUploadZone.addEventListener("drop", (e) => {
            e.preventDefault();
            bgUploadZone.classList.remove("drag-over");
            if (e.dataTransfer.files.length > 0) {
                bgImageInput.files = e.dataTransfer.files;
                showBgImagePreview(e.dataTransfer.files[0]);
            }
        });
        bgImageInput.addEventListener("change", () => {
            if (bgImageInput.files.length > 0) showBgImagePreview(bgImageInput.files[0]);
        });
    }

    function showBgImagePreview(file) {
        const zone = document.getElementById("bgUploadZone");
        const preview = document.getElementById("bgImagePreview");
        const img = document.getElementById("bgImagePreviewImg");
        const fileName = preview.querySelector(".file-name");

        zone.classList.add("has-file");
        zone.querySelector(".upload-icon").style.display = "none";
        zone.querySelector(".upload-text").style.display = "none";
        zone.querySelector(".upload-hint").style.display = "none";

        img.src = URL.createObjectURL(file);
        fileName.textContent = file.name;
        preview.style.display = "block";
    }

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
        if ((!targetFace || !targetFace.files.length) && !cartoonFaceValue) {
            alert("Please select a character face from the gallery.");
            return;
        }
        if (!targetVoice.files.length) {
            alert("Please upload target persona voice audio or video.");
            return;
        }

        // Build form data
        const formData = new FormData();
        formData.append("source_video", sourceVideo.files[0]);
        if (targetFace && targetFace.files.length) {
            formData.append("target_face", targetFace.files[0]);
        }
        if (cartoonFaceValue) {
            formData.append("cartoon_face", cartoonFaceValue);
        }
        formData.append("target_voice", targetVoice.files[0]);
        formData.append("language", document.getElementById("language").value);
        formData.append("voice_name", document.getElementById("voice_name").value);

        // Background change params
        if (bgEnabled) {
            if (bgMode === "prompt" && bgPromptInput.value.trim()) {
                formData.append("bg_prompt", bgPromptInput.value.trim());
            } else if (bgMode === "upload" && bgImageInput && bgImageInput.files.length > 0) {
                formData.append("bg_image", bgImageInput.files[0]);
            }
        }

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
        const stageOrder = bgEnabled
            ? ["extract_audio", "voice_clone", "face_swap", "bg_change", "combine"]
            : ["extract_audio", "voice_clone", "face_swap", "combine"];
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
        const stageWeights = bgEnabled
            ? { extract_audio: 5, voice_clone: 25, face_swap: 30, bg_change: 30, combine: 10 }
            : { extract_audio: 10, voice_clone: 35, face_swap: 45, combine: 10 };
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
        const resultStats = document.getElementById("resultStats");

        resultVideo.src = `/api/download/${jobId}`;
        downloadBtn.href = `/api/download/${jobId}`;

        // Fetch final status to show stats
        fetch(`/api/status/${jobId}`)
            .then(r => r.json())
            .then(data => {
                if (data.stats) {
                    const parts = [];
                    if (data.stats.total_time_seconds) parts.push(`${data.stats.total_time_seconds}s total`);
                    if (data.stats.output_size_mb) parts.push(`${data.stats.output_size_mb} MB`);
                    if (data.stats.face_swap && data.stats.face_swap.swap_rate) parts.push(`${data.stats.face_swap.swap_rate} faces swapped`);
                    if (resultStats && parts.length) resultStats.textContent = parts.join(" \u00b7 ");
                }
            })
            .catch(() => {});

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
        ["source_video", "target_voice"].forEach((id) => {
            const input = document.getElementById(id);
            const zone = document.querySelector(`[data-input="${id}"]`);
            if (!input || !zone) return;
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

        // Reset background UI
        bgEnabled = false;
        if (bgToggle) bgToggle.checked = false;
        if (bgOptions) bgOptions.style.display = "none";
        if (bgStage) bgStage.style.display = "none";
        if (bgPromptInput) bgPromptInput.value = "";
        if (bgPreviewContainer) bgPreviewContainer.style.display = "none";
        if (bgImageInput) bgImageInput.value = "";
        document.querySelectorAll(".bg-preset-btn").forEach((b) => b.classList.remove("active"));
        const bgZone = document.getElementById("bgUploadZone");
        if (bgZone) {
            bgZone.classList.remove("has-file");
            const icon = bgZone.querySelector(".upload-icon");
            const text = bgZone.querySelector(".upload-text");
            const hint = bgZone.querySelector(".upload-hint");
            if (icon) icon.style.display = "";
            if (text) text.style.display = "";
            if (hint) hint.style.display = "";
            const prev = document.getElementById("bgImagePreview");
            if (prev) prev.style.display = "none";
        }
        const stagesGrid = document.querySelector(".progress-stages");
        if (stagesGrid) stagesGrid.style.gridTemplateColumns = "repeat(4, 1fr)";
    }

    newCloneBtn.addEventListener("click", resetUI);
    retryBtn.addEventListener("click", resetUI);
});
