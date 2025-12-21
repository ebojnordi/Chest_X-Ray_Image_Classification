const defaultConfig = {
  page_title: "Pneumonia Predictor",
  page_subtitle: "Upload a chest X-ray image and analyze it with AI",
  app_description:
    "About this application: This powerful ML pneumonia prediction tool has been trained using an advanced artificial neural network (ResNet-18) to analyze chest X-ray images and provide instant predictions of pneumonia. Simply upload a chest X-ray image, click Run Predict, and get results in seconds.",
  upload_button_text: "Choose Image",
  predict_button_text: "Run Prediction",
  background_color: "#667eea",
  surface_color: "#ffffff",
  text_color: "#1f2937",
  primary_action_color: "#10b981",
  secondary_action_color: "#8b5cf6",
  font_family: "system-ui",
  font_size: 16,
};

let uploadedImage = null;

/* -----------------------------
   Upload handlers
------------------------------*/
document.getElementById("upload-button").addEventListener("click", () => {
  document.getElementById("image-upload").click();
});

document.getElementById("upload-area").addEventListener("click", (e) => {
  if (e.target.id !== "upload-button") {
    document.getElementById("image-upload").click();
  }
});

document.getElementById("upload-area").addEventListener("dragover", (e) => {
  e.preventDefault();
  e.currentTarget.style.borderColor = "#667eea";
  e.currentTarget.style.backgroundColor = "#f9fafb";
});

document.getElementById("upload-area").addEventListener("dragleave", (e) => {
  e.currentTarget.style.borderColor = "#d1d5db";
  e.currentTarget.style.backgroundColor = "";
});

document.getElementById("upload-area").addEventListener("drop", (e) => {
  e.preventDefault();
  e.currentTarget.style.borderColor = "#d1d5db";
  e.currentTarget.style.backgroundColor = "";

  const files = e.dataTransfer.files;
  if (files.length > 0 && files[0].type.startsWith("image/")) {
    handleImageUpload(files[0]);
  }
});

document.getElementById("image-upload").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleImageUpload(file);
});

function handleImageUpload(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadedImage = file;
    document.getElementById("image-container").innerHTML = `
      <img src="${e.target.result}" class="w-full h-full object-contain" />
    `;
    document.getElementById("predict-button").disabled = false;
    document.getElementById("results-section").classList.add("hidden");
  };
  reader.readAsDataURL(file);
}

/* -----------------------------
   Prediction (REAL API)
------------------------------*/
document
  .getElementById("predict-button")
  .addEventListener("click", async () => {
    if (!uploadedImage) return;

    const button = document.getElementById("predict-button");
    button.disabled = true;
    button.textContent = "Processing...";

    const formData = new FormData();
    formData.append("file", uploadedImage);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      document.getElementById("prediction-label").textContent = result.label;
      const labelEl = document.getElementById("prediction-label");

      // remove old classes first (important!)
      labelEl.classList.remove("green-label", "red-label");

      if (result.label === "NORMAL") {
        labelEl.classList.add("green-label");
      } else {
        labelEl.classList.add("red-label");
      }

      document.getElementById("confidence-label").textContent =
        result.confidence + "%";
      document.getElementById("time-label").textContent =
        result.processing_time_ms + " ms";
      document.getElementById("result-description").value = result.description;

      document.getElementById("results-section").classList.remove("hidden");
    } catch (error) {
      alert("Prediction failed. Please try again.");
      console.error(error);
    } finally {
      button.disabled = false;
      button.textContent = defaultConfig.predict_button_text;
    }
  });
