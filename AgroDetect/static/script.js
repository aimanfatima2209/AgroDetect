const fileInput = document.getElementById("fileInput");
const previewSection = document.getElementById("previewSection");
const previewImage = document.getElementById("previewImage");
const loading = document.getElementById("loading");

fileInput.addEventListener("change", function () {

    const file = fileInput.files[0];
    if (!file) return;

    // Show preview
    const reader = new FileReader();
    reader.onload = function (e) {
        previewSection.classList.remove("hidden");
        previewImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Show loader
    loading.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.redirect) {
            window.location.href = data.redirect;
        } else {
            alert("Prediction failed.");
        }
    })
    .catch(error => {
        console.error("Error:", error);
        alert("Something went wrong.");
    });
});
