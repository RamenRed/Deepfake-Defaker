let imageInput = document.getElementById("formFile");
const img = document.querySelector('#imagePreview');
// img.style.display = "none";
imageInput.addEventListener("change", function() {
    img.style.display = "block";
    const choosedFile = this.files[0];
    if (choosedFile) {
        const reader = new FileReader(); //FileReader is a predefined function of JS
        reader.addEventListener('load', function() {
            img.setAttribute('src', reader
                .result); // [1] because we have 2 images with id avtar,
        });
        reader.readAsDataURL(choosedFile);
    }
})