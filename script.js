const form = document.getElementById("myForm");
const inputType = document.getElementById("inputType");
const inputField = document.getElementById("inputField");
const imageFile = document.getElementById("imageFile");
const query = document.getElementById("query");
const infoText = document.getElementById("infoText");
const preview = document.getElementById("preview");

form.addEventListener("submit", function (event) {
  event.preventDefault();

  const selectedType = inputType.value;
  let inputData = new FormData();
  inputData.append("inputType", selectedType);
  inputData.append("query", query.value);

  // Adjust the 'inputData' FormData object based on the input type
  if (selectedType === "image") {
    inputData.append("inputData", imageFile.files[0]);
    console.log(inputData);
  } else {
    // For video and webcam, additional logic is needed to handle file/stream
    console.error(`${selectedType} input not yet implemented.`);
    return;
  }

  fetch("http://localhost:5000/process", {
    method: "POST",
    body: inputData,
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      console.log("data:", data);
        if (data.detections && data.detections.length > 0) {
          let detectionsText = data.detections
            .map((det) => {
              return `${det.label} (${det.confidence * 100}%) - Box: [${
                det.bounding_box.xmin
              }, ${det.bounding_box.ymin}, ${det.bounding_box.xmax}, ${
                det.bounding_box.ymax
              }]`;
            })
            .join("\n");
          infoText.textContent = detectionsText;
        } else {
          infoText.textContent = "No detections.";
        }
        
      if (data.preview) {
        preview.src = `data:image/jpeg;base64,${data.preview}`;
        preview.style.display = "block";
      } else {
        preview.style.display = "none";
      }
    })
    .catch((error) => {
      console.error(error);
      infoText.textContent = "Error processing input. Please try again.";
    });
});

inputType.addEventListener("change", function () {
  if (this.value === "image") {
    inputField.style.display = "block";
  } else {
    inputField.style.display = "none";
  }
});
