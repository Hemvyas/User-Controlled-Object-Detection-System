document.getElementById("myForm").addEventListener("submit",async function (e) {
  e.preventDefault(); // Prevent the default form submission

  const inputType = document.getElementById("inputType").value;
  const fileInput = document.getElementById("imageFile");
  const queryInput = document.getElementById("query").value;

  if (inputType !== "image" && inputType !== "video") {
    alert("Currently, only image and video inputs are supported.");
    return;
  }

  if (fileInput.files.length === 0) {
    alert("Please select a file to upload.");
    return;
  }

  const file = fileInput.files[0];

  const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB - Adjust as per your backend configuration

  if (file.size > MAX_FILE_SIZE) {
    alert("File is too large. Maximum size allowed is 16MB.");
    return;
  }

// const controller = new AbortController();
// const timeoutId = setTimeout(() => controller.abort(), 60000);


  const formData = new FormData();
  formData.append("file", file);
  formData.append("query", queryInput);



  try {
    const res= await fetch("http://localhost:5000/upload",{
      method:"POST",
      body:formData,
    })
    if(!res.ok)
    {
      throw new Error('Network response was not ok');
    }
    const data=await  res.json()
    console.log(data);
    displayResults(data);  
  } catch (error) {
    console.error("Error:", error);
    document.getElementById("infoText").textContent = error.message;
    ("Error processing your request. Please try again.");
  }

function displayResults(data) {
  if (data.error) {
    document.getElementById("infoText").textContent = data.error;
    return;
  }

  // Assuming data.predictions is an array of objects with properties like 'class'
  const infoText = document.getElementById("infoText");
  if (data.predictions && data.predictions.length > 0) {
    const resultsText = data.predictions
      .map(
        (prediction) =>
          `${prediction.class} (confidence: ${prediction.confidence}%)`
      )
      .join(", ");
    infoText.textContent = `Detected objects: ${resultsText}`;
  } else {
    infoText.textContent = "No objects detected.";
  }

  // Optionally display the uploaded image (if the backend sends back a URL)
  // For this, your backend needs to save the uploaded file and send back its URL in the response
  // document.getElementById('preview').src = 'URL_TO_THE_UPLOADED_IMAGE';
}
});
