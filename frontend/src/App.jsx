import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];

    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));

    setPrediction("");
    setConfidence("");
  };

  const predictDisease = async () => {
    if (!image) {
      alert("Please choose an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      );

      setPrediction(response.data.class);
      setConfidence(response.data.confidence);
    } catch (err) {
      console.error(err);
      alert("Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const getRecommendation = () => {
    if (prediction.includes("Early")) {
      return "Remove infected leaves and apply a protective fungicide.";
    }

    if (prediction.includes("Late")) {
      return "Immediately isolate infected plants and apply systemic fungicide.";
    }

    return "Healthy plant. Continue proper irrigation and regular monitoring.";
  };

  return (
    <div className="container">

      <h1>🌿 Kloris Potato Disease Detection</h1>

      <div className="card">

        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
        />

        {preview && (
          <img
            src={preview}
            alt="Preview"
            className="preview"
          />
        )}

        <button onClick={predictDisease}>
          Predict Disease
        </button>

        {loading && (
          <div className="loader">
            Predicting...
          </div>
        )}

        {prediction && !loading && (
          <div className="result">

            <h2>Prediction</h2>

            <h3>
              {prediction.replaceAll("_", " ")}
            </h3>

            <p>
              Confidence: <b>{confidence}%</b>
            </p>

            <div className="progress">

              <div
                className="progress-fill"
                style={{
                  width: `${confidence}%`
                }}
              ></div>

            </div>

            <div className="recommendation">

              <h4>Recommendation</h4>

              <p>{getRecommendation()}</p>

            </div>

          </div>
        )}

      </div>

      <footer>
        Made with ❤️ using React • FastAPI • TensorFlow
      </footer>

    </div>
  );
}

export default App;