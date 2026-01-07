import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (error) {
      setError('Error classifying image. Please try again.');
      console.error('Error:', error);
    }
    setLoading(false);
  };

  const getEmoji = (prediction) => {
    const emojis = { pizza: '🍕', steak: '🥩', sushi: '🍣' };
    return emojis[prediction] || '🍽️';
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🍕🥩🍣 Food Classifier</h1>
        <p>Upload an image to classify it as Pizza, Steak, or Sushi!</p>
        
        <div className="upload-section">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
            id="file-input"
          />
          <label htmlFor="file-input" className="upload-btn">
            Choose Image
          </label>
        </div>

        {preview && (
          <div className="preview-section">
            <img src={preview} alt="Preview" className="preview-image" />
            <button 
              onClick={handlePredict} 
              disabled={loading}
              className="predict-btn"
            >
              {loading ? 'Classifying...' : 'Classify Image'}
            </button>
          </div>
        )}

        {error && (
          <div className="error-section">
            <p>{error}</p>
          </div>
        )}

        {result && (
          <div className="result-section">
            <h2>
              {getEmoji(result.prediction)} {result.prediction.toUpperCase()}
            </h2>
            <p>Confidence: {result.confidence}%</p>
            
            <div className="probabilities">
              <h3>All Probabilities:</h3>
              {Object.entries(result.probabilities).map(([cls, prob]) => (
                <div key={cls} className="prob-item">
                  <span>{getEmoji(cls)} {cls}</span>
                  <div className="prob-bar">
                    <div 
                      className="prob-fill" 
                      style={{ width: `${prob}%` }}
                    ></div>
                  </div>
                  <span>{prob}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
