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
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (error) {
      setError('Unable to classify image. Please ensure the backend is running.');
    }
    setLoading(false);
  };

  const getEmoji = (prediction) => {
    const emojis = { 
      burger: '🍔',
      pizza: '🍕', 
      steak: '🥩', 
      sushi: '🍣'
    };
    return emojis[prediction] || '🍽️';
  };

  const getCalorieInfo = (prediction) => {
    const calorieData = {
      burger: { small: 300, large: 600 },
      pizza: { small: 200, large: 400 },
      steak: { small: 250, large: 500 },
      sushi: { small: 150, large: 300 }
    };
    return calorieData[prediction] || { small: 200, large: 400 };
  };

  return (
    <div className="app">
      {/* Navigation */}
      <nav className="nav">
        <div className="nav-container">
          <div className="nav-brand">FoodAI</div>
          <div className="nav-links">
            <a href="#demo">Demo</a>
            <a href="#about">About</a>
            <a href="#tech">Technology</a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section id="demo" className="hero">
        <div className="container">
          <div className="hero-content">
            <h1>Food Classification AI</h1>
            <p className="hero-subtitle">Advanced deep learning model for real-time food recognition</p>
            
            <div className="demo-card">
              <div className="upload-zone">
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileSelect} 
                  id="file-input" 
                  className="file-input"
                />
                <label htmlFor="file-input" className="upload-label">
                  <div className="upload-icon">📁</div>
                  <span>Choose Image</span>
                  <small>Supports JPG, PNG up to 10MB</small>
                </label>
              </div>
              
              {preview && (
                <div className="preview-container">
                  <img src={preview} alt="Preview" className="preview-img" />
                  <button 
                    onClick={handlePredict} 
                    disabled={loading} 
                    className={`analyze-btn ${loading ? 'loading' : ''}`}
                  >
                    {loading ? (
                      <><span className="spinner"></span>Analyzing...</>
                    ) : (
                      'Analyze Image'
                    )}
                  </button>
                </div>
              )}

              {error && (
                <div className="error-message">
                  <span className="error-icon">⚠️</span>
                  {error}
                </div>
              )}
              
              {result && (
                <div className="result-card">
                  <div className="result-header">
                    <span className="result-emoji">{getEmoji(result.prediction)}</span>
                    <div>
                      <h3>{result.prediction}</h3>
                      <p>{result.confidence}% confidence</p>
                    </div>
                  </div>
                  
                  <div className="calorie-info">
                    <h4>Estimated Calories</h4>
                    <div className="calorie-options">
                      <div className="calorie-option">
                        <span className="portion-size">Small</span>
                        <span className="calorie-count">~{getCalorieInfo(result.prediction).small} calories</span>
                      </div>
                      <div className="calorie-option">
                        <span className="portion-size">Large</span>
                        <span className="calorie-count">~{getCalorieInfo(result.prediction).large} calories</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="confidence-bars">
                    {Object.entries(result.probabilities).map(([cls, prob]) => (
                      <div key={cls} className="confidence-item">
                        <div className="confidence-label">
                          <span>{getEmoji(cls)}</span>
                          <span>{cls}</span>
                        </div>
                        <div className="confidence-bar">
                          <div 
                            className="confidence-fill" 
                            style={{ width: `${prob}%` }}
                          ></div>
                        </div>
                        <span className="confidence-value">{prob}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="about">
        <div className="container">
          <div className="section-header">
            <h2>Project Overview</h2>
            <p>A comprehensive deep learning solution for food image classification</p>
          </div>
          
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-number">87.4%</div>
              <div className="stat-label">Test Accuracy</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">300</div>
              <div className="stat-label">Training Images</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">4</div>
              <div className="stat-label">Food Categories</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">~2M</div>
              <div className="stat-label">Parameters</div>
            </div>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3>High Accuracy</h3>
              <p>Achieves 87.4% test accuracy through advanced CNN architecture with batch normalization and dropout regularization.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Real-time Inference</h3>
              <p>Optimized for fast predictions with efficient model architecture and preprocessing pipeline.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🔬</div>
              <h3>Data Augmentation</h3>
              <p>Enhanced generalization through TrivialAugment and geometric transformations on limited dataset.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section id="tech" className="technology">
        <div className="container">
          <div className="section-header">
            <h2>Technology Stack</h2>
            <p>Built with modern deep learning frameworks and web technologies</p>
          </div>
          
          <div className="tech-grid">
            <div className="tech-category">
              <h3>Deep Learning</h3>
              <div className="tech-items">
                <div className="tech-item">
                  <span className="tech-name">PyTorch</span>
                  <span className="tech-desc">Neural network framework</span>
                </div>
                <div className="tech-item">
                  <span className="tech-name">Computer Vision</span>
                  <span className="tech-desc">Image processing & CNN</span>
                </div>
                <div className="tech-item">
                  <span className="tech-name">Data Augmentation</span>
                  <span className="tech-desc">TrivialAugment & transforms</span>
                </div>
              </div>
            </div>
            
            <div className="tech-category">
              <h3>Backend</h3>
              <div className="tech-items">
                <div className="tech-item">
                  <span className="tech-name">Flask</span>
                  <span className="tech-desc">REST API server</span>
                </div>
                <div className="tech-item">
                  <span className="tech-name">Python</span>
                  <span className="tech-desc">Core programming language</span>
                </div>
                <div className="tech-item">
                  <span className="tech-name">PIL/OpenCV</span>
                  <span className="tech-desc">Image preprocessing</span>
                </div>
              </div>
            </div>
            
            <div className="tech-category">
              <h3>Frontend</h3>
              <div className="tech-items">
                <div className="tech-item">
                  <span className="tech-name">React</span>
                  <span className="tech-desc">User interface framework</span>
                </div>
                <div className="tech-item">
                  <span className="tech-name">Axios</span>
                  <span className="tech-desc">HTTP client library</span>
                </div>
                <div className="tech-item">
                  <span className="tech-name">CSS3</span>
                  <span className="tech-desc">Modern styling & animations</span>
                </div>
              </div>
            </div>
          </div>

          <div className="architecture-card">
            <h3>Why PyTorch?</h3>
            <div className="architecture-points">
              <div className="point">
                <strong>Dynamic Computation Graphs:</strong> Flexible model architecture enabling easy debugging and experimentation during development.
              </div>
              <div className="point">
                <strong>Research-Friendly API:</strong> Intuitive interface perfect for rapid prototyping and implementing cutting-edge architectures.
              </div>
              <div className="point">
                <strong>Production Ready:</strong> Seamless deployment with TorchScript and comprehensive ecosystem support.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-brand">
              <div className="brand-name">FoodAI</div>
              <p>Advanced food classification using deep learning</p>
            </div>
            <div className="footer-links">
              <div className="link-group">
                <h4>Project</h4>
                <a href="#demo">Live Demo</a>
                <a href="#about">Overview</a>
                <a href="#tech">Technology</a>
              </div>
              <div className="link-group">
                <h4>Technical</h4>
                <a href="#">Documentation</a>
                <a href="#">API Reference</a>
                <a href="#">Model Details</a>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            <p>© 2024 FoodAI Classification System. Built with PyTorch & React.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
