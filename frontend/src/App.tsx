import React, { useState, useEffect } from 'react';
import './App.css';
import CommercialWebsite from './components/CommercialWebsite';
import OrbitalVisualization from './components/OrbitalVisualization';

function App() {
  const [showVisualization, setShowVisualization] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  if (showVisualization) {
    return (
      <div className={`App visualization-mode ${isLoaded ? 'loaded' : ''}`}>
        <div className="stars-bg"></div>
        <div className="nebula-bg"></div>
        <header className="App-header glass-panel">
          <div className="header-content">
            <div className="brand-section">
              <div className="logo-glow"></div>
              <h1 className="brand-title">
                <span className="brand-orbit">Re</span>
                <span className="brand-clean">OrbitAr</span>
              </h1>
              <span className="header-subtitle">3D Orbital Visualization</span>
            </div>
            <button
              className="back-button neon-button"
              onClick={() => setShowVisualization(false)}
            >
              <span className="button-icon">←</span>
              <span className="button-text">Return to Dashboard</span>
            </button>
          </div>
        </header>
        <main className="App-main">
          <OrbitalVisualization />
        </main>
      </div>
    );
  }

  return (
    <div className={`App ${isLoaded ? 'loaded' : ''}`}>
      <div className="stars-bg"></div>
      <div className="nebula-bg"></div>
      <CommercialWebsite onShowVisualization={() => setShowVisualization(true)} />
    </div>
  );
}

export default App;