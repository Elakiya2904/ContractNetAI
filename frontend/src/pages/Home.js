import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Upload, Network, Shield, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';
import '../styles/Home.css';

const Home = () => {
  const navigate = useNavigate();
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const fileInputRef = useRef(null);
  const uploadSectionRef = useRef(null);

  const handleGetStarted = () => {
    uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      processFile(file);
    } else {
      alert('Please upload a CSV file');
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file) => {
    setUploadedFile(file);
    
    // Read and preview CSV
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split('\n').slice(0, 6); // First 5 rows + header
      setFilePreview(lines);
    };
    reader.readAsText(file);
  };

  const handleAnalyze = () => {
    if (!uploadedFile) return;
    
    // Navigate to results page with the file
    // Results page will handle the API call
    navigate('/results', { state: { file: uploadedFile } });
  };

  return (
    <div className="home-container">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-background">
          <div className="gradient-orb orb-1"></div>
          <div className="gradient-orb orb-2"></div>
          <div className="gradient-orb orb-3"></div>
        </div>
        
        <div className="hero-content">
          <div className="hero-badge">
            <Network className="badge-icon" size={16} />
            <span>Cross-Contract Intelligence Platform</span>
          </div>
          
          <h1 className="hero-title">
            <span className="title-icon">⛓️</span>
            ContractNetAI
          </h1>
          
          <p className="hero-subtitle">Cross-Contract Intelligence & Risk Awareness</p>
          
          <div className="hero-features">
            <div className="feature-line">
              <CheckCircle size={18} className="feature-icon" />
              <span>See beyond individual smart contracts</span>
            </div>
            <div className="feature-line">
              <AlertTriangle size={18} className="feature-icon" />
              <span>Detect hidden dependencies before they become risks</span>
            </div>
            <div className="feature-line">
              <TrendingUp size={18} className="feature-icon" />
              <span>Understand your contract ecosystem</span>
            </div>
          </div>
          
          <Button 
            onClick={handleGetStarted}
            className="cta-button"
            size="lg"
          >
            Get Started
            <span className="button-arrow">→</span>
          </Button>
        </div>
      </section>

      {/* Explanation Section */}
      <section className="explanation-section">
        <div className="explanation-content">
          <h2 className="section-title">What is ContractNetAI?</h2>
          
          <div className="features-grid">
            <Card className="feature-card">
              <div className="feature-icon-wrapper">
                <Network className="feature-icon-large" />
              </div>
              <h3>Multi-Contract Analysis</h3>
              <p>Analyzes relationships across multiple smart contracts to identify interconnected dependencies and potential conflicts.</p>
            </Card>
            
            <Card className="feature-card">
              <div className="feature-icon-wrapper">
                <Shield className="feature-icon-large" />
              </div>
              <h3>Risk Detection</h3>
              <p>Detects shared counterparties, circular dependencies, and vulnerability patterns to prevent operational and financial risks.</p>
            </Card>
            
            <Card className="feature-card">
              <div className="feature-icon-wrapper">
                <TrendingUp className="feature-icon-large" />
              </div>
              <h3>Predictive Insights</h3>
              <p>Provides actionable recommendations and risk scores to help you make informed decisions about contract management.</p>
            </Card>
          </div>
        </div>
      </section>

      {/* Upload Section */}
      <section className="upload-section" ref={uploadSectionRef}>
        <div className="upload-content">
          <h2 className="section-title">Upload Transaction Data</h2>
          <p className="section-description">
            Upload your smart contract transaction data in CSV format to begin analysis
          </p>
          
          <Card className="upload-card">
            <div 
              className={`upload-zone ${isDragging ? 'dragging' : ''} ${uploadedFile ? 'uploaded' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="upload-icon" size={48} />
              <h3>{uploadedFile ? 'File Uploaded Successfully' : 'Drag & Drop CSV File'}</h3>
              <p>{uploadedFile ? uploadedFile.name : 'or click to browse'}</p>
              <input 
                ref={fileInputRef}
                type="file" 
                accept=".csv" 
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
            </div>
            
            {filePreview && (
              <div className="file-preview">
                <h4>File Preview</h4>
                <div className="preview-content">
                  {filePreview.map((line, index) => (
                    <div key={index} className="preview-line">
                      {line}
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {uploadedFile && (
              <div className="upload-actions">
                <Button 
                  onClick={handleAnalyze}
                  className="analyze-button"
                  size="lg"
                >
                  Analyze Contracts
                  <span className="button-arrow">→</span>
                </Button>
              </div>
            )}
          </Card>
        </div>
      </section>
    </div>
  );
};

export default Home;
