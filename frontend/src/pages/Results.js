import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Progress } from '../components/ui/progress';
import { Separator } from '../components/ui/separator';
import { 
  ArrowLeft, 
  Download, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  TrendingUp,
  Activity,
  Users,
  FileText
} from 'lucide-react';
import { generateCSVReport, generateTXTReport } from '../mock';
import '../styles/Results.css';

// REACT_APP_BACKEND_URL should already include the /api prefix
const API = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000/api';

const Results = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [analysisId, setAnalysisId] = useState(null);

  // Keep the full results for downloads, but only show top 5 unique pairs in the UI
  const displayPairs = useMemo(() => {
    if (!results?.contractPairs) return [];

    const seen = new Set();
    const uniquePairs = [];

    for (const pair of results.contractPairs) {
      const key = `${pair.contractA}__${pair.contractB}`;
      if (seen.has(key)) continue;
      seen.add(key);
      uniquePairs.push({
        ...pair,
        reasons: Array.from(new Set(pair.reasons || [])),
        suggestions: Array.from(new Set(pair.suggestions || [])),
      });
    }

    uniquePairs.sort((a, b) => (b.riskScore || 0) - (a.riskScore || 0));
    return uniquePairs.slice(0, 5);
  }, [results]);

  useEffect(() => {
    const analyzeFile = async () => {
      // Check if we have a file to analyze from location state
      const fileFromState = location.state?.file;
      
      if (!fileFromState) {
        setError('No file provided. Please upload a CSV from the Home page.');
        setLoading(false);
        return;
      }
      
      try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', fileFromState);
        
        // Call backend API
        const response = await axios.post(`${API}/analyze`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        
        // Store analysis ID and results
        setResults(response.data);
        setAnalysisId(response.data.id || null);
        setLoading(false);
        
      } catch (err) {
        console.error('Analysis failed:', err);
        setError(err.response?.data?.detail || 'Analysis failed. Please try again.');
        setResults(null);
        setLoading(false);
      }
    };
    
    analyzeFile();
  }, [location.state]);

  const getRiskBadge = (level) => {
    const configs = {
      high: { label: 'High Risk', className: 'risk-high', icon: XCircle },
      medium: { label: 'Medium Risk', className: 'risk-medium', icon: AlertTriangle },
      low: { label: 'Low Risk', className: 'risk-low', icon: CheckCircle }
    };
    const config = configs[level] || configs.medium;
    const Icon = config.icon;
    return (
      <Badge className={`risk-badge ${config.className}`}>
        <Icon size={14} />
        {config.label}
      </Badge>
    );
  };

  const downloadCSV = async () => {
    if (!results) return;
    
    try {
      if (analysisId && analysisId !== 'latest') {
        // Download from backend
        const response = await axios.get(`${API}/download/csv/${analysisId}`, {
          responseType: 'blob',
        });
        const blob = new Blob([response.data], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'contractnetai_analysis.csv';
        a.click();
      } else {
        // Generate locally
        const csv = generateCSVReport(results);
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'contractnetai_analysis.csv';
        a.click();
      }
    } catch (err) {
      console.error('CSV download failed:', err);
      // Fallback to local generation
      const csv = generateCSVReport(results);
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'contractnetai_analysis.csv';
      a.click();
    }
  };

  const downloadTXT = async () => {
    if (!results) return;
    
    try {
      if (analysisId && analysisId !== 'latest') {
        // Download from backend
        const response = await axios.get(`${API}/download/txt/${analysisId}`, {
          responseType: 'blob',
        });
        const blob = new Blob([response.data], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'contractnetai_report.txt';
        a.click();
      } else {
        // Generate locally
        const txt = generateTXTReport(results);
        const blob = new Blob([txt], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'contractnetai_report.txt';
        a.click();
      }
    } catch (err) {
      console.error('TXT download failed:', err);
      // Fallback to local generation
      const txt = generateTXTReport(results);
      const blob = new Blob([txt], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'contractnetai_report.txt';
      a.click();
    }
  };

  if (loading) {
    return (
      <div className="results-container loading">
        <div className="loading-content">
          <div className="loading-spinner"></div>
          <h2>Analyzing Contract Relationships...</h2>
          <p>Processing transaction data and detecting dependencies</p>
          {error && (
            <p className="error-message" style={{ color: '#ff6b35', marginTop: '1rem' }}>
              {error}
            </p>
          )}
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="results-container loading">
        <div className="loading-content">
          <div className="loading-spinner"></div>
          <h2>Awaiting Analysis</h2>
          <p>Please go back and upload a CSV to run the analysis.</p>
          {error && (
            <p className="error-message" style={{ color: '#ff6b35', marginTop: '1rem' }}>
              {error}
            </p>
          )}
          <Button 
            variant="ghost" 
            onClick={() => navigate('/')}
            className="back-button"
            style={{ marginTop: '1rem' }}
          >
            <ArrowLeft size={20} />
            Back to Upload
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="results-container">
      {/* Header */}
      <div className="results-header">
        <Button 
          variant="ghost" 
          onClick={() => navigate('/')}
          className="back-button"
        >
          <ArrowLeft size={20} />
          Back to Upload
        </Button>
        
        <h1 className="results-title">Analysis Results</h1>
      </div>

      {/* Summary Metrics */}
      <section className="metrics-section">
        <Card className="metric-card">
          <div className="metric-icon-wrapper cyan">
            <FileText size={24} />
          </div>
          <div className="metric-content">
            <p className="metric-label">Total Contracts</p>
            <h3 className="metric-value">{results.summary.totalContracts}</h3>
          </div>
        </Card>
        
        <Card className="metric-card">
          <div className="metric-icon-wrapper lime">
            <Activity size={24} />
          </div>
          <div className="metric-content">
            <p className="metric-label">Linked Contracts</p>
            <h3 className="metric-value">{results.summary.linkedContracts}</h3>
          </div>
        </Card>
        
        <Card className="metric-card">
          <div className="metric-icon-wrapper orange">
            <Users size={24} />
          </div>
          <div className="metric-content">
            <p className="metric-label">Shared Counterparties</p>
            <h3 className="metric-value">{results.summary.sharedCounterparties}</h3>
          </div>
        </Card>
        
        <Card className="metric-card">
          <div className="metric-icon-wrapper red">
            <TrendingUp size={24} />
          </div>
          <div className="metric-content">
            <p className="metric-label">Avg Risk Score</p>
            <h3 className="metric-value">{(results.summary.avgRiskScore * 100).toFixed(0)}%</h3>
          </div>
        </Card>
      </section>

      {/* Contract Relationships */}
      <section className="relationships-section">
        <h2 className="section-title">Contract Relationships</h2>
        
        <div className="relationships-grid">
          {displayPairs.map((pair) => (
            <Card key={pair.id} className="relationship-card">
              <div className="relationship-header">
                <div className="contract-names">
                  <span className="contract-name">{pair.contractA}</span>
                  <span className="connector">↔</span>
                  <span className="contract-name">{pair.contractB}</span>
                </div>
                {getRiskBadge(pair.riskLevel)}
              </div>
              
              <div className="shared-counterparty">
                <Users size={16} />
                <span>Shared Counterparty: <strong>{pair.sharedCounterparty}</strong></span>
              </div>
              
              <div className="risk-score">
                <div className="score-header">
                  <span>Risk Score</span>
                  <span className="score-value">{(pair.riskScore * 100).toFixed(0)}%</span>
                </div>
                <Progress value={pair.riskScore * 100} className="risk-progress" />
              </div>
              
              <Separator className="card-separator" />
              
              <div className="relationship-details">
                <h4>Why They Are Linked:</h4>
                <ul className="reasons-list">
                  {pair.reasons.map((reason, index) => (
                    <li key={index}>
                      <span className="bullet">•</span>
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="relationship-stats">
                <div className="stat">
                  <span className="stat-label">Transactions</span>
                  <span className="stat-value">{pair.transactionCount}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Failure Rate</span>
                  <span className="stat-value">{pair.failureRate}%</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Total Value</span>
                  <span className="stat-value">${(pair.totalValue / 1000000).toFixed(1)}M</span>
                </div>
              </div>
              
              <Separator className="card-separator" />
              
              <div className="suggestions">
                <h4>Recommendations:</h4>
                <ul className="suggestions-list">
                  {pair.suggestions.map((suggestion, index) => (
                    <li key={index}>
                      <span className="suggestion-arrow">→</span>
                      {suggestion}
                    </li>
                  ))}
                </ul>
              </div>
            </Card>
          ))}
        </div>
      </section>

      {/* Download Section */}
      <section className="download-section">
        <h2 className="section-title">Download Reports</h2>
        
        <div className="download-cards">
          <Card className="download-card">
            <FileText size={32} className="download-icon" />
            <h3>CSV Report</h3>
            <p>Download full contract relationship data in spreadsheet format</p>
            <Button onClick={downloadCSV} className="download-button">
              <Download size={18} />
              Download CSV
            </Button>
          </Card>
          
          <Card className="download-card">
            <FileText size={32} className="download-icon" />
            <h3>TXT Report</h3>
            <p>Download comprehensive analysis report with explanations and suggestions</p>
            <Button onClick={downloadTXT} className="download-button">
              <Download size={18} />
              Download TXT
            </Button>
          </Card>
        </div>
      </section>
    </div>
  );
};

export default Results;
