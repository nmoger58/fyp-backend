import React, { useState, useEffect } from 'react';
import { LandingPage } from './components/LandingPage';
import { AuthPage } from './components/AuthPage';
import { Dashboard } from './components/Dashboard';
import { ReadyPage } from './components/ReadyPage';
import { AnalyzingPage } from './components/AnalyzingPage';
import { ResultPage } from './components/ResultPage';
import { api } from './utils/api';
import './App.css';

const App = () => {
  const [currentPage, setCurrentPage] = useState(api.isAuthenticated() ? 'dashboard' : 'landing');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    // Check health on mount
    api.health()
      .then((health) => console.log('API Health:', health))
      .catch((error) => console.error('API is not available:', error));
  }, []);

  useEffect(() => {
    const validateSession = async () => {
      if (!api.isAuthenticated()) return;
      try {
        await api.me();
      } catch (error) {
        api.logout();
        setCurrentPage('auth');
      }
    };

    validateSession();
  }, []);

  useEffect(() => {
    const protectedPages = new Set(['dashboard', 'ready', 'analyzing', 'result']);
    if (protectedPages.has(currentPage) && !api.isAuthenticated()) {
      setCurrentPage('auth');
    }
  }, [currentPage]);

  const handleUploadFile = (file) => {
    const fileSizeInMB = (file.size / (1024 * 1024)).toFixed(2);
    setUploadedFile({
      file,
      name: file.name,
      size: `${fileSizeInMB} MB`
    });
    setCurrentPage('ready');
  };

  const handleStartAnalysis = async () => {
    if (!uploadedFile?.file) return;

    setIsAnalyzing(true);
    setCurrentPage('analyzing');

    try {
      const result = await api.predictVideo(uploadedFile.file);
      const normalizedResult = {
        ...result,
        is_deepfake: result?.prediction?.is_deepfake ?? false,
        confidence: result?.prediction?.confidence ?? 0,
        label: result?.prediction?.label ?? 'N/A',
      };
      setAnalysisResult(normalizedResult);
      setCurrentPage('result');
    } catch (error) {
      console.error('Analysis error:', error);

      if (error.status === 401) {
        api.logout();
        alert('Session expired. Please login again.');
        setCurrentPage('auth');
        return;
      }

      setAnalysisResult({
        error: 'Failed to analyze video. Please try again.',
        filename: uploadedFile.name
      });
      setCurrentPage('result');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleLogout = () => {
    api.logout();
    setUploadedFile(null);
    setAnalysisResult(null);
    setCurrentPage('auth');
  };

  const handleCancelReady = () => {
    setUploadedFile(null);
    setCurrentPage('dashboard');
  };

  const handleNewAnalysis = () => {
    setUploadedFile(null);
    setAnalysisResult(null);
    setCurrentPage('dashboard');
  };

  const handleDownloadReport = () => {
    const reportContent = `
Deepfake Detection Report
========================
Filename: ${uploadedFile?.name}
File Size: ${uploadedFile?.size}
Analysis Status: ${analysisResult?.status || 'completed'}

Result: ${analysisResult?.is_deepfake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC VIDEO'}
Confidence: ${(analysisResult?.confidence * 100).toFixed(2)}%
Label: ${analysisResult?.label || analysisResult?.prediction?.label || 'N/A'}

Generated: ${new Date().toLocaleString()}
    `.trim();

    const element = document.createElement('a');
    element.setAttribute('href', `data:text/plain;charset=utf-8,${encodeURIComponent(reportContent)}`);
    element.setAttribute('download', `report-${uploadedFile?.name?.split('.')[0]}.txt`);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const cssVariables = {
    '--primary': '#4CADE4',
    '--primary-dark': '#063356',
    '--neutral-100': '#EEF3F6',
    '--neutral-300': '#C2C8CE',
    '--neutral-600': '#303235',
    '--neutral-900': '#1A1D1F',
    '--accent-ai': '#7A5CFF',
    '--success': '#12C99B',
    '--danger': '#FF4C4C',
    '--warning': '#F4AA2A'
  };

  const styleSheet = `
    .bg-primary { background-color: var(--primary); }
    .bg-primary-dark { background-color: var(--primary-dark); }
    .bg-neutral-100 { background-color: var(--neutral-100); }
    .bg-neutral-300 { background-color: var(--neutral-300); }
    .bg-neutral-600 { background-color: var(--neutral-600); }
    .bg-neutral-900 { background-color: var(--neutral-900); }
    .bg-accent-ai { background-color: var(--accent-ai); }
    .bg-success { background-color: var(--success); }
    .bg-danger { background-color: var(--danger); }
    .bg-warning { background-color: var(--warning); }

    .text-primary { color: var(--primary); }
    .text-success { color: var(--success); }
    .text-danger { color: var(--danger); }
    .text-accent-ai { color: var(--accent-ai); }
    .text-neutral-300 { color: var(--neutral-300); }

    .border-primary { border-color: var(--primary); }
    .border-success { border-color: var(--success); }
    .border-danger { border-color: var(--danger); }
    .border-neutral-600 { border-color: var(--neutral-600); }
    .border-neutral-300 { border-color: var(--neutral-300); }
    .border-warning { border-color: var(--warning); }
  `;

  return (
    <div style={cssVariables}>
      <style>{styleSheet}</style>

      {currentPage === 'landing' && (
        <LandingPage onGetStarted={() => setCurrentPage(api.isAuthenticated() ? 'dashboard' : 'auth')} />
      )}

      {currentPage === 'auth' && (
        <AuthPage onLoginSuccess={() => setCurrentPage('dashboard')} />
      )}

      {currentPage === 'dashboard' && (
        <Dashboard onUploadFile={handleUploadFile} onLogout={handleLogout} />
      )}

      {currentPage === 'ready' && (
        <ReadyPage
          uploadedFile={uploadedFile}
          onStartAnalysis={handleStartAnalysis}
          onCancel={handleCancelReady}
        />
      )}

      {currentPage === 'analyzing' && (
        <AnalyzingPage />
      )}

      {currentPage === 'result' && (
        <ResultPage
          uploadedFile={uploadedFile}
          analysisResult={analysisResult}
          onDownloadReport={handleDownloadReport}
          onNewAnalysis={handleNewAnalysis}
        />
      )}
    </div>
  );
};

export default App;
