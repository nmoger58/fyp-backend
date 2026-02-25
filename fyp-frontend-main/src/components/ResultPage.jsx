import React from 'react';
import { FileVideo, XCircle, CheckCircle, Download, Upload } from 'lucide-react';

export const ResultPage = ({ uploadedFile, analysisResult, onDownloadReport, onNewAnalysis }) => {
  const getIndicators = () => {
    if (analysisResult?.is_deepfake) {
      return [
        'Detected frame inconsistencies',
        'Unnatural eye movements',
        'Audio mismatch detected',
        'Facial manipulation markers'
      ];
    } else {
      return [
        'Natural facial movements',
        'Consistent lighting',
        'No frame artifacts',
        'Audio-visual sync verified'
      ];
    }
  };

  const indicators = getIndicators();

  return (
    <div className="min-h-screen bg-neutral-900 flex items-center justify-center p-4">
      <div className="w-full max-w-3xl">
        <div className="flex justify-between items-start mb-8">
          <h2 className="text-white text-3xl font-bold">Analysis Complete</h2>
          <button 
            onClick={onNewAnalysis}
            className="text-danger hover:text-red-400"
          >
            <XCircle className="w-7 h-7" />
          </button>
        </div>

        <div className="bg-neutral-900 rounded-lg p-6 mb-6 flex items-center gap-4 border border-neutral-600">
          <div className="w-16 h-16 bg-primary-dark rounded-lg flex items-center justify-center">
            <FileVideo className="w-8 h-8 text-primary" />
          </div>
          <div>
            <div className="text-white font-semibold text-lg">{uploadedFile?.name}</div>
            <div className="text-neutral-300 text-sm">{uploadedFile?.size}</div>
          </div>
        </div>

        {analysisResult?.error ? (
          // Error state
          <div className="rounded-xl p-8 mb-6 border-2 bg-black bg-opacity-10 border-warning">
            <div className="flex items-center gap-4 mb-6">
              <XCircle className="w-12 h-12 text-warning" />
              <div>
                <h3 className="text-white text-2xl font-bold">Analysis Error</h3>
                <p className="text-neutral-300">{analysisResult.error}</p>
              </div>
            </div>
            <button 
              onClick={onNewAnalysis}
              className="w-full bg-primary text-white py-4 rounded-lg font-semibold hover:bg-opacity-90 transition flex items-center justify-center gap-2"
            >
              <Upload className="w-5 h-5" />
              Try Another Video
            </button>
          </div>
        ) : (
          // Success state
          <>
            <div className={`rounded-xl p-8 mb-6 border-2 ${
              analysisResult?.is_deepfake 
                ? 'bg-black bg-opacity-10 border-danger' 
                : 'bg-black bg-opacity-10 border-success'
            }`}>
              <div className="flex items-center gap-4 mb-6">
                {analysisResult?.is_deepfake ? (
                  <XCircle className="w-12 h-12 text-danger" />
                ) : (
                  <CheckCircle className="w-12 h-12 text-success" />
                )}
                <div>
                  <h3 className="text-white text-2xl font-bold">
                    {analysisResult?.is_deepfake ? 'Deepfake Detected' : 'Authentic Video'}
                  </h3>
                  <p className="text-neutral-300">
                    Confidence: {(analysisResult?.confidence * 100).toFixed(2)}%
                  </p>
                </div>
              </div>

              <div className="w-full bg-neutral-900 rounded-full h-3 mb-6">
                <div 
                  className={`h-3 rounded-full ${analysisResult?.is_deepfake ? 'bg-danger' : 'bg-success'}`}
                  style={{width: `${analysisResult?.confidence * 100}%`}}
                ></div>
              </div>

              <div>
                <h4 className="text-white font-bold mb-3">Key Indicators:</h4>
                <ul className="space-y-2">
                  {indicators.map((indicator, idx) => (
                    <li key={idx} className="text-neutral-300 flex items-start gap-2">
                      <span className={analysisResult?.is_deepfake ? 'text-danger' : 'text-success'}>•</span>
                      {indicator}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <button 
                onClick={onDownloadReport}
                className="bg-neutral-600 text-white py-4 rounded-lg font-semibold hover:bg-opacity-80 transition flex items-center justify-center gap-2"
              >
                <Download className="w-5 h-5" />
                Download Report
              </button>
              <button 
                onClick={onNewAnalysis}
                className="bg-primary text-white py-4 rounded-lg font-semibold hover:bg-opacity-90 transition flex items-center justify-center gap-2"
              >
                <Upload className="w-5 h-5" />
                New Analysis
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
