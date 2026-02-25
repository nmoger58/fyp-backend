import React from 'react';
import { FileVideo, XCircle } from 'lucide-react';

export const ReadyPage = ({ uploadedFile, onStartAnalysis, onCancel }) => {
  return (
    <div className="min-h-screen bg-neutral-900 flex items-center justify-center p-4">
      <div className="w-full max-w-3xl bg-neutral-600 rounded-xl p-8">
        <div className="flex justify-between items-start mb-8">
          <h2 className="text-white text-2xl font-bold">Ready to Analyze</h2>
          <button 
            onClick={onCancel}
            className="text-danger hover:text-red-400"
          >
            <XCircle className="w-6 h-6" />
          </button>
        </div>

        <div className="bg-neutral-900 rounded-lg p-6 mb-8 flex items-center gap-4">
          <div className="w-16 h-16 bg-primary-dark rounded-lg flex items-center justify-center">
            <FileVideo className="w-8 h-8 text-primary" />
          </div>
          <div>
            <div className="text-white font-semibold text-lg">{uploadedFile?.name}</div>
            <div className="text-neutral-300 text-sm">{uploadedFile?.size}</div>
          </div>
        </div>

        <button
          onClick={onStartAnalysis}
          className="w-full bg-primary text-white py-4 rounded-lg font-semibold text-lg hover:bg-opacity-90 transition flex items-center justify-center gap-3"
        >
          <FileVideo className="w-6 h-6" />
          Start Analysis
        </button>
      </div>
    </div>
  );
};
