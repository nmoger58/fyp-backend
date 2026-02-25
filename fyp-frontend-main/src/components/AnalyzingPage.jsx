import React from 'react';

export const AnalyzingPage = () => (
  <div className="min-h-screen bg-neutral-900 flex items-center justify-center p-4">
    <div className="text-center">
      <div className="w-24 h-24 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-8"></div>
      <h2 className="text-white text-3xl font-bold mb-4">Analyzing Video...</h2>
      <p className="text-neutral-300 text-lg">This may take a few moments</p>
    </div>
  </div>
);
