import React from 'react';
import { Shield, ArrowRight, BarChart3, FileVideo } from 'lucide-react';

export const LandingPage = ({ onGetStarted }) => (
  <div className="min-h-screen bg-neutral-900 text-white">
    {/* Navigation */}
    <nav className="flex justify-between items-center px-8 py-6 border-b border-neutral-600">
      <div className="flex items-center gap-3">
        <Shield className="w-8 h-8 text-primary" />
        <span className="text-2xl font-bold">VeriSynth</span>
      </div>
      <button 
        onClick={onGetStarted}
        className="px-6 py-2 bg-primary text-white rounded-lg hover:bg-opacity-90 transition"
      >
        Get Started
      </button>
    </nav>

    {/* Hero Section */}
    <div className="max-w-6xl mx-auto px-8 py-20">
      <div className="text-center mb-16">
        <h1 className="text-6xl font-bold mb-6 leading-tight">
          Detect Deepfakes with
          <span className="text-primary"> AI Precision</span>
        </h1>
        <p className="text-xl text-neutral-300 mb-8 max-w-2xl mx-auto">
          Advanced deepfake detection platform powered by cutting-edge AI technology. 
          Protect your content authenticity with 99.2% accuracy.
        </p>
        <button 
          onClick={onGetStarted}
          className="px-8 py-4 bg-primary text-white rounded-lg text-lg font-semibold hover:bg-opacity-90 transition inline-flex items-center gap-2"
        >
          Start Analyzing <ArrowRight className="w-5 h-5" />
        </button>
      </div>

      {/* Features */}
      <div className="grid md:grid-cols-3 gap-8 mt-20">
        <div className="bg-neutral-600 bg-opacity-30 p-8 rounded-xl border border-neutral-600">
          <div className="w-14 h-14 bg-primary bg-opacity-20 rounded-lg flex items-center justify-center mb-4">
            <Shield className="w-7 h-7 text-primary" />
          </div>
          <h3 className="text-xl font-bold mb-3">Advanced Detection</h3>
          <p className="text-neutral-300">
            State-of-the-art AI algorithms detect even the most sophisticated deepfakes
          </p>
        </div>

        <div className="bg-neutral-600 bg-opacity-30 p-8 rounded-xl border border-neutral-600">
          <div className="w-14 h-14 bg-success bg-opacity-20 rounded-lg flex items-center justify-center mb-4">
            <BarChart3 className="w-7 h-7 text-success" />
          </div>
          <h3 className="text-xl font-bold mb-3">99.2% Accuracy</h3>
          <p className="text-neutral-300">
            Industry-leading accuracy rates backed by continuous model training
          </p>
        </div>

        <div className="bg-neutral-600 bg-opacity-30 p-8 rounded-xl border border-neutral-600">
          <div className="w-14 h-14 bg-accent-ai bg-opacity-20 rounded-lg flex items-center justify-center mb-4">
            <FileVideo className="w-7 h-7 text-accent-ai" />
          </div>
          <h3 className="text-xl font-bold mb-3">Multiple Formats</h3>
          <p className="text-neutral-300">
            Support for MP4, AVI, MOV, WebM formats up to 500MB
          </p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid md:grid-cols-4 gap-8 mt-20 text-center">
        <div>
          <div className="text-4xl font-bold text-primary mb-2">127</div>
          <div className="text-neutral-300">Videos Analyzed</div>
        </div>
        <div>
          <div className="text-4xl font-bold text-success mb-2">99.2%</div>
          <div className="text-neutral-300">Detection Accuracy</div>
        </div>
        <div>
          <div className="text-4xl font-bold text-primary mb-2">&lt;3s</div>
          <div className="text-neutral-300">Average Analysis Time</div>
        </div>
        <div>
          <div className="text-4xl font-bold text-success mb-2">24/7</div>
          <div className="text-neutral-300">Platform Availability</div>
        </div>
      </div>
    </div>
  </div>
);
