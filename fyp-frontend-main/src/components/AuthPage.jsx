import React, { useState } from 'react';
import { Mail, Lock, User, Eye, EyeOff, Shield, ArrowRight } from 'lucide-react';
import { api } from '../utils/api';

const LoginForm = ({ onLogin, onSwitchToSignup }) => {
  const [loginData, setLoginData] = useState({ username: '', password: '' });
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);

    try {
      await onLogin(loginData);
    } catch (err) {
      setError(err.message || 'Login failed. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div>
      <h2 className="text-2xl font-bold text-white mb-6">Welcome Back</h2>
      <form onSubmit={handleLogin}>
        <div className="mb-4">
          <label className="text-neutral-300 text-sm mb-2 block">Username</label>
          <div className="relative">
            <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-300" />
            <input
              type="text"
              placeholder="Enter your username"
              value={loginData.username}
              onChange={(e) => setLoginData({ ...loginData, username: e.target.value })}
              className="w-full bg-neutral-600 text-white pl-12 pr-4 py-3 rounded-lg border border-transparent focus:border-primary focus:outline-none"
              required
            />
          </div>
        </div>

        <div className="mb-4">
          <label className="text-neutral-300 text-sm mb-2 block">Password</label>
          <div className="relative">
            <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-300" />
            <input
              type={showPassword ? 'text' : 'password'}
              placeholder="Password"
              value={loginData.password}
              onChange={(e) => setLoginData({ ...loginData, password: e.target.value })}
              className="w-full bg-neutral-600 text-white pl-12 pr-12 py-3 rounded-lg border border-transparent focus:border-primary focus:outline-none"
              required
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-neutral-300"
            >
              {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between mb-4">
          <label className="flex items-center gap-2 text-neutral-300 cursor-pointer">
            <input
              type="checkbox"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
              className="w-4 h-4"
            />
            <span className="text-sm">Remember me</span>
          </label>
          <button type="button" className="text-sm text-primary hover:underline">
            Forgot password?
          </button>
        </div>

        {error && <p className="text-danger text-sm mb-4">{error}</p>}

        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full bg-primary text-white py-3 rounded-lg font-semibold hover:bg-opacity-90 transition flex items-center justify-center gap-2 disabled:opacity-60"
        >
          {isSubmitting ? 'Signing In...' : 'Sign In'} <ArrowRight className="w-5 h-5" />
        </button>

        <p className="text-center text-neutral-300 text-sm mt-4">
          Don't have an account?{' '}
          <button
            type="button"
            onClick={onSwitchToSignup}
            className="text-primary hover:underline"
          >
            Sign Up
          </button>
        </p>
      </form>
    </div>
  );
};

const SignupForm = ({ onSignup, onSwitchToLogin }) => {
  const [signupData, setSignupData] = useState({
    username: '',
    password: '',
    confirmPassword: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSignup = async (e) => {
    e.preventDefault();
    setError('');

    if (signupData.password !== signupData.confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setIsSubmitting(true);
    try {
      await onSignup({ username: signupData.username, password: signupData.password });
    } catch (err) {
      setError(err.message || 'Signup failed. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div>
      <h2 className="text-2xl font-bold text-white mb-6">Create Account</h2>
      <form onSubmit={handleSignup}>
        <div className="mb-4">
          <label className="text-neutral-300 text-sm mb-2 block">Username</label>
          <div className="relative">
            <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-300" />
            <input
              type="text"
              placeholder="Choose a username"
              value={signupData.username}
              onChange={(e) => setSignupData({ ...signupData, username: e.target.value })}
              className="w-full bg-neutral-600 text-white pl-12 pr-4 py-3 rounded-lg border border-transparent focus:border-primary focus:outline-none"
              required
            />
          </div>
        </div>

        <div className="mb-4">
          <label className="text-neutral-300 text-sm mb-2 block">Password</label>
          <div className="relative">
            <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-300" />
            <input
              type={showPassword ? 'text' : 'password'}
              placeholder="Password"
              value={signupData.password}
              onChange={(e) => setSignupData({ ...signupData, password: e.target.value })}
              className="w-full bg-neutral-600 text-white pl-12 pr-12 py-3 rounded-lg border border-transparent focus:border-primary focus:outline-none"
              required
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-neutral-300"
            >
              {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
            </button>
          </div>
        </div>

        <div className="mb-6">
          <label className="text-neutral-300 text-sm mb-2 block">Confirm Password</label>
          <div className="relative">
            <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-300" />
            <input
              type="password"
              placeholder="Confirm password"
              value={signupData.confirmPassword}
              onChange={(e) => setSignupData({ ...signupData, confirmPassword: e.target.value })}
              className="w-full bg-neutral-600 text-white pl-12 pr-4 py-3 rounded-lg border border-transparent focus:border-primary focus:outline-none"
              required
            />
          </div>
        </div>

        {error && <p className="text-danger text-sm mb-4">{error}</p>}

        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full bg-primary text-white py-3 rounded-lg font-semibold hover:bg-opacity-90 transition flex items-center justify-center gap-2 disabled:opacity-60"
        >
          {isSubmitting ? 'Creating...' : 'Create Account'} <ArrowRight className="w-5 h-5" />
        </button>

        <p className="text-center text-neutral-300 text-sm mt-4">
          Already have an account?{' '}
          <button
            type="button"
            onClick={onSwitchToLogin}
            className="text-primary hover:underline"
          >
            Login
          </button>
        </p>
      </form>
    </div>
  );
};

export const AuthPage = ({ onLoginSuccess }) => {
  const [authMode, setAuthMode] = useState('login');

  const handleLoginSuccess = async (credentials) => {
    await api.login(credentials);
    onLoginSuccess();
  };

  const handleSignupSuccess = async (credentials) => {
    await api.signup(credentials);
    onLoginSuccess();
  };

  return (
    <div className="min-h-screen bg-neutral-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <Shield className="w-16 h-16 text-primary mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-white">VeriSynth</h1>
        </div>

        <div className="flex gap-2 mb-8">
          <button
            onClick={() => setAuthMode('login')}
            className={`flex-1 py-3 rounded-lg font-semibold transition ${
              authMode === 'login'
                ? 'bg-primary text-white'
                : 'bg-neutral-600 text-neutral-300'
            }`}
          >
            Login
          </button>
          <button
            onClick={() => setAuthMode('signup')}
            className={`flex-1 py-3 rounded-lg font-semibold transition ${
              authMode === 'signup'
                ? 'bg-primary text-white'
                : 'bg-neutral-600 text-neutral-300'
            }`}
          >
            Sign Up
          </button>
        </div>

        {authMode === 'login' ? (
          <LoginForm
            onLogin={handleLoginSuccess}
            onSwitchToSignup={() => setAuthMode('signup')}
          />
        ) : (
          <SignupForm
            onSignup={handleSignupSuccess}
            onSwitchToLogin={() => setAuthMode('login')}
          />
        )}
      </div>
    </div>
  );
};
