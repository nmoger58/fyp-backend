const API_BASE_URL = 'http://localhost:8000';
const ACCESS_TOKEN_KEY = 'access_token';

const getStoredToken = () => localStorage.getItem(ACCESS_TOKEN_KEY);

const buildHeaders = (headers = {}, authRequired = false) => {
  const finalHeaders = {
    accept: 'application/json',
    ...headers,
  };

  if (authRequired) {
    const token = getStoredToken();
    if (token) {
      finalHeaders.Authorization = `Bearer ${token}`;
    }
  }

  return finalHeaders;
};

const request = async (path, options = {}, authRequired = false) => {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: buildHeaders(options.headers, authRequired),
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const error = new Error(data?.detail || data?.error || 'Request failed');
    error.status = response.status;

    if (response.status === 401) {
      localStorage.removeItem(ACCESS_TOKEN_KEY);
    }

    throw error;
  }

  return data;
};

export const api = {
  login: async ({ username, password }) => {
    const data = await request(
      '/auth/login',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      },
      false
    );

    if (data?.access_token) {
      localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
    }

    return data;
  },

  signup: async ({ username, password }) => {
    const data = await request(
      '/auth/signup',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      },
      false
    );

    if (data?.access_token) {
      localStorage.setItem(ACCESS_TOKEN_KEY, data.access_token);
    }

    return data;
  },

  me: async () => request('/auth/me', { method: 'GET' }, true),

  logout: () => {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
  },

  isAuthenticated: () => Boolean(getStoredToken()),

  health: async () => {
    try {
      return await request('/health', { method: 'GET' }, false);
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  predictVideo: async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      return await request(
        '/predict_video',
        {
          method: 'POST',
          body: formData,
        },
        true
      );
    } catch (error) {
      console.error('Prediction failed:', error);
      throw error;
    }
  },
};
