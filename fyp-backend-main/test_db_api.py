import requests
import sys
import os

BASE_URL = "http://localhost:8000"
USERNAME = "testuser_db"
PASSWORD = "testpassword123"

# 1. Signup
print("1. Testing Signup...")
resp = requests.post(f"{BASE_URL}/auth/signup", json={
    "username": USERNAME,
    "password": PASSWORD,
    "full_name": "Test User DB"
})
if resp.status_code == 409:
    print("User already exists, proceeding to login...")
elif resp.status_code != 200:
    print(f"Signup failed: {resp.text}")
    sys.exit(1)
else:
    print("Signup successful")

# 2. Login
print("\n2. Testing Login...")
resp = requests.post(f"{BASE_URL}/auth/login", data={
    "username": USERNAME,
    "password": PASSWORD
})
if resp.status_code != 200:
    print(f"Login failed: {resp.text}")
    sys.exit(1)

token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
print("Login successful, token retrieved")

# 3. Check /auth/me
print("\n3. Testing /auth/me...")
resp = requests.get(f"{BASE_URL}/auth/me", headers=headers)
if resp.status_code != 200:
    print(f"/auth/me failed: {resp.text}")
    sys.exit(1)
print(f"/auth/me successful: {resp.json()}")

# 4. Check /history (should be empty initially)
print("\n4. Testing GET /history...")
resp = requests.get(f"{BASE_URL}/history", headers=headers)
if resp.status_code != 200:
    print(f"GET /history failed: {resp.text}")
    sys.exit(1)
print(f"GET /history successful: {resp.json()}")

print("\nAll tests passed successfully!")
