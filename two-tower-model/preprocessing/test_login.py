import requests

API_KEY = "ff54d254c2b14bad850b6df24c5dacbf"
BASE_URL = "http://localhost:25600"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

response = requests.get(f"{BASE_URL}/api/v1/libraries", headers=headers)

if response.status_code == 200:
    libraries = response.json()
    print("📚 Libraries:")
    for lib in libraries:
        print(f"- {lib['name']} (ID: {lib['id']})")
else:
    print(f"❌ Gagal ambil libraries: {response.status_code}")
    print(response.text)
