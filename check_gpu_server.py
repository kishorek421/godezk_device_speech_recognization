"""Check remote GPU server health and model provenance."""
import requests
import sys

REMOTE_GPU_API = "http://124.123.89.162:8001"

def check_health():
    try:
        response = requests.get(f"{REMOTE_GPU_API}/health", timeout=5)
        print(f"Health endpoint status: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except requests.Timeout:
        print("Health endpoint: TIMEOUT")
        return False
    except requests.ConnectionError:
        print("Health endpoint: CONNECTION ERROR")
        return False
    except Exception as e:
        print(f"Health endpoint: ERROR - {e}")
        return False

def check_model_provenance():
    try:
        response = requests.get(f"{REMOTE_GPU_API}/model/provenance", timeout=5)
        print(f"\nModel provenance status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Response: {response.text}")
        return response.status_code == 200
    except requests.Timeout:
        print("Model provenance: TIMEOUT")
        return False
    except requests.ConnectionError:
        print("Model provenance: CONNECTION ERROR")
        return False
    except Exception as e:
        print(f"Model provenance: ERROR - {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("REMOTE GPU SERVER CHECK")
    print(f"Endpoint: {REMOTE_GPU_API}")
    print("=" * 60)
    
    health_ok = check_health()
    model_ok = check_model_provenance()
    
    print("\n" + "=" * 60)
    if health_ok and model_ok:
        print("RESULT: GPU server is accessible and healthy")
        sys.exit(0)
    else:
        print("RESULT: GPU server is NOT accessible or not healthy")
        print("End-to-end ASR validation is BLOCKED")
        sys.exit(1)
