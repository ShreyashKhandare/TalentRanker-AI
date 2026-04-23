#!/usr/bin/env python3
"""
Quick test for live TalentRanker-AI deployment
"""
import requests
import json

def test_live_deployment(base_url):
    """Test the live deployment"""
    print("🚀 Testing live TalentRanker-AI deployment...")
    
    # Test 1: Health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"✅ Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health test failed: {e}")
    
    # Test 2: Root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"✅ Root: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Root test failed: {e}")
    
    # Test 3: Rank endpoint
    try:
        test_data = {
            "resume": "Experienced software engineer with Python and machine learning expertise",
            "jobs": [
                "Senior Python Developer needed for ML project",
                "Junior Java Developer position available", 
                "Data Scientist role requiring Python skills"
            ]
        }
        response = requests.post(f"{base_url}/rank", json=test_data, timeout=30)
        result = response.json()
        print(f"✅ Rank: {response.status_code}")
        print(f"📊 Ranked {len(result.get('ranked_jobs', []))} jobs successfully")
        print(f"🏆 Top match: {result.get('ranked_jobs', [{}])[0].get('job', 'N/A')}")
    except Exception as e:
        print(f"❌ Rank test failed: {e}")

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:10000"
    test_live_deployment(url)
