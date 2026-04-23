#!/usr/bin/env python3
"""
Deployment Verification Script for TalentRanker-AI
Tests all endpoints and validates deployment health
"""

import requests
import json
import time
import sys

def test_deployment(base_url="http://localhost:10000"):
    """Test all deployment endpoints"""
    print("🚀 Starting deployment verification...")
    
    results = {}
    
    # Test 1: Health endpoint
    try:
        print("📊 Testing /health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        results['health'] = {
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else None
        }
        print(f"✅ Health: {response.status_code} - {results['health']['response_time']:.3f}s")
    except Exception as e:
        results['health'] = {'error': str(e)}
        print(f"❌ Health test failed: {e}")
    
    # Test 2: Root endpoint
    try:
        print("🏠 Testing / endpoint...")
        response = requests.get(f"{base_url}/", timeout=10)
        results['root'] = {
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else None
        }
        print(f"✅ Root: {response.status_code} - {results['root']['response_time']:.3f}s")
    except Exception as e:
        results['root'] = {'error': str(e)}
        print(f"❌ Root test failed: {e}")
    
    # Test 3: Metrics endpoint
    try:
        print("📈 Testing /metrics endpoint...")
        response = requests.get(f"{base_url}/metrics", timeout=10)
        results['metrics'] = {
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds(),
            'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else None
        }
        print(f"✅ Metrics: {response.status_code} - {results['metrics']['response_time']:.3f}s")
    except Exception as e:
        results['metrics'] = {'error': str(e)}
        print(f"❌ Metrics test failed: {e}")
    
    # Test 4: Rank endpoint
    try:
        print("🎯 Testing /rank endpoint...")
        test_data = {
            "resume": "Experienced software engineer with Python and machine learning expertise",
            "jobs": [
                "Senior Python Developer needed for ML project",
                "Junior Java Developer position available",
                "Data Scientist role requiring Python skills"
            ]
        }
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/rank", 
            json=test_data, 
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        end_time = time.time()
        
        results['rank'] = {
            'status_code': response.status_code,
            'response_time': end_time - start_time,
            'data': response.json() if response.headers.get('content-type', '').startswith('application/json') else None
        }
        print(f"✅ Rank: {response.status_code} - {results['rank']['response_time']:.3f}s")
        
        # Validate ranking results
        if results['rank']['data'] and results['rank']['data'].get('ranked_jobs'):
            ranked_jobs = results['rank']['data']['ranked_jobs']
            print(f"📊 Ranked {len(ranked_jobs)} jobs successfully")
            
            # Check if scores are properly sorted
            scores = [job['similarity_score'] for job in ranked_jobs]
            if scores == sorted(scores, reverse=True):
                print("✅ Rankings are properly sorted")
            else:
                print("⚠️ Rankings may not be properly sorted")
        
    except Exception as e:
        results['rank'] = {'error': str(e)}
        print(f"❌ Rank test failed: {e}")
    
    # Test 5: Docs endpoint
    try:
        print("📚 Testing /docs endpoint...")
        response = requests.get(f"{base_url}/docs", timeout=10)
        results['docs'] = {
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds()
        }
        print(f"✅ Docs: {response.status_code} - {results['docs']['response_time']:.3f}s")
    except Exception as e:
        results['docs'] = {'error': str(e)}
        print(f"❌ Docs test failed: {e}")
    
    # Summary
    print("\n📋 DEPLOYMENT VERIFICATION SUMMARY")
    print("=" * 50)
    
    success_count = 0
    total_tests = len([k for k, v in results.items() if 'error' not in v])
    
    for test_name, result in results.items():
        if 'error' not in result:
            success_count += 1
            status = "✅ PASS"
            details = f"Status: {result.get('status_code', 'N/A')} - Time: {result.get('response_time', 0):.3f}s"
        else:
            status = "❌ FAIL"
            details = f"Error: {result['error']}"
        
        print(f"{test_name.upper().ljust(10)}: {status} - {details}")
    
    print("=" * 50)
    print(f"Overall: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 DEPLOYMENT VERIFICATION: SUCCESS")
        print("✅ TalentRanker-AI is ready for production!")
        return True
    else:
        print("⚠️ DEPLOYMENT VERIFICATION: ISSUES FOUND")
        print("❌ Some endpoints may need attention")
        return False

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:10000"
    success = test_deployment(base_url)
    sys.exit(0 if success else 1)
