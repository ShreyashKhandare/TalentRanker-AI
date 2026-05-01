from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import re
from urllib.parse import urlparse, parse_qs
import os

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "engine": "simple-text-similarity"}).encode())
        elif self.path == '/':
            try:
                with open('static/index.html', 'r', encoding='utf-8') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(content.encode())
            except FileNotFoundError:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"message": "TalentRanker AI - Working!", "status": "running"}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/rank':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                resume = data.get('resume', '')
                jobs = data.get('jobs', [])
                
                if not resume or not jobs:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Resume and jobs required"}).encode())
                    return
                
                # Calculate similarity scores
                scores = []
                for job in jobs:
                    similarity = simple_text_similarity(resume, job)
                    scores.append(similarity)
                
                # Get the first job's score for match_score
                match_score = scores[0] if scores else 0.0
                
                # Create explanation based on score
                if match_score > 0.7:
                    explanation = f"Strong match with {match_score:.1%} similarity. Resume contains many keywords from the job description."
                elif match_score > 0.4:
                    explanation = f"Moderate match with {match_score:.1%} similarity. Some relevant skills found but could be improved."
                else:
                    explanation = f"Weak match with {match_score:.1%} similarity. Consider updating resume with more relevant keywords."
                
                result = {
                    "status": "success",
                    "match_score": match_score,
                    "explanation": explanation,
                    "ranked_jobs": [
                        {
                            "rank": i + 1,
                            "job": jobs[i][:200] + "..." if len(jobs[i]) > 200 else jobs[i],
                            "similarity_score": round(scores[i], 4),
                            "match_quality": "High" if scores[i] > 0.7 else "Medium" if scores[i] > 0.4 else "Low"
                        }
                        for i in range(len(jobs))
                    ],
                    "engine": "simple-text-similarity",
                    "processed_jobs": len(jobs)
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

def simple_text_similarity(resume: str, job: str) -> float:
    """Simple text similarity without ML dependencies"""
    # Convert to lowercase and split into words
    resume_words = set(re.findall(r'\w+', resume.lower()))
    job_words = set(re.findall(r'\w+', job.lower()))
    
    # Calculate Jaccard similarity
    intersection = len(resume_words & job_words)
    union = len(resume_words | job_words)
    
    if union == 0:
        return 0.0
    
    return intersection / union

def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    print(f"Server running on port {port}")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
