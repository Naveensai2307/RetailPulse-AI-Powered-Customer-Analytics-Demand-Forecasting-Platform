import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Target configuration
URL = "http://localhost:8501"  # Local dashboard
CONCURRENT_USERS = 10
TOTAL_REQUESTS = 100

def simulate_user(user_id):
    success = 0
    failure = 0
    latencies = []
    
    for _ in range(TOTAL_REQUESTS // CONCURRENT_USERS):
        start_time = time.time()
        try:
            response = requests.get(URL, timeout=5)
            latency = time.time() - start_time
            if response.status_code == 200:
                success += 1
                latencies.append(latency)
            else:
                failure += 1
        except Exception as e:
            failure += 1
            
    return success, failure, latencies

def run_load_test():
    print(f"Starting Load Test on {URL}")
    print(f"Users: {CONCURRENT_USERS} | Total Requests: {TOTAL_REQUESTS}")
    print("-" * 40)
    
    start_all = time.time()
    
    with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        results = list(executor.map(simulate_user, range(CONCURRENT_USERS)))
    
    total_time = time.time() - start_all
    total_success = sum(r[0] for r in results)
    total_failure = sum(r[1] for r in results)
    all_latencies = [l for r in results for l in r[2]]
    
    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    throughput = total_success / total_time
    
    print("\nLOAD TEST RESULTS")
    print(f"Successful Requests: {total_success}")
    print(f"Failed Requests: {total_failure}")
    print(f"Total Execution Time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Avg Latency: {avg_latency*1000:.2f}ms")
    print("-" * 40)

if __name__ == "__main__":
    run_load_test()
