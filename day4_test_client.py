#!/usr/bin/env python3
"""
day4_test_client.py
Test client for Day 4 FastAPI backend
Comprehensive testing of all API endpoints
"""

import httpx
import asyncio
import time
import json
from pathlib import Path
import tempfile
import os

class Day4APIClient:
    """Test client for Day 4 FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout
        
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    async def test_health_check(self):
        """Test the health check endpoint"""
        print("🏥 Testing health check...")
        
        response = await self.client.get(f"{self.base_url}/health")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health: {health_data['status']}")
            print(f"   ⏰ Uptime: {health_data['uptime_seconds']:.1f}s")
            print(f"   👷 Active jobs: {health_data['active_jobs']}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    
    async def test_root_endpoint(self):
        """Test the root endpoint"""
        print("\n🏠 Testing root endpoint...")
        
        response = await self.client.get(f"{self.base_url}/")
        
        if response.status_code == 200:
            root_data = response.json()
            print(f"   ✅ API: {root_data['message']}")
            print(f"   📋 Version: {root_data['version']}")
            print(f"   📅 Day: {root_data['day']}")
            return True
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
            return False
    
    def create_test_files(self):
        """Create test DBC and ASC files"""
        
        # Test DBC content
        dbc_content = """VERSION ""

NS_ : 
    NS_DESC_
    CM_
    BA_DEF_
    BA_
    VAL_
    CAT_DEF_
    CAT_
    FILTER
    BA_DEF_DEF_
    EV_DATA_
    ENVVAR_DATA_
    SGTYPE_
    SGTYPE_VAL_
    BA_DEF_SGTYPE_
    BA_SGTYPE_
    SIG_VALTYPE_
    SIGTYPE_VALTYPE_
    BO_TX_BU_
    BA_DEF_REL_
    BA_REL_
    BA_DEF_DEF_REL_
    BU_SG_REL_
    BU_EV_REL_
    BU_BO_REL_
    SG_MUL_VAL_

BS_:

BU_: ECU1 ECU2 Gateway

BO_ 1234 EngineData: 8 ECU1
 SG_ RPM : 0|16@1+ (0.25,0) [0|16383.75] "rpm" ECU2
 SG_ Throttle : 16|8@1+ (0.4,0) [0|102] "%" ECU2
 SG_ Temperature : 24|8@1+ (1,-40) [-40|215] "C" ECU2

BO_ 1235 VehicleSpeed: 8 ECU2  
 SG_ Speed : 0|16@1+ (0.1,0) [0|6553.5] "km/h" Gateway
 SG_ GatewayRPM : 16|16@1+ (0.25,0) [0|16383.75] "rpm" Gateway
"""
        
        # Test ASC content with correlated signals
        asc_content = """date Wed Nov 15 14:30:00 2023
base hex  timestamps absolute
internal events logged
// version 9.0.0
Begin Triggerblock Wed Nov 15 14:30:00 2023
   0.000000 1  4D2             Rx   d 8  10 00 40 14 00 00 12 34
   0.100000 1  4D2             Rx   d 8  11 00 41 15 00 00 12 35
   0.200000 1  4D2             Rx   d 8  12 00 42 16 00 00 12 36
   0.300000 1  4D2             Rx   d 8  13 00 43 17 00 00 12 37
   0.400000 1  4D2             Rx   d 8  14 00 44 18 00 00 12 38
   0.050000 1  4D3             Rx   d 8  C8 00 10 00 00 00 00 00
   0.150000 1  4D3             Rx   d 8  D2 00 11 00 00 00 00 00
   0.250000 1  4D3             Rx   d 8  DC 00 12 00 00 00 00 00
   0.350000 1  4D3             Rx   d 8  E6 00 13 00 00 00 00 00
   0.450000 1  4D3             Rx   d 8  F0 00 14 00 00 00 00 00
End TriggerBlock
"""
        
        # Create temporary files
        files = {}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dbc', delete=False) as f:
            f.write(dbc_content)
            files['dbc'] = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.asc', delete=False) as f:
            f.write(asc_content)
            files['asc'] = f.name
        
        return files
    
    async def test_file_upload_and_analysis(self):
        """Test file upload and analysis workflow"""
        print("\n📁 Testing file upload and analysis...")
        
        # Create test files
        test_files = self.create_test_files()
        
        try:
            # Prepare files for upload
            files = []
            
            # Upload DBC file
            with open(test_files['dbc'], 'rb') as f:
                dbc_content = f.read()
            files.append(('files', ('test.dbc', dbc_content, 'application/octet-stream')))
            
            # Upload ASC file
            with open(test_files['asc'], 'rb') as f:
                asc_content = f.read()
            files.append(('files', ('test.asc', asc_content, 'application/octet-stream')))
            
            print(f"   📤 Uploading {len(files)} files...")
            
            # Start analysis
            response = await self.client.post(f"{self.base_url}/analyze", files=files)
            
            if response.status_code != 200:
                print(f"   ❌ Upload failed: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                return False
            
            analysis_response = response.json()
            job_id = analysis_response['job_id']
            
            print(f"   ✅ Analysis started: {job_id}")
            print(f"   📋 Status: {analysis_response['status']}")
            
            return job_id
            
        finally:
            # Cleanup temp files
            for filepath in test_files.values():
                try:
                    os.unlink(filepath)
                except:
                    pass
    
    async def test_job_status_monitoring(self, job_id: str):
        """Test job status monitoring"""
        print(f"\n📊 Monitoring job status: {job_id}")
        
        max_wait_time = 120  # 2 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            response = await self.client.get(f"{self.base_url}/jobs/{job_id}/status")
            
            if response.status_code != 200:
                print(f"   ❌ Status check failed: {response.status_code}")
                return False
            
            status_data = response.json()
            status = status_data['status']
            progress = status_data['progress']
            message = status_data['message']
            
            print(f"   📈 Progress: {progress:.1f}% - {status} - {message}")
            
            if status == 'completed':
                print(f"   ✅ Job completed successfully!")
                return True
            elif status == 'failed':
                error = status_data.get('error', 'Unknown error')
                print(f"   ❌ Job failed: {error}")
                return False
            
            # Wait before next check
            await asyncio.sleep(2)
        
        print(f"   ⏰ Job timeout after {max_wait_time} seconds")
        return False
    
    async def test_results_retrieval(self, job_id: str):
        """Test results retrieval"""
        print(f"\n📊 Testing results retrieval: {job_id}")
        
        response = await self.client.get(f"{self.base_url}/jobs/{job_id}/results")
        
        if response.status_code != 200:
            print(f"   ❌ Results retrieval failed: {response.status_code}")
            return False
        
        results_data = response.json()
        
        print(f"   ✅ Results retrieved successfully!")
        print(f"   📊 Signals: {results_data['signal_count']}")
        print(f"   🔗 Correlations: {results_data['correlation_count']}")
        print(f"   🚪 Gateway paths: {results_data['gateway_paths']}")
        print(f"   ⏱️  Processing time: {results_data['processing_time']:.2f}s")
        print(f"   📁 Download links: {len(results_data['download_links'])}")
        
        return results_data
    
    async def test_file_download(self, job_id: str, download_links: dict):
        """Test file download"""
        print(f"\n⬇️  Testing file downloads: {job_id}")
        
        downloaded_files = 0
        
        for file_type, download_url in download_links.items():
            full_url = f"{self.base_url}{download_url}"
            
            try:
                response = await self.client.get(full_url)
                
                if response.status_code == 200:
                    file_size = len(response.content)
                    print(f"   ✅ Downloaded {file_type}: {file_size} bytes")
                    downloaded_files += 1
                else:
                    print(f"   ❌ Download failed {file_type}: {response.status_code}")
            
            except Exception as e:
                print(f"   ❌ Download error {file_type}: {e}")
        
        print(f"   📁 Successfully downloaded {downloaded_files}/{len(download_links)} files")
        return downloaded_files == len(download_links)
    
    async def test_job_listing(self):
        """Test job listing endpoint"""
        print(f"\n📋 Testing job listing...")
        
        response = await self.client.get(f"{self.base_url}/jobs")
        
        if response.status_code != 200:
            print(f"   ❌ Job listing failed: {response.status_code}")
            return False
        
        jobs = response.json()
        print(f"   ✅ Found {len(jobs)} jobs")
        
        for job in jobs[:3]:  # Show first 3 jobs
            print(f"   📄 Job {job['job_id'][:8]}... - {job['status']} - {job['progress']:.1f}%")
        
        return True
    
    async def test_system_stats(self):
        """Test system statistics endpoint"""
        print(f"\n📈 Testing system statistics...")
        
        response = await self.client.get(f"{self.base_url}/stats")
        
        if response.status_code != 200:
            print(f"   ❌ Stats failed: {response.status_code}")
            return False
        
        stats = response.json()
        
        print(f"   ✅ System stats retrieved")
        print(f"   ⏰ Uptime: {stats['system']['uptime_seconds']:.1f}s")
        print(f"   👷 Total jobs: {stats['jobs']['total']}")
        print(f"   ✅ Completed: {stats['jobs']['completed']}")
        print(f"   ❌ Failed: {stats['jobs']['failed']}")
        print(f"   🔄 Active: {stats['jobs']['active']}")
        print(f"   💾 Storage: {stats['storage']['total_bytes']:,} bytes")
        
        return True
    
    async def test_job_deletion(self, job_id: str):
        """Test job deletion"""
        print(f"\n🗑️  Testing job deletion: {job_id}")
        
        response = await self.client.delete(f"{self.base_url}/jobs/{job_id}")
        
        if response.status_code != 200:
            print(f"   ❌ Job deletion failed: {response.status_code}")
            return False
        
        delete_response = response.json()
        print(f"   ✅ Job deleted: {delete_response['message']}")
        
        # Verify job is gone
        response = await self.client.get(f"{self.base_url}/jobs/{job_id}/status")
        if response.status_code == 404:
            print(f"   ✅ Job successfully removed from system")
            return True
        else:
            print(f"   ❌ Job still exists after deletion")
            return False
    
    async def run_complete_test_suite(self):
        """Run the complete test suite"""
        print("🧪 Day 4 FastAPI Backend - Complete Test Suite")
        print("=" * 60)
        
        test_results = []
        
        # Test 1: Health check
        result = await self.test_health_check()
        test_results.append(("Health Check", result))
        
        # Test 2: Root endpoint
        result = await self.test_root_endpoint()
        test_results.append(("Root Endpoint", result))
        
        # Test 3: File upload and analysis
        job_id = await self.test_file_upload_and_analysis()
        if job_id:
            test_results.append(("File Upload", True))
            
            # Test 4: Job status monitoring
            result = await self.test_job_status_monitoring(job_id)
            test_results.append(("Status Monitoring", result))
            
            if result:  # Only test results if job completed
                # Test 5: Results retrieval
                results_data = await self.test_results_retrieval(job_id)
                if results_data:
                    test_results.append(("Results Retrieval", True))
                    
                    # Test 6: File download
                    result = await self.test_file_download(job_id, results_data['download_links'])
                    test_results.append(("File Download", result))
                else:
                    test_results.append(("Results Retrieval", False))
                    test_results.append(("File Download", False))
            
            # Test 7: Job listing
            result = await self.test_job_listing()
            test_results.append(("Job Listing", result))
            
            # Test 8: System stats
            result = await self.test_system_stats()
            test_results.append(("System Stats", result))
            
            # Test 9: Job deletion
            result = await self.test_job_deletion(job_id)
            test_results.append(("Job Deletion", result))
            
        else:
            # Mark remaining tests as failed if upload failed
            for test_name in ["Status Monitoring", "Results Retrieval", "File Download", 
                            "Job Listing", "System Stats", "Job Deletion"]:
                test_results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name:<20} {status}")
            if result:
                passed += 1
        
        print("\n" + "=" * 60)
        print(f"📈 OVERALL RESULT: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Day 4 FastAPI Backend is working perfectly!")
            print("✅ Ready for Day 10: React Frontend integration!")
        elif passed >= total * 0.8:
            print("⚠️  Most tests passed - minor issues to fix")
        else:
            print("❌ Significant issues found - needs debugging")
        
        return passed == total

async def main():
    """Main test function"""
    
    print("🚀 Starting Day 4 FastAPI Backend Tests")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("Run: python day4_fastapi_main.py")
    print()
    
    client = Day4APIClient()
    
    try:
        success = await client.run_complete_test_suite()
        return success
    
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await client.close()

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)