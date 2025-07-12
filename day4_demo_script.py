#!/usr/bin/env python3
"""
day4_demo_script.py
Interactive demo of Day 4 FastAPI backend
Shows all API capabilities with real automotive data
"""

import requests
import time
import json
import tempfile
import os
from pathlib import Path

class Day4APIDemo:
    """Interactive demo for Day 4 FastAPI backend"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 300  # 5 minutes
    
    def print_header(self, title):
        """Print a formatted header"""
        print("\n" + "="*60)
        print(f"🚀 {title}")
        print("="*60)
    
    def print_step(self, step_num, title):
        """Print a step header"""
        print(f"\n📋 Step {step_num}: {title}")
        print("-" * 40)
    
    def demo_health_check(self):
        """Demo: Health check endpoint"""
        self.print_step(1, "Health Check")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                health_data = response.json()
                print("✅ API is healthy!")
                print(f"   Status: {health_data['status']}")
                print(f"   Version: {health_data['version']}")
                print(f"   Uptime: {health_data['uptime_seconds']:.1f} seconds")
                print(f"   Active jobs: {health_data['active_jobs']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("Make sure the FastAPI server is running on http://localhost:8000")
            return False
    
    def create_demo_files(self):
        """Create realistic automotive demo files"""
        print("📁 Creating demo automotive files...")
        
        # Realistic DBC file with multiple ECUs
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

BU_: EngineECU TransmissionECU BodyECU Gateway

BO_ 256 EngineData: 8 EngineECU
 SG_ EngineRPM : 0|16@1+ (0.25,0) [0|16383.75] "rpm" Gateway,BodyECU
 SG_ ThrottlePosition : 16|8@1+ (0.4,0) [0|102] "%" Gateway,BodyECU
 SG_ CoolantTemp : 24|8@1+ (1,-40) [-40|215] "C" Gateway,BodyECU
 SG_ EngineLoad : 32|8@1+ (0.4,0) [0|102] "%" Gateway

BO_ 257 TransmissionData: 8 TransmissionECU
 SG_ GearPosition : 0|4@1+ (1,0) [0|15] "" Gateway,BodyECU
 SG_ TransmissionTemp : 4|8@1+ (1,-40) [-40|215] "C" Gateway
 SG_ TorqueConverterLockup : 12|1@1+ (1,0) [0|1] "" Gateway

BO_ 512 VehicleSpeed: 8 Gateway
 SG_ VehicleSpeed : 0|16@1+ (0.1,0) [0|6553.5] "km/h" EngineECU,TransmissionECU,BodyECU
 SG_ WheelSpeedFL : 16|16@1+ (0.1,0) [0|6553.5] "km/h" BodyECU
 SG_ WheelSpeedFR : 32|16@1+ (0.1,0) [0|6553.5] "km/h" BodyECU
 SG_ WheelSpeedRL : 48|16@1+ (0.1,0) [0|6553.5] "km/h" BodyECU

BO_ 513 GatewayRelay: 8 Gateway
 SG_ RelayedRPM : 0|16@1+ (0.25,0) [0|16383.75] "rpm" BodyECU
 SG_ RelayedThrottle : 16|8@1+ (0.4,0) [0|102] "%" BodyECU
 SG_ RelayedGear : 24|4@1+ (1,0) [0|15] "" BodyECU

BO_ 768 BodyControl: 8 BodyECU
 SG_ DoorStatus : 0|4@1+ (1,0) [0|15] "" Gateway
 SG_ LightStatus : 4|4@1+ (1,0) [0|15] "" Gateway
 SG_ WindowStatus : 8|4@1+ (1,0) [0|15] "" Gateway
"""
        
        # Realistic ASC log with correlated gateway signals
        asc_content = """date Wed Nov 15 14:30:00 2023
base hex  timestamps absolute
internal events logged
// Universal Network Analyzer Demo Log
Begin Triggerblock Wed Nov 15 14:30:00 2023
"""
        
        # Generate realistic driving scenario data
        import math
        
        for i in range(100):  # 10 seconds of data
            t = i * 0.1  # 100ms intervals
            
            # Simulate realistic driving scenario
            rpm = int(1000 + 800 * math.sin(t * 0.5))  # 1000-1800 RPM
            throttle = int(30 + 20 * math.sin(t * 0.3))  # 30-50% throttle
            coolant_temp = int(85 + 5 * math.sin(t * 0.1))  # 80-90°C
            engine_load = int(40 + 15 * math.sin(t * 0.4))  # 25-55% load
            
            vehicle_speed = int(50 + 30 * math.sin(t * 0.2))  # 20-80 km/h
            gear = min(5, max(1, int(3 + 2 * math.sin(t * 0.15))))  # Gears 1-5
            
            # Convert to hex bytes for DBC signals
            rpm_raw = int(rpm / 0.25)
            throttle_raw = int(throttle / 0.4)
            speed_raw = int(vehicle_speed / 0.1)
            
            rpm_bytes = f"{rpm_raw & 0xFF:02X} {(rpm_raw >> 8) & 0xFF:02X}"
            throttle_byte = f"{throttle_raw:02X}"
            coolant_byte = f"{coolant_temp + 40:02X}"
            load_byte = f"{int(engine_load / 0.4):02X}"
            
            speed_bytes = f"{speed_raw & 0xFF:02X} {(speed_raw >> 8) & 0xFF:02X}"
            
            # Engine data (256 = 0x100)
            asc_content += f"   {t:.6f} 1  100             Rx   d 8  {rpm_bytes} {throttle_byte} {coolant_byte} {load_byte} 00 00 00\n"
            
            # Transmission data (257 = 0x101)
            trans_temp = 70 + int(10 * math.sin(t * 0.05))
            asc_content += f"   {t + 0.01:.6f} 1  101             Rx   d 8  {gear:02X} {trans_temp + 40:02X} 00 00 00 00 00 00\n"
            
            # Vehicle speed from gateway (512 = 0x200) - slight delay
            wheel_speeds = speed_raw + int(10 * math.sin(t))  # Slight variation
            ws_bytes = f"{wheel_speeds & 0xFF:02X} {(wheel_speeds >> 8) & 0xFF:02X}"
            asc_content += f"   {t + 0.05:.6f} 1  200             Rx   d 8  {speed_bytes} {ws_bytes} {ws_bytes} {ws_bytes}\n"
            
            # Gateway relay (513 = 0x201) - correlated with engine data but delayed
            relay_rpm = rpm_raw  # Should correlate perfectly with engine RPM
            relay_throttle = throttle_raw  # Should correlate with throttle
            relay_rpm_bytes = f"{relay_rpm & 0xFF:02X} {(relay_rpm >> 8) & 0xFF:02X}"
            asc_content += f"   {t + 0.1:.6f} 1  201             Rx   d 8  {relay_rpm_bytes} {relay_throttle:02X} {gear:02X} 00 00 00 00\n"
        
        asc_content += "End TriggerBlock\n"
        
        # Create temporary files
        files = {}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dbc', delete=False) as f:
            f.write(dbc_content)
            files['dbc'] = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.asc', delete=False) as f:
            f.write(asc_content)
            files['asc'] = f.name
        
        print(f"   ✅ Created realistic automotive DBC file")
        print(f"   ✅ Created ASC log with 100 samples over 10 seconds")
        print(f"   📊 Includes: Engine, Transmission, Gateway, Body ECUs")
        print(f"   🔗 Contains correlated signals for gateway analysis")
        
        return files
    
    def demo_file_upload(self, files):
        """Demo: File upload and analysis start"""
        self.print_step(2, "File Upload & Analysis Start")
        
        try:
            # Prepare files for upload
            file_data = []
            
            with open(files['dbc'], 'rb') as f:
                file_data.append(('files', ('demo_vehicle.dbc', f.read(), 'application/octet-stream')))
            
            with open(files['asc'], 'rb') as f:
                file_data.append(('files', ('demo_log.asc', f.read(), 'application/octet-stream')))
            
            print("📤 Uploading automotive files to API...")
            print("   - demo_vehicle.dbc (CAN database)")
            print("   - demo_log.asc (CAN trace log)")
            
            response = self.session.post(f"{self.base_url}/analyze", files=file_data)
            
            if response.status_code == 200:
                result = response.json()
                job_id = result['job_id']
                
                print("✅ Files uploaded successfully!")
                print(f"   Job ID: {job_id}")
                print(f"   Status: {result['status']}")
                print(f"   Message: {result['message']}")
                
                return job_id
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return None
    
    def demo_progress_monitoring(self, job_id):
        """Demo: Real-time progress monitoring"""
        self.print_step(3, "Real-time Progress Monitoring")
        
        print(f"📊 Monitoring analysis progress for job: {job_id}")
        print("   (This shows the power of background processing)")
        print()
        
        start_time = time.time()
        last_progress = -1
        
        while True:
            try:
                response = self.session.get(f"{self.base_url}/jobs/{job_id}/status")
                
                if response.status_code == 200:
                    status_data = response.json()
                    
                    status = status_data['status']
                    progress = status_data['progress']
                    message = status_data['message']
                    
                    # Only print when progress changes significantly
                    if progress - last_progress >= 10 or status in ['completed', 'failed']:
                        elapsed = time.time() - start_time
                        print(f"   [{elapsed:5.1f}s] {progress:5.1f}% - {status.upper()} - {message}")
                        last_progress = progress
                    
                    if status == 'completed':
                        print("✅ Analysis completed successfully!")
                        return True
                    elif status == 'failed':
                        error = status_data.get('error', 'Unknown error')
                        print(f"❌ Analysis failed: {error}")
                        return False
                    
                else:
                    print(f"❌ Status check failed: {response.status_code}")
                    return False
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                return False
    
    def demo_results_analysis(self, job_id):
        """Demo: Results retrieval and analysis"""
        self.print_step(4, "Results Analysis")
        
        try:
            response = self.session.get(f"{self.base_url}/jobs/{job_id}/results")
            
            if response.status_code == 200:
                results = response.json()
                
                print("📊 Analysis Results Summary:")
                print(f"   🎯 Signals processed: {results['signal_count']:,}")
                print(f"   🔗 Correlations found: {results['correlation_count']}")
                print(f"   🚪 Gateway paths: {results['gateway_paths']}")
                print(f"   ⏱️  Processing time: {results['processing_time']:.2f} seconds")
                
                # Show metadata
                metadata = results['analysis_metadata']
                print(f"\n📋 Technical Details:")
                print(f"   🌐 Protocols analyzed: {', '.join(metadata['protocols_analyzed'])}")
                print(f"   📅 Analysis timestamp: {metadata['analysis_timestamp']}")
                
                # Show available downloads
                downloads = results['download_links']
                print(f"\n📁 Available Downloads ({len(downloads)} files):")
                for file_type, download_url in downloads.items():
                    print(f"   📄 {file_type}: {download_url}")
                
                return results
            else:
                print(f"❌ Results retrieval failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Results error: {e}")
            return None
    
    def demo_file_downloads(self, job_id, download_links):
        """Demo: File download capabilities"""
        self.print_step(5, "File Downloads")
        
        print("⬇️  Testing file download capabilities...")
        
        downloaded_files = {}
        
        for file_type, download_url in download_links.items():
            try:
                full_url = f"{self.base_url}{download_url}"
                response = self.session.get(full_url)
                
                if response.status_code == 200:
                    file_size = len(response.content)
                    downloaded_files[file_type] = response.content
                    
                    print(f"   ✅ {file_type}: {file_size:,} bytes")
                    
                    # Show preview for text files
                    if file_type in ['report', 'executive_report']:
                        preview = response.text[:200] + "..." if len(response.text) > 200 else response.text
                        print(f"      Preview: {preview}")
                else:
                    print(f"   ❌ {file_type}: Download failed ({response.status_code})")
                    
            except Exception as e:
                print(f"   ❌ {file_type}: Error - {e}")
        
        print(f"\n📦 Successfully downloaded {len(downloaded_files)}/{len(download_links)} files")
        return downloaded_files
    
    def demo_system_monitoring(self):
        """Demo: System monitoring and statistics"""
        self.print_step(6, "System Monitoring")
        
        try:
            # Get system stats
            response = self.session.get(f"{self.base_url}/stats")
            
            if response.status_code == 200:
                stats = response.json()
                
                print("📊 System Performance Dashboard:")
                
                # System info
                system = stats['system']
                uptime_hours = system['uptime_seconds'] / 3600
                print(f"   🖥️  System uptime: {uptime_hours:.2f} hours")
                print(f"   🔢 API version: {system['version']}")
                
                # Job statistics
                jobs = stats['jobs']
                success_rate = (jobs['completed'] / jobs['total'] * 100) if jobs['total'] > 0 else 0
                print(f"\n   📊 Job Statistics:")
                print(f"      Total jobs: {jobs['total']}")
                print(f"      Completed: {jobs['completed']} ({success_rate:.1f}% success rate)")
                print(f"      Failed: {jobs['failed']}")
                print(f"      Currently active: {jobs['active']}")
                
                # Storage info
                storage = stats['storage']
                total_mb = storage['total_bytes'] / (1024 * 1024)
                uploads_mb = storage['uploads_bytes'] / (1024 * 1024)
                results_mb = storage['results_bytes'] / (1024 * 1024)
                
                print(f"\n   💾 Storage Usage:")
                print(f"      Total: {total_mb:.1f} MB")
                print(f"      Uploads: {uploads_mb:.1f} MB")
                print(f"      Results: {results_mb:.1f} MB")
                
                return True
            else:
                print(f"❌ Stats retrieval failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            return False
    
    def demo_job_management(self, job_id):
        """Demo: Job management capabilities"""
        self.print_step(7, "Job Management")
        
        try:
            # List all jobs
            response = self.session.get(f"{self.base_url}/jobs")
            
            if response.status_code == 200:
                jobs = response.json()
                
                print(f"📋 Job Management Dashboard ({len(jobs)} total jobs):")
                
                # Show recent jobs
                for i, job in enumerate(jobs[:5]):  # Show last 5 jobs
                    created_time = job['created_at'][:19]  # Remove microseconds
                    print(f"   {i+1}. Job {job['job_id'][:8]}... ({created_time})")
                    print(f"      Status: {job['status']} - {job['progress']:.1f}%")
                    print(f"      Message: {job['message']}")
                
                if len(jobs) > 5:
                    print(f"   ... and {len(jobs) - 5} more jobs")
                
                # Demonstrate job deletion (optional)
                print(f"\n🗑️  Job Cleanup Capability:")
                print(f"   Current job {job_id[:8]}... can be deleted when no longer needed")
                print(f"   (Skipping deletion for demo purposes)")
                
                return True
            else:
                print(f"❌ Job listing failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Job management error: {e}")
            return False
    
    def cleanup_demo_files(self, files):
        """Clean up temporary demo files"""
        print("\n🧹 Cleaning up demo files...")
        
        for file_type, filepath in files.items():
            try:
                os.unlink(filepath)
                print(f"   ✅ Deleted {file_type} file")
            except:
                print(f"   ⚠️  Could not delete {file_type} file")
    
    def run_complete_demo(self):
        """Run the complete interactive demo"""
        self.print_header("Day 4 FastAPI Backend - Interactive Demo")
        
        print("This demo showcases the complete FastAPI backend capabilities:")
        print("• File upload with multi-protocol support")
        print("• Background job processing")
        print("• Real-time progress monitoring")
        print("• Advanced automotive network analysis")
        print("• Professional results and reporting")
        print("• System monitoring and management")
        
        input("\nPress Enter to start the demo...")
        
        demo_results = []
        demo_files = None
        job_id = None
        
        try:
            # Step 1: Health check
            result = self.demo_health_check()
            demo_results.append(("Health Check", result))
            
            if not result:
                print("\n❌ Cannot continue - API is not responding")
                return False
            
            # Step 2: Create and upload files
            demo_files = self.create_demo_files()
            job_id = self.demo_file_upload(demo_files)
            demo_results.append(("File Upload", job_id is not None))
            
            if not job_id:
                print("\n❌ Cannot continue - file upload failed")
                return False
            
            # Step 3: Monitor progress
            result = self.demo_progress_monitoring(job_id)
            demo_results.append(("Progress Monitoring", result))
            
            if not result:
                print("\n❌ Analysis failed")
                return False
            
            # Step 4: Retrieve results
            results = self.demo_results_analysis(job_id)
            demo_results.append(("Results Analysis", results is not None))
            
            if results:
                # Step 5: Download files
                downloaded = self.demo_file_downloads(job_id, results['download_links'])
                demo_results.append(("File Downloads", len(downloaded) > 0))
            
            # Step 6: System monitoring
            result = self.demo_system_monitoring()
            demo_results.append(("System Monitoring", result))
            
            # Step 7: Job management
            result = self.demo_job_management(job_id)
            demo_results.append(("Job Management", result))
            
        except KeyboardInterrupt:
            print("\n\n⏸️  Demo interrupted by user")
            return False
        
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            return False
        
        finally:
            # Cleanup
            if demo_files:
                self.cleanup_demo_files(demo_files)
        
        # Demo summary
        self.print_header("Demo Results Summary")
        
        passed = 0
        total = len(demo_results)
        
        for demo_name, result in demo_results:
            status = "✅ SUCCESS" if result else "❌ FAILED"
            print(f"   {demo_name:<20} {status}")
            if result:
                passed += 1
        
        print(f"\n📊 Demo Score: {passed}/{total} components working ({(passed/total)*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 PERFECT! Day 4 FastAPI Backend is fully operational!")
            print("✅ All enterprise features working flawlessly")
            print("🚀 Ready for production deployment")
            print("🎯 Ready for Day 10: React Frontend integration")
        elif passed >= total * 0.8:
            print("\n⚠️  Mostly working - minor issues detected")
            print("🔧 Fix remaining issues before production")
        else:
            print("\n❌ Significant issues found")
            print("🚨 Requires debugging before proceeding")
        
        print(f"\n📋 Next Steps:")
        print(f"   1. Day 6: Advanced file upload (large files, progress)")
        print(f"   2. Day 9: Performance optimization")
        print(f"   3. Day 10: React frontend development")
        print(f"   4. Day 14: Production deployment")
        
        return passed == total

def main():
    """Main demo function"""
    print("🚀 Day 4 FastAPI Backend - Interactive Demo")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("Start with: python day4_fastapi_main.py")
    
    demo = Day4APIDemo()
    
    try:
        success = demo.run_complete_demo()
        return success
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)