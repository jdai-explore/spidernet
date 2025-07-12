#!/usr/bin/env python3
"""
day4_minimal_analyzer.py
Minimal version of Day 4 FastAPI backend without heavy dependencies
Works on Windows without C++ build tools
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import time
import uuid
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Universal Network Analyzer API (Minimal)",
    description="Day 4 - Minimal FastAPI backend for Windows compatibility",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
analysis_jobs = {}
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Models
class AnalysisJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: datetime

class AnalysisStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class MinimalResults(BaseModel):
    job_id: str
    files_processed: int
    analysis_time: float
    mock_signal_count: int
    mock_correlation_count: int
    download_links: Dict[str, str]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    message: str

app_start_time = time.time()

def simple_can_parser(dbc_content: str, asc_content: str) -> Dict[str, Any]:
    """
    Minimal CAN parser without cantools dependency
    Extracts basic info from DBC and ASC files
    """
    results = {
        "dbc_messages": [],
        "asc_samples": 0,
        "mock_signals": [],
        "mock_correlations": []
    }
    
    # Parse DBC messages (simple regex-like parsing)
    if "BO_" in dbc_content:
        lines = dbc_content.split('\n')
        for line in lines:
            if line.strip().startswith('BO_'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        msg_id = int(parts[1])
                        msg_name = parts[2].rstrip(':')
                        results["dbc_messages"].append({
                            "id": msg_id,
                            "name": msg_name
                        })
                    except:
                        pass
            elif " SG_ " in line:
                parts = line.split()
                if len(parts) >= 2:
                    signal_name = parts[1]
                    results["mock_signals"].append(signal_name)
    
    # Count ASC samples
    if asc_content:
        lines = asc_content.split('\n')
        for line in lines:
            if line.strip() and not line.startswith(('date', 'base', '//', 'Begin', 'End', 'internal')):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        float(parts[0])  # Try to parse timestamp
                        results["asc_samples"] += 1
                    except:
                        pass
    
    # Generate mock correlations
    if len(results["mock_signals"]) >= 2:
        results["mock_correlations"] = [
            f"{results['mock_signals'][0]} <-> {results['mock_signals'][1]}"
        ]
    
    return results

async def process_minimal_analysis(job_id: str, files: Dict[str, str]):
    """Minimal analysis processing without heavy dependencies"""
    try:
        analysis_jobs[job_id].update({
            "status": "processing",
            "progress": 20.0,
            "message": "Starting minimal analysis..."
        })
        
        # Read uploaded files
        dbc_content = ""
        asc_content = ""
        
        for filename, filepath in files.items():
            if filename.endswith('.dbc'):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    dbc_content = f.read()
            elif filename.endswith('.asc'):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    asc_content = f.read()
        
        analysis_jobs[job_id].update({
            "progress": 50.0,
            "message": "Parsing automotive files..."
        })
        
        # Simple parsing
        results = simple_can_parser(dbc_content, asc_content)
        
        analysis_jobs[job_id].update({
            "progress": 80.0,
            "message": "Generating results..."
        })
        
        # Create mock result files
        results_dir = RESULTS_DIR / job_id
        results_dir.mkdir(exist_ok=True)
        
        # Generate summary report
        summary = {
            "job_id": job_id,
            "analysis_type": "minimal",
            "files_processed": len(files),
            "messages_found": len(results["dbc_messages"]),
            "asc_samples": results["asc_samples"],
            "signals_found": len(results["mock_signals"]),
            "correlations_found": len(results["mock_correlations"]),
            "processing_time": time.time() - analysis_jobs[job_id]["start_time"]
        }
        
        summary_file = results_dir / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate detailed report
        report_lines = [
            "🚀 MINIMAL NETWORK ANALYZER REPORT",
            "=" * 40,
            f"Job ID: {job_id}",
            f"Analysis Time: {summary['processing_time']:.2f} seconds",
            f"Files Processed: {summary['files_processed']}",
            "",
            "📊 DBC ANALYSIS:",
            f"  Messages Found: {summary['messages_found']}",
            f"  Signals Found: {summary['signals_found']}",
            "",
            "📈 ASC ANALYSIS:",
            f"  Samples Found: {summary['asc_samples']}",
            "",
            "🔗 CORRELATIONS:",
            f"  Mock Correlations: {summary['correlations_found']}",
            "",
            "📋 DETECTED MESSAGES:"
        ]
        
        for msg in results["dbc_messages"][:10]:  # First 10 messages
            report_lines.append(f"  - {msg['name']} (ID: 0x{msg['id']:X})")
        
        if results["mock_signals"]:
            report_lines.extend([
                "",
                "🎯 DETECTED SIGNALS:"
            ])
            for signal in results["mock_signals"][:10]:  # First 10 signals
                report_lines.append(f"  - {signal}")
        
        report_lines.extend([
            "",
            "✅ MINIMAL ANALYSIS COMPLETE",
            "Note: This is a lightweight analysis for Windows compatibility.",
            "Full analysis requires Day 1-3 components with numpy/pandas."
        ])
        
        report_file = results_dir / "analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Success
        analysis_jobs[job_id].update({
            "status": "completed",
            "progress": 100.0,
            "message": "Minimal analysis completed",
            "completed_at": datetime.now(),
            "results": summary,
            "exported_files": {
                "summary": str(summary_file),
                "report": str(report_file)
            }
        })
        
        logger.info(f"✅ Minimal analysis completed: {job_id}")
        
    except Exception as e:
        logger.error(f"❌ Minimal analysis failed: {job_id} - {e}")
        analysis_jobs[job_id].update({
            "status": "failed",
            "progress": 0.0,
            "message": f"Analysis failed: {str(e)}",
            "completed_at": datetime.now(),
            "error": str(e)
        })

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0-minimal",
        message="Minimal FastAPI backend running (Windows compatible)"
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Universal Network Analyzer API - Minimal Version",
        "version": "1.0.0-minimal",
        "compatibility": "Windows-optimized without heavy dependencies",
        "features": [
            "File upload and basic parsing",
            "Background job processing",
            "Simple CAN DBC/ASC analysis",
            "Mock correlation detection",
            "JSON and text report generation"
        ],
        "limitations": [
            "No advanced signal processing (requires numpy/pandas)",
            "No statistical correlation analysis",
            "No complex gateway latency measurement"
        ],
        "next_steps": [
            "Install Visual Studio Build Tools for full functionality",
            "Or use Docker for complete analysis capabilities"
        ]
    }

@app.post("/analyze", response_model=AnalysisJobResponse)
async def start_minimal_analysis(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Start minimal analysis (Windows compatible)"""
    
    # Validate files
    supported_files = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext in ['.dbc', '.asc', '.txt', '.log']:
            supported_files.append(file)
    
    if not supported_files:
        raise HTTPException(
            status_code=400,
            detail="No supported files found. Supported: .dbc, .asc, .txt, .log"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    # Save files
    saved_files = {}
    for file in supported_files:
        file_path = job_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        saved_files[file.filename] = str(file_path)
    
    # Initialize job
    analysis_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "message": f"Queued minimal analysis with {len(saved_files)} files",
        "created_at": datetime.now(),
        "start_time": time.time(),
        "files": saved_files
    }
    
    # Start background processing
    background_tasks.add_task(process_minimal_analysis, job_id, saved_files)
    
    logger.info(f"🚀 Started minimal analysis: {job_id}")
    
    return AnalysisJobResponse(
        job_id=job_id,
        status="queued",
        message=f"Minimal analysis started with {len(saved_files)} files",
        created_at=datetime.now()
    )

@app.get("/jobs/{job_id}/status", response_model=AnalysisStatus)
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = analysis_jobs[job_id]
    return AnalysisStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        error=job.get("error")
    )

@app.get("/jobs/{job_id}/results", response_model=MinimalResults)
async def get_job_results(job_id: str):
    """Get job results"""
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    results = job.get("results", {})
    exported_files = job.get("exported_files", {})
    
    # Create download links
    download_links = {}
    for file_type, filepath in exported_files.items():
        filename = Path(filepath).name
        download_links[file_type] = f"/download/{job_id}/{filename}"
    
    return MinimalResults(
        job_id=job_id,
        files_processed=results.get("files_processed", 0),
        analysis_time=results.get("processing_time", 0.0),
        mock_signal_count=results.get("signals_found", 0),
        mock_correlation_count=results.get("correlations_found", 0),
        download_links=download_links
    )

@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download result file"""
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # Find the file
    results_dir = RESULTS_DIR / job_id
    file_path = results_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, filename=filename)

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return [
        {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "created_at": job["created_at"],
            "message": job["message"]
        }
        for job_id, job in analysis_jobs.items()
    ]

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job"""
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Cleanup files
    job_dir = UPLOAD_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    
    results_dir = RESULTS_DIR / job_id
    if results_dir.exists():
        shutil.rmtree(results_dir)
    
    del analysis_jobs[job_id]
    
    return {"message": f"Job {job_id} deleted"}

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    total_jobs = len(analysis_jobs)
    completed_jobs = sum(1 for j in analysis_jobs.values() if j["status"] == "completed")
    failed_jobs = sum(1 for j in analysis_jobs.values() if j["status"] == "failed")
    active_jobs = sum(1 for j in analysis_jobs.values() if j["status"] in ["queued", "processing"])
    
    return {
        "system": {
            "uptime_seconds": time.time() - app_start_time,
            "version": "1.0.0-minimal",
            "type": "Windows-compatible minimal backend"
        },
        "jobs": {
            "total": total_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs,
            "active": active_jobs
        },
        "features": {
            "heavy_dependencies": False,
            "numpy_pandas": False,
            "basic_parsing": True,
            "windows_compatible": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Minimal FastAPI Backend (Windows Compatible)")
    print("=" * 60)
    print("Features:")
    print("  ✅ File upload and basic parsing")
    print("  ✅ Background job processing")
    print("  ✅ Simple CAN analysis")
    print("  ✅ Windows compatible (no C++ build tools needed)")
    print("")
    print("Limitations:")
    print("  ❌ No advanced signal processing")
    print("  ❌ No statistical analysis")
    print("  ❌ No correlation engine")
    print("")
    print("API available at: http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)