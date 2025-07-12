#!/usr/bin/env python3
"""
day4_fastapi_main.py
Day 4: FastAPI Backend for Universal Network Analyzer
Production-ready web API with file upload and background processing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import time
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from datetime import datetime

# Import our Day 1-3 components
from day3_network_analyzer import Day3NetworkAnalyzer
from day2_universal_signal import ProtocolType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Universal Network Analyzer API",
    description="Day 4 - Production-ready automotive network analysis API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for analysis jobs
analysis_jobs: Dict[str, Dict[str, Any]] = {}

# Directories for file storage
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Pydantic models for API responses
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
    results_available: bool = False
    
class AnalysisResults(BaseModel):
    job_id: str
    analysis_metadata: Dict[str, Any]
    signal_count: int
    correlation_count: int
    gateway_paths: int
    processing_time: float
    download_links: Dict[str, str]

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    active_jobs: int

# Global variables for health monitoring
app_start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize the FastAPI application"""
    logger.info("🚀 Universal Network Analyzer API starting up...")
    logger.info(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
    logger.info(f"📁 Results directory: {RESULTS_DIR.absolute()}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when shutting down"""
    logger.info("🛑 Universal Network Analyzer API shutting down...")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    uptime = time.time() - app_start_time
    active_jobs = len([job for job in analysis_jobs.values() 
                      if job['status'] in ['queued', 'processing']])
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=uptime,
        active_jobs=active_jobs
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Universal Network Analyzer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "day": 4,
        "status": "🚀 Ready for automotive network analysis!"
    }

def detect_file_protocol(filename: str) -> ProtocolType:
    """Detect protocol from file extension"""
    ext = Path(filename).suffix.lower()
    
    if ext == '.dbc':
        return ProtocolType.CAN
    elif ext == '.ldf':
        return ProtocolType.LIN
    elif ext in ['.pcap', '.pcapng']:
        return ProtocolType.ETHERNET
    elif ext in ['.asc', '.blf', '.trc']:
        # Could be any protocol - default to CAN
        return ProtocolType.CAN
    else:
        return ProtocolType.CAN  # Default

def validate_uploaded_files(files: List[UploadFile]) -> Dict[str, List[str]]:
    """Validate uploaded files and categorize them"""
    databases = []
    logs = []
    errors = []
    
    for file in files:
        filename = file.filename
        ext = Path(filename).suffix.lower()
        
        # Database files
        if ext in ['.dbc', '.ldf', '.arxml']:
            databases.append(filename)
        # Log files
        elif ext in ['.asc', '.blf', '.trc', '.pcap', '.pcapng', '.log', '.txt']:
            logs.append(filename)
        else:
            errors.append(f"Unsupported file type: {filename}")
    
    return {
        "databases": databases,
        "logs": logs,
        "errors": errors
    }

async def save_uploaded_files(job_id: str, files: List[UploadFile]) -> Dict[str, str]:
    """Save uploaded files to disk"""
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    
    saved_files = {}
    
    for file in files:
        file_path = job_dir / file.filename
        
        try:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            saved_files[file.filename] = str(file_path)
            logger.info(f"💾 Saved file: {file.filename} ({len(content)} bytes)")
            
        except Exception as e:
            logger.error(f"❌ Failed to save {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save {file.filename}")
    
    return saved_files

async def process_analysis_job(job_id: str, saved_files: Dict[str, str]):
    """Background task to process analysis job"""
    try:
        logger.info(f"🔬 Starting analysis job: {job_id}")
        
        # Update job status
        analysis_jobs[job_id].update({
            "status": "processing",
            "progress": 10.0,
            "message": "Initializing analysis..."
        })
        
        # Initialize analyzer
        analyzer = Day3NetworkAnalyzer()
        
        # Categorize files
        databases = {}
        log_files = []
        
        for filename, filepath in saved_files.items():
            ext = Path(filename).suffix.lower()
            
            if ext in ['.dbc', '.ldf', '.arxml']:
                protocol = detect_file_protocol(filename)
                databases[protocol] = filepath
            elif ext in ['.asc', '.blf', '.trc', '.pcap', '.pcapng', '.log', '.txt']:
                protocol = detect_file_protocol(filename)
                log_files.append({"path": filepath, "protocol": protocol})
        
        analysis_jobs[job_id].update({
            "progress": 30.0,
            "message": f"Loading {len(databases)} databases..."
        })
        
        # Load protocol databases
        for protocol, db_path in databases.items():
            success = analyzer.day2_analyzer.add_protocol_database(protocol, db_path)
            if not success:
                raise Exception(f"Failed to load {protocol.value} database: {db_path}")
        
        analysis_jobs[job_id].update({
            "progress": 50.0,
            "message": f"Analyzing {len(log_files)} log files..."
        })
        
        # Create log configs
        log_configs = []
        for log_info in log_files:
            # Find matching database
            protocol = log_info["protocol"]
            if protocol in databases:
                log_configs.append({
                    "protocol": protocol,
                    "database": databases[protocol],
                    "path": log_info["path"]
                })
            else:
                logger.warning(f"No database found for {protocol.value} log: {log_info['path']}")
        
        if not log_configs:
            raise Exception("No valid log/database pairs found")
        
        analysis_jobs[job_id].update({
            "progress": 70.0,
            "message": "Running complete network analysis..."
        })
        
        # Run complete analysis
        results = analyzer.analyze_complete_network(log_configs)
        
        if not results:
            raise Exception("Analysis failed - no results generated")
        
        analysis_jobs[job_id].update({
            "progress": 90.0,
            "message": "Exporting results..."
        })
        
        # Export results
        results_prefix = str(RESULTS_DIR / job_id)
        exported_files = analyzer.export_complete_results(results_prefix)
        
        # Success!
        analysis_jobs[job_id].update({
            "status": "completed",
            "progress": 100.0,
            "message": "Analysis completed successfully",
            "completed_at": datetime.now(),
            "results": results,
            "exported_files": exported_files,
            "results_available": True
        })
        
        logger.info(f"✅ Analysis job completed: {job_id}")
        
    except Exception as e:
        logger.error(f"❌ Analysis job failed: {job_id} - {e}")
        analysis_jobs[job_id].update({
            "status": "failed",
            "progress": 0.0,
            "message": f"Analysis failed: {str(e)}",
            "completed_at": datetime.now(),
            "error": str(e)
        })

@app.post("/analyze", response_model=AnalysisJobResponse)
async def start_analysis(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Start a new analysis job with uploaded files
    
    Accepts multiple files:
    - Database files: .dbc (CAN), .ldf (LIN), .arxml (Ethernet)
    - Log files: .asc, .blf, .trc, .pcap, .log, .txt
    """
    
    # Validate files
    validation = validate_uploaded_files(files)
    
    if validation["errors"]:
        raise HTTPException(
            status_code=400, 
            detail=f"File validation errors: {', '.join(validation['errors'])}"
        )
    
    if not validation["databases"]:
        raise HTTPException(
            status_code=400,
            detail="At least one database file (.dbc, .ldf, .arxml) is required"
        )
    
    if not validation["logs"]:
        raise HTTPException(
            status_code=400,
            detail="At least one log file is required"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save files
    try:
        saved_files = await save_uploaded_files(job_id, files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Initialize job tracking
    analysis_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "message": "Job queued for processing",
        "created_at": datetime.now(),
        "files": saved_files,
        "file_count": len(files),
        "databases": validation["databases"],
        "logs": validation["logs"]
    }
    
    # Start background processing
    background_tasks.add_task(process_analysis_job, job_id, saved_files)
    
    logger.info(f"🚀 Started analysis job: {job_id} with {len(files)} files")
    
    return AnalysisJobResponse(
        job_id=job_id,
        status="queued",
        message=f"Analysis job started with {len(files)} files",
        created_at=datetime.now()
    )

@app.get("/jobs/{job_id}/status", response_model=AnalysisStatus)
async def get_job_status(job_id: str):
    """Get the status of an analysis job"""
    
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
        error=job.get("error"),
        results_available=job.get("results_available", False)
    )

@app.get("/jobs/{job_id}/results", response_model=AnalysisResults)
async def get_job_results(job_id: str):
    """Get the results of a completed analysis job"""
    
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if not job.get("results_available"):
        raise HTTPException(status_code=404, detail="Results not available")
    
    results = job["results"]
    exported_files = job.get("exported_files", {})
    
    # Create download links
    download_links = {}
    for file_type, filepath in exported_files.items():
        filename = Path(filepath).name
        download_links[file_type] = f"/download/{job_id}/{filename}"
    
    return AnalysisResults(
        job_id=job_id,
        analysis_metadata=results["analysis_metadata"],
        signal_count=results["basic_analysis"]["total_signals"],
        correlation_count=results["correlation_analysis"]["total_correlations"],
        gateway_paths=results["latency_analysis"]["total_gateway_paths"],
        processing_time=results["analysis_metadata"]["analysis_time"],
        download_links=download_links
    )

@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a result file"""
    
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = analysis_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # Find the file
    exported_files = job.get("exported_files", {})
    file_path = None
    
    for filepath in exported_files.values():
        if Path(filepath).name == filename:
            file_path = filepath
            break
    
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/jobs", response_model=List[AnalysisStatus])
async def list_jobs():
    """List all analysis jobs"""
    
    job_list = []
    for job_id, job in analysis_jobs.items():
        job_list.append(AnalysisStatus(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            message=job["message"],
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
            error=job.get("error"),
            results_available=job.get("results_available", False)
        ))
    
    return sorted(job_list, key=lambda x: x.created_at, reverse=True)

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete an analysis job and its files"""
    
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete uploaded files
    job_dir = UPLOAD_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    
    # Delete result files
    job = analysis_jobs[job_id]
    exported_files = job.get("exported_files", {})
    for filepath in exported_files.values():
        try:
            Path(filepath).unlink()
        except:
            pass
    
    # Remove from tracking
    del analysis_jobs[job_id]
    
    logger.info(f"🗑️  Deleted job: {job_id}")
    
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    
    total_jobs = len(analysis_jobs)
    completed_jobs = len([j for j in analysis_jobs.values() if j["status"] == "completed"])
    failed_jobs = len([j for j in analysis_jobs.values() if j["status"] == "failed"])
    active_jobs = len([j for j in analysis_jobs.values() if j["status"] in ["queued", "processing"]])
    
    # Disk usage
    upload_size = sum(f.stat().st_size for f in UPLOAD_DIR.rglob('*') if f.is_file())
    results_size = sum(f.stat().st_size for f in RESULTS_DIR.rglob('*') if f.is_file())
    
    return {
        "system": {
            "uptime_seconds": time.time() - app_start_time,
            "version": "1.0.0",
            "day": 4
        },
        "jobs": {
            "total": total_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs,
            "active": active_jobs
        },
        "storage": {
            "uploads_bytes": upload_size,
            "results_bytes": results_size,
            "total_bytes": upload_size + results_size
        }
    }

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "day4_fastapi_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )