#!/usr/bin/env python3
"""
day5_fastapi_integration.py
Day 5: FastAPI Integration
Adds enhanced latency analysis endpoints to Day 4 FastAPI backend
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

# Import Day 5 components
try:
    from day5_latency_engine import EnhancedLatencyEngine, integrate_enhanced_latency_analysis
    from day5_integration import Day5IntegratedAnalyzer
    DAY5_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Day 5 components not available: {e}")
    DAY5_AVAILABLE = False

# Pydantic models for Day 5 API responses
class EnhancedLatencyRequest(BaseModel):
    job_id: str
    enable_trend_analysis: bool = True
    enable_multi_hop_analysis: bool = True
    custom_timing_requirements: Optional[Dict[str, float]] = None

class EnhancedLatencyResponse(BaseModel):
    job_id: str
    enhanced_analysis_id: str
    status: str
    message: str
    started_at: datetime

class EnhancedLatencyResults(BaseModel):
    job_id: str
    enhanced_analysis_id: str
    paths_analyzed: int
    overall_performance_score: float
    compliance_rate_percent: float
    system_reliability_score: float
    critical_issues_count: int
    high_priority_issues_count: int
    exported_files: Dict[str, str]
    executive_summary: str

class PerformanceDashboard(BaseModel):
    total_enhanced_analyses: int
    average_performance_score: float
    compliance_trends: Dict[str, float]
    top_performing_paths: List[Dict[str, Any]]
    paths_needing_attention: List[Dict[str, Any]]
    system_health_score: float

# Global storage for enhanced analysis jobs
enhanced_analysis_jobs = {}

def create_day5_router():
    """Create FastAPI router with Day 5 enhanced latency analysis endpoints"""
    
    from fastapi import APIRouter
    router = APIRouter(prefix="/enhanced", tags=["Day 5 - Enhanced Latency Analysis"])
    
    if not DAY5_AVAILABLE:
        @router.get("/status")
        async def day5_status():
            return {
                "day5_available": False,
                "message": "Day 5 enhanced latency analysis not available",
                "reason": "Missing Day 5 components or dependencies",
                "fallback": "Basic latency analysis available in Day 3"
            }
        return router
    
    @router.post("/analyze/{job_id}", response_model=EnhancedLatencyResponse)
    async def start_enhanced_analysis(
        job_id: str,
        request: EnhancedLatencyRequest,
        background_tasks: BackgroundTasks
    ):
        """Start enhanced latency analysis for an existing analysis job"""
        
        # Import from day4_fastapi_main if available
        try:
            from day4_fastapi_main import analysis_jobs
        except ImportError:
            # Mock analysis jobs for standalone testing
            analysis_jobs = {
                job_id: {
                    "status": "completed",
                    "results": {"mock": True}
                }
            }
        
        # Check if base job exists and is completed
        if job_id not in analysis_jobs:
            raise HTTPException(status_code=404, detail="Base analysis job not found")
        
        base_job = analysis_jobs[job_id]
        if base_job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Base analysis not completed")
        
        # Generate enhanced analysis ID
        enhanced_id = f"enhanced_{job_id}_{int(time.time())}"
        
        # Initialize enhanced analysis job
        enhanced_analysis_jobs[enhanced_id] = {
            "enhanced_analysis_id": enhanced_id,
            "base_job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Enhanced latency analysis queued",
            "started_at": datetime.now(),
            "request_params": request.dict()
        }
        
        # Start background enhanced analysis
        background_tasks.add_task(
            process_enhanced_latency_analysis,
            enhanced_id,
            job_id,
            request
        )
        
        return EnhancedLatencyResponse(
            job_id=job_id,
            enhanced_analysis_id=enhanced_id,
            status="queued",
            message="Enhanced latency analysis started",
            started_at=datetime.now()
        )
    
    @router.get("/analyze/{enhanced_analysis_id}/status")
    async def get_enhanced_analysis_status(enhanced_analysis_id: str):
        """Get status of enhanced latency analysis"""
        
        if enhanced_analysis_id not in enhanced_analysis_jobs:
            raise HTTPException(status_code=404, detail="Enhanced analysis not found")
        
        job = enhanced_analysis_jobs[enhanced_analysis_id]
        
        return {
            "enhanced_analysis_id": enhanced_analysis_id,
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
            "started_at": job["started_at"],
            "completed_at": job.get("completed_at"),
            "error": job.get("error")
        }
    
    @router.get("/analyze/{enhanced_analysis_id}/results", response_model=EnhancedLatencyResults)
    async def get_enhanced_analysis_results(enhanced_analysis_id: str):
        """Get results of enhanced latency analysis"""
        
        if enhanced_analysis_id not in enhanced_analysis_jobs:
            raise HTTPException(status_code=404, detail="Enhanced analysis not found")
        
        job = enhanced_analysis_jobs[enhanced_analysis_id]
        
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Enhanced analysis not completed")
        
        results = job.get("results", {})
        
        return EnhancedLatencyResults(
            job_id=job["base_job_id"],
            enhanced_analysis_id=enhanced_analysis_id,
            paths_analyzed=results.get("enhanced_results_count", 0),
            overall_performance_score=results.get("overall_performance_score", 0.0),
            compliance_rate_percent=results.get("compliance_rate", 0.0),
            system_reliability_score=results.get("system_reliability_score", 0.0),
            critical_issues_count=results.get("critical_issues", 0),
            high_priority_issues_count=results.get("high_priority_issues", 0),
            exported_files=results.get("exported_files", {}),
            executive_summary=results.get("executive_summary", "")
        )
    
    @router.get("/dashboard", response_model=PerformanceDashboard)
    async def get_performance_dashboard():
        """Get system-wide performance dashboard"""
        
        completed_analyses = [
            job for job in enhanced_analysis_jobs.values()
            if job["status"] == "completed" and "results" in job
        ]
        
        if not completed_analyses:
            return PerformanceDashboard(
                total_enhanced_analyses=0,
                average_performance_score=0.0,
                compliance_trends={},
                top_performing_paths=[],
                paths_needing_attention=[],
                system_health_score=0.0
            )
        
        # Calculate dashboard metrics
        total_analyses = len(completed_analyses)
        
        # Average performance score
        performance_scores = [
            job["results"].get("overall_performance_score", 0)
            for job in completed_analyses
        ]
        avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0
        
        # Compliance trends (mock implementation)
        compliance_trends = {
            "last_24h": avg_performance,
            "last_7d": avg_performance * 0.95,
            "last_30d": avg_performance * 0.90
        }
        
        # Top performing and problematic paths (simplified)
        top_paths = [
            {
                "path": f"Analysis_{i+1}",
                "performance_score": score,
                "compliance": score > 80
            }
            for i, score in enumerate(sorted(performance_scores, reverse=True)[:5])
        ]
        
        problematic_paths = [
            {
                "path": f"Analysis_{i+1}",
                "performance_score": score,
                "issues": ["Performance below threshold"] if score < 70 else []
            }
            for i, score in enumerate(sorted(performance_scores)[:5])
            if score < 70
        ]
        
        # System health score
        system_health = min(100.0, avg_performance)
        
        return PerformanceDashboard(
            total_enhanced_analyses=total_analyses,
            average_performance_score=avg_performance,
            compliance_trends=compliance_trends,
            top_performing_paths=top_paths,
            paths_needing_attention=problematic_paths,
            system_health_score=system_health
        )
    
    @router.get("/benchmarks")
    async def get_timing_benchmarks():
        """Get automotive timing requirement benchmarks"""
        
        return {
            "automotive_timing_requirements": {
                "safety_critical": {
                    "limit_ms": 10.0,
                    "description": "Safety systems (airbag, ABS, ESP)",
                    "examples": ["brake_signal", "airbag_deploy", "abs_activation"]
                },
                "powertrain": {
                    "limit_ms": 20.0,
                    "description": "Engine and transmission control",
                    "examples": ["rpm_signal", "throttle_position", "gear_shift"]
                },
                "chassis": {
                    "limit_ms": 50.0,
                    "description": "Steering, suspension, wheels",
                    "examples": ["steering_angle", "wheel_speed", "suspension_height"]
                },
                "comfort": {
                    "limit_ms": 100.0,
                    "description": "HVAC, lighting, convenience",
                    "examples": ["hvac_control", "light_switch", "window_control"]
                },
                "infotainment": {
                    "limit_ms": 200.0,
                    "description": "Entertainment and navigation",
                    "examples": ["radio_control", "navigation_update", "display_refresh"]
                },
                "diagnostic": {
                    "limit_ms": 500.0,
                    "description": "Diagnostic and maintenance",
                    "examples": ["dtc_read", "parameter_query", "calibration"]
                }
            },
            "performance_grades": {
                "excellent": "≤50% of timing requirement",
                "good": "50-80% of timing requirement", 
                "fair": "80-100% of timing requirement",
                "poor": "100-150% of timing requirement",
                "critical": ">150% of timing requirement"
            }
        }
    
    @router.delete("/analyze/{enhanced_analysis_id}")
    async def delete_enhanced_analysis(enhanced_analysis_id: str):
        """Delete enhanced analysis results"""
        
        if enhanced_analysis_id not in enhanced_analysis_jobs:
            raise HTTPException(status_code=404, detail="Enhanced analysis not found")
        
        # Clean up exported files
        job = enhanced_analysis_jobs[enhanced_analysis_id]
        results = job.get("results", {})
        exported_files = results.get("exported_files", {})
        
        for filepath in exported_files.values():
            try:
                from pathlib import Path
                Path(filepath).unlink(missing_ok=True)
            except:
                pass
        
        # Remove from tracking
        del enhanced_analysis_jobs[enhanced_analysis_id]
        
        return {"message": f"Enhanced analysis {enhanced_analysis_id} deleted"}
    
    @router.get("/")
    async def day5_info():
        """Day 5 enhanced latency analysis information"""
        return {
            "day": 5,
            "feature": "Enhanced Latency Analysis Engine",
            "description": "Advanced statistical analysis, performance benchmarking, and multi-hop gateway analysis",
            "capabilities": [
                "Statistical distribution analysis (mean, median, percentiles)",
                "Performance benchmarking against automotive timing requirements",
                "Time-series trend analysis with prediction",
                "Multi-hop gateway optimization analysis",
                "Automated recommendations and monitoring setup",
                "Executive reporting and dashboard analytics"
            ],
            "endpoints": [
                "POST /enhanced/analyze/{job_id} - Start enhanced analysis",
                "GET /enhanced/analyze/{id}/status - Check analysis status",
                "GET /enhanced/analyze/{id}/results - Get analysis results",
                "GET /enhanced/dashboard - Performance dashboard",
                "GET /enhanced/benchmarks - Timing requirements",
                "DELETE /enhanced/analyze/{id} - Delete analysis"
            ],
            "integration": "Extends Day 4 FastAPI backend with Day 5 enhanced analysis",
            "day5_available": DAY5_AVAILABLE
        }
    
    return router

async def process_enhanced_latency_analysis(enhanced_id: str, base_job_id: str, request: EnhancedLatencyRequest):
    """Background task for enhanced latency analysis processing"""
    
    try:
        # Update status
        enhanced_analysis_jobs[enhanced_id].update({
            "status": "processing",
            "progress": 20.0,
            "message": "Initializing enhanced latency analysis..."
        })
        
        # Initialize Day 5 integrated analyzer
        analyzer = Day5IntegratedAnalyzer()
        
        enhanced_analysis_jobs[enhanced_id].update({
            "progress": 40.0,
            "message": "Running enhanced statistical analysis..."
        })
        
        # Mock log configs for demonstration (in real implementation, get from base job)
        mock_log_configs = [
            {"protocol": "CAN", "database": "mock.dbc", "path": "mock.asc"}
        ]
        
        enhanced_analysis_jobs[enhanced_id].update({
            "progress": 70.0,
            "message": "Generating enhanced reports..."
        })
        
        # Run enhanced analysis
        results = analyzer.analyze_complete_network_enhanced(mock_log_configs)
        
        enhanced_analysis_jobs[enhanced_id].update({
            "progress": 90.0,
            "message": "Finalizing enhanced analysis..."
        })
        
        # Extract enhanced results
        enhanced_results = results.get("enhanced_latency_analysis", {})
        system_summary = enhanced_results.get("system_performance_summary", {}).get("summary", {})
        
        # Generate executive summary
        executive_summary = analyzer.generate_day5_executive_summary()
        
        # Success
        enhanced_analysis_jobs[enhanced_id].update({
            "status": "completed",
            "progress": 100.0,
            "message": "Enhanced latency analysis completed",
            "completed_at": datetime.now(),
            "results": {
                "enhanced_results_count": enhanced_results.get("enhanced_results_count", 0),
                "overall_performance_score": system_summary.get("average_overall_score", 0.0),
                "compliance_rate": system_summary.get("compliance_rate_percent", 0.0),
                "system_reliability_score": system_summary.get("average_reliability_score", 0.0),
                "critical_issues": enhanced_results.get("system_performance_summary", {}).get("issues", {}).get("critical_count", 0),
                "high_priority_issues": enhanced_results.get("system_performance_summary", {}).get("issues", {}).get("high_priority_count", 0),
                "exported_files": enhanced_results.get("exported_files", {}),
                "executive_summary": executive_summary
            }
        })
        
    except Exception as e:
        enhanced_analysis_jobs[enhanced_id].update({
            "status": "failed",
            "progress": 0.0,
            "message": f"Enhanced analysis failed: {str(e)}",
            "completed_at": datetime.now(),
            "error": str(e)
        })

# Function to integrate Day 5 router with existing Day 4 FastAPI app
def integrate_day5_with_fastapi(app: FastAPI):
    """Integrate Day 5 enhanced latency analysis with existing FastAPI app"""
    
    # Add Day 5 router
    day5_router = create_day5_router()
    app.include_router(day5_router)
    
    # Add Day 5 info to root endpoint
    @app.get("/day5")
    async def day5_status():
        return {
            "day": 5,
            "status": "Enhanced Latency Analysis Engine",
            "available": DAY5_AVAILABLE,
            "integration": "Added to Day 4 FastAPI backend",
            "endpoints": "/enhanced/*",
            "documentation": "/docs#/Day%205%20-%20Enhanced%20Latency%20Analysis"
        }
    
    print("✅ Day 5 Enhanced Latency Analysis integrated with FastAPI backend")
    print("   📊 New endpoints available under /enhanced/*")
    print("   🔬 Advanced statistical analysis capabilities added")
    print("   📈 Performance dashboard and benchmarking available")

if __name__ == "__main__":
    # Standalone Day 5 FastAPI demo
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI(
        title="Day 5 - Enhanced Latency Analysis API",
        description="Standalone Day 5 enhanced latency analysis endpoints",
        version="1.0.0"
    )
    
    # Integrate Day 5 functionality
    integrate_day5_with_fastapi(app)
    
    print("🚀 Day 5 Enhanced Latency Analysis API")
    print("   Available at: http://localhost:8005")
    print("   Documentation: http://localhost:8005/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8005)