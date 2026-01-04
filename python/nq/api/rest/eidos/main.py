"""
FastAPI application for Eidos REST API.

Usage:
    python -m nq.api.rest.eidos.main
    # or
    uvicorn nq.api.rest.eidos.main:app --host 0.0.0.0 --port 8000
"""

import argparse
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nq.api.rest.eidos.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Eidos REST API server...")
    yield
    # Shutdown
    logger.info("Shutting down Eidos REST API server...")


app = FastAPI(
    title="Eidos REST API",
    description="Universal Backtest Attribution System REST API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Eidos REST API",
        "version": "0.1.0",
        "description": "Universal Backtest Attribution System",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Main entry point for running the server."""
    parser = argparse.ArgumentParser(description="Eidos REST API Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)",
    )
    
    args = parser.parse_args()
    
    print(f"Starting Eidos REST API server on http://{args.host}:{args.port}")
    print(f"API documentation: http://{args.host}:{args.port}/docs")
    print(f"Health check: http://{args.host}:{args.port}/health")
    
    uvicorn.run(
        "nq.api.rest.eidos.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

