#!/bin/bash

# Voice Agent Development Startup Script
echo "🎤 Voice Agent Development Startup"
echo "=================================="

# Check if backend is running
echo "🔍 Checking if backend is running..."
if curl -s http://localhost:8200/voice/health > /dev/null 2>&1; then
    echo "✅ Backend is running on HTTP"
    BACKEND_MODE="http"
elif curl -k -s https://localhost:8200/voice/health > /dev/null 2>&1; then
    echo "✅ Backend is running on HTTPS"
    BACKEND_MODE="https"
else
    echo "❌ Backend is not running"
    echo ""
    echo "🚀 Starting backend in HTTP mode..."
    cd ../..
    python main.py --http &
    BACKEND_PID=$!
    echo "✅ Backend started with PID: $BACKEND_PID"
    BACKEND_MODE="http"
    sleep 3
fi

# Set environment variable based on backend mode
if [ "$BACKEND_MODE" = "http" ]; then
    export NEXT_PUBLIC_BACKEND_URL=http://localhost:8200
    echo "🌐 Using HTTP mode"
else
    export NEXT_PUBLIC_BACKEND_URL=https://localhost:8200
    echo "🔒 Using HTTPS mode"
fi

echo ""
echo "🎯 Starting frontend..."
echo "Frontend will be available at: http://localhost:3000"
echo "Backend is available at: $NEXT_PUBLIC_BACKEND_URL"
echo ""
echo "💡 Tips:"
echo "  - Use the Connection Test component to verify connectivity"
echo "  - Check browser console for any errors"
echo "  - Press Ctrl+C to stop the frontend"
echo ""

# Start frontend
npm run dev
