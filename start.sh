#!/bin/bash
# Start backend
cd backend && python app_working_final.py &

# Wait for backend to start
sleep 5

# Start frontend
cd ../frontend && python -m http.server 3000