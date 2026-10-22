# WicketTracker

AI-powered cricket wicket detection and delivery analysis system.

## Overview

WicketTracker is a computer-vision-based cricket analysis system designed to identify wicket-taking deliveries from cricket match videos.

The system combines scoreboard OCR, pitch detection, cricket ball detection, ball tracking, trajectory analysis, and heatmap generation.

## System Pipeline

Video Input
→ Scoreboard OCR
→ Wicket Detection
→ Delivery Clip Extraction
→ Pitch Detection
→ Ball Detection
→ Ball Tracking
→ Trajectory Generation
→ Heatmap Generation
→ Results

## Main Modules

### OCR
Detects and reads scoreboard information to identify changes in the wicket count and locate potential wicket events.

### Ball Detector
Uses an object detection model to detect the cricket ball in video frames.

### Pitch Detector
Identifies the cricket pitch/playing region to improve spatial filtering and ball detection.

### Tracking
Tracks the detected cricket ball across consecutive frames and generates its movement coordinates.

### Heatmaps
Visualizes ball locations, bounce points, trajectories, and delivery patterns.

### Frontend
Provides the user interface for uploading videos and viewing analysis results.

### Backend
Connects the frontend with the computer-vision pipeline and manages processing requests and results.

## Team

- Person 1 — OCR & Ball Detection
- Person 2 — Pitch Detection
- Person 3 — Ball Tracking & Heatmaps
- Person 4,5 — Frontend & Backend Integration

## Project Status

Under development.

## Documentation

Detailed project documentation is available in the `docs/` directory.
