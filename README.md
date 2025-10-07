# 🎨 CIE 1931 Chromaticity Diagram Simulator & Comparator

![CIE Chromaticity Diagram](https://img.shields.io/badge/Color-Science-blue)
![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)

An interactive web application for visualizing and comparing LED color specifications using the CIE 1931 chromaticity diagram. Perfect for LED manufacturers, display engineers, and color science professionals.

## 🌟 Features

- **🔄 Multi-LED Comparison**: Compare up to 6 different LED specifications simultaneously
- **🎯 Real-time Visualization**: Instant updates with interactive controls
- **📊 Professional Metrics**: Centroid distances, polygon areas, and color analysis
- **💾 Export Capabilities**: Download high-resolution PNG images
- **🔒 Secure Access**: Password-protected application
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ciecompare.streamlit.app/)

credentials for access: " Rohit123 "

📖 How It Works
🎨 The Science Behind
The CIE 1931 chromaticity diagram is a fundamental color space that represents all colors visible to the human eye. Our simulator:

Generates the color map mathematically using color science principles

Converts coordinates to actual visible colors using colour-science library

Plots LED specifications as polygons for visual comparison

Calculates metrics for professional analysis

User Input (x,y coordinates)
        ↓
CIE xyY Color Space
        ↓
xyY_to_XYZ() → Human Vision Model
        ↓
XYZ_to_sRGB() → Computer Colors
        ↓
Matplotlib Display → Visual Diagram

📊 Key Metrics Calculated
Centroid: Average color point of LED specification

Polygon Area: Color gamut coverage

Distance Matrix: Differences between multiple LEDs

Color Purity: How saturated the colors are

💡 Usage Guide
Single LED Analysis
Set "Number of LED sets" to 1

Enter your LED's (x,y) coordinates

View the color gamut polygon on the diagram

Analyze centroid position and area metrics

Multiple LED Comparison
Choose 2-6 LED sets to compare

Name each set meaningfully (e.g., "Samsung QLED", "LG OLED")

Enter coordinates for each specification

Compare visual overlap and centroid distances

Use the distance matrix for quantitative analysis

Display Options
Fill Polygons: Show filled color areas

Show Borders: Display polygon outlines

Show Points: Plot individual coordinate points

Show Centroids: Mark average color points

🏭 Industry Applications
LED Manufacturing
Quality control of color consistency

Specification verification against targets

Batch-to-batch comparison

Display Engineering
Color gamut analysis for displays

OLED vs LCD technology comparison

Color accuracy validation

Research & Development
New material color performance

Competitor product analysis

Standard compliance checking (DCI-P3, Rec. 709, etc.)
