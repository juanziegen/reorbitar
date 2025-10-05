# Satellite Debris Removal Service - 3D Visualization Frontend

This React application provides a 3D orbital visualization for the Satellite Debris Removal Service platform.

## Features

- **3D Earth Visualization**: Realistic Earth rendering with atmosphere glow
- **Earth-Centered Coordinate System**: ECI (Earth-Centered Inertial) coordinate system with axis indicators
- **Satellite Rendering**: Real-time satellite position calculation and visualization
- **Orbital Mechanics**: Simplified orbital propagation for demonstration
- **Interactive Controls**: Zoom, pan, and rotate the 3D scene
- **Reference Indicators**: LEO boundaries, ISS orbit, and coordinate grids

## Technology Stack

- **React 18** with TypeScript
- **Three.js** for 3D graphics
- **@react-three/fiber** for React Three.js integration
- **@react-three/drei** for additional Three.js helpers

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm start
```

Opens the application at `http://localhost:3000`

### Build

```bash
npm run build
```

## Architecture

### Components

- **OrbitalVisualization**: Main 3D scene container
- **Earth**: Earth sphere with texture and atmosphere
- **CoordinateSystem**: ECI coordinate axes and reference grids
- **SatelliteRenderer**: Satellite objects and orbital paths

### Coordinate System

The visualization uses an Earth-Centered Inertial (ECI) coordinate system:
- **X-axis (Red)**: Points towards vernal equinox
- **Y-axis (Green)**: 90° ahead in equatorial plane
- **Z-axis (Blue)**: Points towards North Pole

### Orbital Mechanics

The application includes simplified orbital mechanics calculations:
- Satellite position propagation
- Orbital period calculations
- Delta-v approximations
- LEO classification

## Configuration

### Satellite Data

Satellites are defined with:
- TLE (Two-Line Element) data
- Orbital elements (semi-major axis, eccentricity, inclination, etc.)
- Physical properties (mass, material composition)
- Status (active, debris, decommissioned)

### Visualization Settings

- Earth radius: 6,371 km
- LEO range: 160 - 2,000 km altitude
- Time acceleration: 1 minute per second
- Camera limits: 7,000 - 50,000 km distance

## Future Enhancements

- Integration with real satellite databases (NORAD, Space-Track)
- Advanced orbital propagation (SGP4/SDP4)
- Route optimization visualization
- Real-time satellite tracking
- Mission planning interface