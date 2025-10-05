# Satellite Debris Removal Service

A comprehensive commercial platform for satellite debris removal services combining genetic algorithm route optimization, biprop cost calculations ($1.27 per m/s Δv), and professional client interfaces.

## Project Structure

```
debris_removal_service/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── satellite.py          # Satellite data models and TLE parsing
│   ├── route.py              # Route and hop models for missions
│   ├── cost.py               # Cost calculation models
│   └── service_request.py    # Client service request models
└── utils/
    ├── __init__.py
    ├── validation.py         # Satellite data validation
    └── tle_parser.py         # TLE parsing utilities
```

## Core Features

- **Satellite Data Models**: Complete satellite representation with TLE parsing and orbital elements
- **Route Optimization**: Multi-hop route planning with cost and feasibility analysis
- **Cost Modeling**: Accurate biprop propulsion cost calculations
- **Service Requests**: Client request management with timeline and budget constraints
- **Data Validation**: Comprehensive validation for satellite data and TLE formats

## Getting Started

```python
from debris_removal_service.models import Satellite, ServiceRequest
from debris_removal_service.utils import SatelliteDataValidator

# Create a satellite
satellite = Satellite(
    id="25544",
    name="ISS",
    tle_line1="1 25544U 98067A   21001.00000000  .00002182  00000-0  40768-4 0  9992",
    tle_line2="2 25544  51.6461 339.2911 0002829 242.9350 117.0717 15.48919103123456",
    mass=420000.0,
    material_composition={'aluminum': 0.4, 'steel': 0.3, 'electronics': 0.2, 'other': 0.1},
    decommission_date=datetime(2030, 1, 1)
)

# Validate satellite data
is_valid, errors = SatelliteDataValidator.validate_satellite(satellite)
```

## Testing

Run the test suite:
```bash
python test_simple_models.py
```

## Next Steps

This completes the core project structure and data models. The next tasks will implement:
1. Biprop cost calculation engine
2. Genetic algorithm route optimizer
3. REST API backend services
4. 3D visualization frontend
5. Commercial website interface