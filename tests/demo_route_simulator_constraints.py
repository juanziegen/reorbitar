"""
Demonstration of RouteSimulator constraint handling and convergence detection capabilities.

This script demonstrates the key features implemented for task 3.3:
- Constraint handling for budget limits and timeline requirements
- Convergence detection and early termination capabilities
- Genetic algorithm integration with cost calculations
"""

from datetime import datetime, timedelta
from debris_removal_service.models.satellite import Satellite, OrbitalElements
from debris_removal_service.models.service_request import (
    ServiceRequest, TimelineConstraints, BudgetConstraints, 
    ProcessingPreferences, ProcessingType
)
from debris_removal_service.services.route_simulator import RouteSimulator


def create_demo_satellites():
    """Create demonstration satellites with realistic orbital parameters."""
    satellites = []
    
    # LEO satellites at different altitudes and inclinations
    orbital_params = [
        (7000, 0.01, 45, "LEO-1"),
        (7200, 0.02, 55, "LEO-2"), 
        (7400, 0.015, 65, "LEO-3"),
        (7600, 0.025, 75, "LEO-4"),
        (7800, 0.03, 85, "LEO-5")
    ]
    
    for i, (altitude, ecc, inc, name) in enumerate(orbital_params):
        orbital_elements = OrbitalElements(
            semi_major_axis=altitude,  # km
            eccentricity=ecc,
            inclination=inc,  # degrees
            raan=i * 45,  # degrees
            argument_of_perigee=i * 30,  # degrees
            mean_anomaly=i * 60,  # degrees
            mean_motion=15.0 - i * 0.2,  # revolutions per day
            epoch=datetime.now()
        )
        
        satellite = Satellite(
            id=str(i + 1),
            name=name,
            tle_line1=f"1 {i+1:05d}U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  9999",
            tle_line2=f"2 {i+1:05d}  {inc:7.4f} 000.0000 {int(ecc*10000000):07d} 000.0000 000.0000 15.0000000{i:05d}",
            mass=800.0 + i * 200,  # kg
            material_composition={"aluminum": 0.6, "steel": 0.25, "titanium": 0.1, "other": 0.05},
            decommission_date=datetime.now() + timedelta(days=60),
            orbital_elements=orbital_elements
        )
        satellites.append(satellite)
    
    return satellites


def create_demo_scenarios():
    """Create different service request scenarios to demonstrate constraint handling."""
    
    # Scenario 1: Normal budget and timeline
    scenario1 = ServiceRequest(
        client_id="demo_client_1",
        satellites=["1", "2", "3"],
        timeline_requirements=TimelineConstraints(
            earliest_start=datetime.now() + timedelta(days=7),
            latest_completion=datetime.now() + timedelta(days=45)
        ),
        budget_constraints=BudgetConstraints(
            max_total_cost=75000.0,
            preferred_cost=60000.0
        ),
        processing_preferences=ProcessingPreferences(
            preferred_processing_types=[ProcessingType.ISS_RECYCLING],
            processing_timeline=timedelta(days=10)
        )
    )
    
    # Scenario 2: Tight budget constraint
    scenario2 = ServiceRequest(
        client_id="demo_client_2",
        satellites=["1", "2", "3", "4"],
        timeline_requirements=TimelineConstraints(
            earliest_start=datetime.now() + timedelta(days=5),
            latest_completion=datetime.now() + timedelta(days=60)
        ),
        budget_constraints=BudgetConstraints(
            max_total_cost=25000.0,  # Very tight budget
            preferred_cost=20000.0
        ),
        processing_preferences=ProcessingPreferences(
            preferred_processing_types=[ProcessingType.HEO_STORAGE],
            processing_timeline=timedelta(days=5)
        )
    )
    
    # Scenario 3: Urgent timeline
    scenario3 = ServiceRequest(
        client_id="demo_client_3",
        satellites=["2", "3", "4", "5"],
        timeline_requirements=TimelineConstraints(
            earliest_start=datetime.now() + timedelta(days=1),
            latest_completion=datetime.now() + timedelta(days=15)  # Very urgent
        ),
        budget_constraints=BudgetConstraints(
            max_total_cost=100000.0,
            preferred_cost=80000.0
        ),
        processing_preferences=ProcessingPreferences(
            preferred_processing_types=[ProcessingType.SOLAR_FORGE],
            processing_timeline=timedelta(days=3)
        )
    )
    
    # Scenario 4: Large mission
    scenario4 = ServiceRequest(
        client_id="demo_client_4",
        satellites=["1", "2", "3", "4", "5"],  # All satellites
        timeline_requirements=TimelineConstraints(
            earliest_start=datetime.now() + timedelta(days=14),
            latest_completion=datetime.now() + timedelta(days=90)
        ),
        budget_constraints=BudgetConstraints(
            max_total_cost=150000.0,
            preferred_cost=120000.0,
            cost_breakdown_limits={
                "collection": 100000.0,
                "processing": 40000.0,
                "storage": 10000.0
            }
        ),
        processing_preferences=ProcessingPreferences(
            preferred_processing_types=[ProcessingType.ISS_RECYCLING, ProcessingType.SOLAR_FORGE],
            processing_timeline=timedelta(days=20)
        )
    )
    
    return [
        ("Normal Mission", scenario1),
        ("Tight Budget", scenario2), 
        ("Urgent Timeline", scenario3),
        ("Large Mission", scenario4)
    ]


def demonstrate_constraint_analysis(route_simulator, satellites, scenarios):
    """Demonstrate constraint analysis capabilities."""
    print("=" * 80)
    print("CONSTRAINT ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    for scenario_name, service_request in scenarios:
        print(f"\n--- {scenario_name} ---")
        
        # Analyze constraints
        analysis = route_simulator._analyze_constraints(service_request, satellites)
        
        print(f"Feasible: {analysis['feasible']}")
        print(f"Constraint Violations: {len(analysis['constraint_violations'])}")
        print(f"Warnings: {len(analysis['warnings'])}")
        print(f"Recommendations: {len(analysis['recommendations'])}")
        
        if analysis['constraint_violations']:
            print("Violations:")
            for violation in analysis['constraint_violations']:
                print(f"  - {violation}")
        
        if analysis['warnings']:
            print("Warnings:")
            for warning in analysis['warnings']:
                print(f"  - {warning}")
        
        if analysis['recommendations']:
            print("Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  - {rec}")


def demonstrate_budget_analysis(route_simulator, satellites, scenarios):
    """Demonstrate budget constraint analysis."""
    print("\n" + "=" * 80)
    print("BUDGET CONSTRAINT ANALYSIS")
    print("=" * 80)
    
    for scenario_name, service_request in scenarios:
        print(f"\n--- {scenario_name} ---")
        
        # Get requested satellites
        requested_satellites = [
            sat for sat in satellites 
            if sat.id in service_request.satellites
        ]
        
        analysis = route_simulator._analyze_budget_constraints(
            service_request, requested_satellites
        )
        
        if 'estimated_costs' in analysis:
            costs = analysis['estimated_costs']
            print(f"Estimated Collection Cost: ${costs['collection_cost']:,.2f}")
            print(f"Estimated Processing Cost: ${costs['processing_cost']:,.2f}")
            print(f"Estimated Total Cost: ${costs['total_cost']:,.2f}")
            print(f"Budget: ${service_request.budget_constraints.max_total_cost:,.2f}")
            print(f"Budget Utilization: {costs['budget_utilization']:.1%}")
            
            if costs['budget_utilization'] > 1.0:
                print("⚠️  BUDGET EXCEEDED!")
            elif costs['budget_utilization'] > 0.9:
                print("⚠️  Budget nearly exhausted")
            else:
                print("✅ Within budget")


def demonstrate_timeline_analysis(route_simulator, satellites, scenarios):
    """Demonstrate timeline constraint analysis."""
    print("\n" + "=" * 80)
    print("TIMELINE CONSTRAINT ANALYSIS")
    print("=" * 80)
    
    for scenario_name, service_request in scenarios:
        print(f"\n--- {scenario_name} ---")
        
        # Get requested satellites
        requested_satellites = [
            sat for sat in satellites 
            if sat.id in service_request.satellites
        ]
        
        analysis = route_simulator._analyze_timeline_constraints(
            service_request, requested_satellites
        )
        
        if 'estimated_timeline' in analysis:
            timeline = analysis['estimated_timeline']
            print(f"Estimated Duration: {timeline['estimated_duration_hours']:.1f} hours")
            print(f"Available Duration: {timeline['available_duration_hours']:.1f} hours")
            print(f"Timeline Utilization: {timeline['timeline_utilization']:.1%}")
            print(f"Is Urgent: {timeline['is_urgent']}")
            
            if timeline['timeline_utilization'] > 1.0:
                print("⚠️  TIMELINE EXCEEDED!")
            elif timeline['timeline_utilization'] > 0.9:
                print("⚠️  Timeline nearly exhausted")
            else:
                print("✅ Within timeline")


def demonstrate_convergence_detection(route_simulator, satellites, scenarios):
    """Demonstrate convergence detection and early termination."""
    print("\n" + "=" * 80)
    print("CONVERGENCE DETECTION & EARLY TERMINATION")
    print("=" * 80)
    
    # Use the normal mission scenario for demonstration
    scenario_name, service_request = scenarios[0]
    print(f"\nDemonstrating with: {scenario_name}")
    
    # Setup optimization with convergence parameters
    optimization_options = {
        'population_size': 20,  # Small for demo
        'max_generations': 50,
        'mutation_rate': 0.15,
        'crossover_rate': 0.8
    }
    
    print(f"Optimization Parameters:")
    print(f"  Population Size: {optimization_options['population_size']}")
    print(f"  Max Generations: {optimization_options['max_generations']}")
    print(f"  Convergence Threshold: {route_simulator.convergence_threshold}")
    print(f"  Max Stagnant Generations: {route_simulator.max_stagnant_generations}")
    print(f"  Early Termination Enabled: {route_simulator.early_termination_enabled}")
    
    # Run optimization
    print(f"\nRunning constrained optimization...")
    result = route_simulator.optimize_route_with_constraints(
        service_request, satellites, optimization_options
    )
    
    print(f"Optimization Success: {result['success']}")
    
    if result['success']:
        convergence_info = result['convergence_info']
        metadata = result['optimization_metadata']
        
        print(f"\nConvergence Information:")
        print(f"  Converged: {convergence_info['converged']}")
        print(f"  Generations Run: {convergence_info['generations_run']}")
        print(f"  Final Fitness: {convergence_info['final_fitness']:.2f}")
        print(f"  Convergence Reason: {convergence_info['convergence_reason']}")
        print(f"  Early Termination: {convergence_info.get('early_termination', False)}")
        
        print(f"\nOptimization Metadata:")
        print(f"  Execution Time: {metadata['execution_time_seconds']:.2f} seconds")
        print(f"  Total Generations: {metadata['total_generations']}")
        print(f"  Final Fitness: {metadata['final_fitness']:.2f}")
        print(f"  Constraint Violations: {len(metadata['constraint_violations'])}")
        
        if convergence_info.get('early_termination', False):
            print("✅ Early termination successfully detected!")
        else:
            print("ℹ️  Optimization completed normally")
    else:
        print(f"Optimization failed: {result.get('error', result.get('message', 'Unknown error'))}")


def demonstrate_fitness_evaluation(route_simulator, satellites, scenarios):
    """Demonstrate constrained fitness evaluation."""
    print("\n" + "=" * 80)
    print("CONSTRAINED FITNESS EVALUATION")
    print("=" * 80)
    
    scenario_name, service_request = scenarios[0]
    print(f"\nDemonstrating with: {scenario_name}")
    
    # Create route constraints
    constraints = route_simulator._create_route_constraints_with_penalties(
        service_request, satellites
    )
    
    print(f"Route Constraints:")
    print(f"  Max Budget: ${constraints['max_budget_usd']:,.2f}")
    print(f"  Max Delta-v Budget: {constraints['max_deltav_budget_ms']:,.0f} m/s")
    print(f"  Max Duration: {constraints['max_mission_duration_seconds']/3600:.1f} hours")
    print(f"  Budget Violation Penalty: ${constraints['budget_violation_penalty']:.2f} per dollar")
    print(f"  Timeline Violation Penalty: ${constraints['timeline_violation_penalty']:.2f} per hour")
    
    # Create mock routes with different constraint satisfaction levels
    class MockRoute:
        def __init__(self, cost, deltav, duration_hours):
            self.total_cost = cost
            self.total_delta_v = deltav
            self.mission_duration = timedelta(hours=duration_hours)
    
    class MockCost:
        def __init__(self, cost):
            self.total_cost = cost
    
    test_routes = [
        ("Within Constraints", MockRoute(40000, 1500, 20), MockCost(40000)),
        ("Budget Exceeded", MockRoute(80000, 1500, 20), MockCost(80000)),
        ("Timeline Exceeded", MockRoute(40000, 1500, 800), MockCost(40000)),
        ("Both Exceeded", MockRoute(80000, 3000, 800), MockCost(80000))
    ]
    
    print(f"\nFitness Evaluation Results:")
    for route_name, route, cost in test_routes:
        fitness = route_simulator._evaluate_constrained_fitness(
            route, cost, service_request, constraints
        )
        
        satisfaction = route_simulator._evaluate_constraint_satisfaction(
            route, service_request, constraints
        )
        
        print(f"\n{route_name}:")
        print(f"  Cost: ${route.total_cost:,.2f}")
        print(f"  Delta-v: {route.total_delta_v:,.0f} m/s")
        print(f"  Duration: {route.mission_duration.total_seconds()/3600:.1f} hours")
        print(f"  Fitness Score: {fitness:.2f}")
        print(f"  Budget Satisfied: {satisfaction['budget_satisfied']}")
        print(f"  Timeline Satisfied: {satisfaction['timeline_satisfied']}")
        print(f"  Overall Satisfied: {satisfaction['overall_satisfied']}")
        if satisfaction['violations']:
            print(f"  Violations: {len(satisfaction['violations'])}")


def main():
    """Run the complete demonstration."""
    print("RouteSimulator Constraint Handling & Convergence Detection Demo")
    print("Task 3.3: Build route optimization engine")
    print("Features: Genetic algorithms + cost calculations + constraint handling + convergence detection")
    
    # Initialize components
    route_simulator = RouteSimulator()
    satellites = create_demo_satellites()
    scenarios = create_demo_scenarios()
    
    print(f"\nDemo Setup:")
    print(f"  Satellites: {len(satellites)}")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Convergence Threshold: {route_simulator.convergence_threshold}")
    print(f"  Max Stagnant Generations: {route_simulator.max_stagnant_generations}")
    print(f"  Early Termination: {route_simulator.early_termination_enabled}")
    
    # Run demonstrations
    demonstrate_constraint_analysis(route_simulator, satellites, scenarios)
    demonstrate_budget_analysis(route_simulator, satellites, scenarios)
    demonstrate_timeline_analysis(route_simulator, satellites, scenarios)
    demonstrate_fitness_evaluation(route_simulator, satellites, scenarios)
    demonstrate_convergence_detection(route_simulator, satellites, scenarios)
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("✅ RouteSimulator successfully demonstrates:")
    print("   • Constraint handling for budget limits and timeline requirements")
    print("   • Convergence detection and early termination capabilities") 
    print("   • Genetic algorithm integration with cost calculations")
    print("   • Comprehensive fitness evaluation with penalty functions")
    print("   • Real-time optimization progress tracking")


if __name__ == "__main__":
    main()