#!/usr/bin/env python3
"""
Memory profiler for genetic route optimizer.

Provides detailed memory usage analysis during optimization runs.
"""

import psutil
import os
import time
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime


@dataclass
class MemorySnapshot:
    """Single memory usage snapshot."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    generation: Optional[int] = None
    phase: Optional[str] = None


class MemoryProfiler:
    """Profiles memory usage during genetic algorithm execution."""
    
    def __init__(self, sampling_interval: float = 0.1):
        """Initialize memory profiler.
        
        Args:
            sampling_interval: Time between memory samples in seconds
        """
        self.sampling_interval = sampling_interval
        self.snapshots: List[MemorySnapshot] = []
        self.process = psutil.Process(os.getpid())
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        self.current_generation = None
        self.current_phase = None
    
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.snapshots.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def set_generation(self, generation: int):
        """Set current generation for context."""
        self.current_generation = generation
    
    def set_phase(self, phase: str):
        """Set current phase for context."""
        self.current_phase = phase
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                
                snapshot = MemorySnapshot(
                    timestamp=time.time() - self.start_time,
                    rss_mb=memory_info.rss / 1024 / 1024,
                    vms_mb=memory_info.vms / 1024 / 1024,
                    percent=memory_percent,
                    generation=self.current_generation,
                    phase=self.current_phase
                )
                
                self.snapshots.append(snapshot)
                
            except Exception as e:
                print(f"Memory monitoring error: {e}")
            
            time.sleep(self.sampling_interval)
    
    def get_peak_memory(self) -> float:
        """Get peak RSS memory usage in MB."""
        if not self.snapshots:
            return 0.0
        return max(snapshot.rss_mb for snapshot in self.snapshots)
    
    def get_average_memory(self) -> float:
        """Get average RSS memory usage in MB."""
        if not self.snapshots:
            return 0.0
        return sum(snapshot.rss_mb for snapshot in self.snapshots) / len(self.snapshots)
    
    def get_memory_growth_rate(self) -> float:
        """Get memory growth rate in MB/second."""
        if len(self.snapshots) < 2:
            return 0.0
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        time_diff = last.timestamp - first.timestamp
        memory_diff = last.rss_mb - first.rss_mb
        
        return memory_diff / time_diff if time_diff > 0 else 0.0
    
    def analyze_memory_by_phase(self) -> Dict[str, Dict[str, float]]:
        """Analyze memory usage by optimization phase."""
        phase_stats = {}
        
        for snapshot in self.snapshots:
            if snapshot.phase:
                if snapshot.phase not in phase_stats:
                    phase_stats[snapshot.phase] = []
                phase_stats[snapshot.phase].append(snapshot.rss_mb)
        
        # Calculate statistics for each phase
        analysis = {}
        for phase, memory_values in phase_stats.items():
            analysis[phase] = {
                'avg_memory': sum(memory_values) / len(memory_values),
                'peak_memory': max(memory_values),
                'min_memory': min(memory_values),
                'memory_range': max(memory_values) - min(memory_values),
                'sample_count': len(memory_values)
            }
        
        return analysis
    
    def analyze_memory_by_generation(self) -> Dict[int, Dict[str, float]]:
        """Analyze memory usage by generation."""
        gen_stats = {}
        
        for snapshot in self.snapshots:
            if snapshot.generation is not None:
                if snapshot.generation not in gen_stats:
                    gen_stats[snapshot.generation] = []
                gen_stats[snapshot.generation].append(snapshot.rss_mb)
        
        # Calculate statistics for each generation
        analysis = {}
        for generation, memory_values in gen_stats.items():
            analysis[generation] = {
                'avg_memory': sum(memory_values) / len(memory_values),
                'peak_memory': max(memory_values),
                'min_memory': min(memory_values),
                'sample_count': len(memory_values)
            }
        
        return analysis
    
    def detect_memory_leaks(self, threshold_mb: float = 10.0) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        leaks = []
        
        if len(self.snapshots) < 10:
            return leaks
        
        # Look for sustained memory growth
        window_size = max(10, len(self.snapshots) // 10)
        
        for i in range(window_size, len(self.snapshots), window_size):
            start_idx = i - window_size
            end_idx = i
            
            start_memory = sum(s.rss_mb for s in self.snapshots[start_idx:start_idx+3]) / 3
            end_memory = sum(s.rss_mb for s in self.snapshots[end_idx-3:end_idx]) / 3
            
            growth = end_memory - start_memory
            time_span = self.snapshots[end_idx-1].timestamp - self.snapshots[start_idx].timestamp
            
            if growth > threshold_mb and time_span > 0:
                leaks.append({
                    'start_time': self.snapshots[start_idx].timestamp,
                    'end_time': self.snapshots[end_idx-1].timestamp,
                    'memory_growth_mb': growth,
                    'growth_rate_mb_per_sec': growth / time_span,
                    'start_generation': self.snapshots[start_idx].generation,
                    'end_generation': self.snapshots[end_idx-1].generation
                })
        
        return leaks
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        if not self.snapshots:
            return {'error': 'No memory data collected'}
        
        report = {
            'summary': {
                'total_samples': len(self.snapshots),
                'monitoring_duration': self.snapshots[-1].timestamp,
                'peak_memory_mb': self.get_peak_memory(),
                'average_memory_mb': self.get_average_memory(),
                'memory_growth_rate_mb_per_sec': self.get_memory_growth_rate(),
                'initial_memory_mb': self.snapshots[0].rss_mb,
                'final_memory_mb': self.snapshots[-1].rss_mb
            },
            'phase_analysis': self.analyze_memory_by_phase(),
            'generation_analysis': self.analyze_memory_by_generation(),
            'memory_leaks': self.detect_memory_leaks(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        peak_memory = self.get_peak_memory()
        growth_rate = self.get_memory_growth_rate()
        leaks = self.detect_memory_leaks()
        
        if peak_memory > 1000:  # > 1GB
            recommendations.append("High peak memory usage detected. Consider reducing population size or implementing memory-efficient data structures.")
        
        if growth_rate > 1.0:  # > 1MB/sec growth
            recommendations.append("Significant memory growth detected. Check for memory leaks in fitness evaluation or genetic operators.")
        
        if leaks:
            recommendations.append(f"Potential memory leaks detected in {len(leaks)} time periods. Review object lifecycle management.")
        
        phase_analysis = self.analyze_memory_by_phase()
        if phase_analysis:
            max_phase = max(phase_analysis.items(), key=lambda x: x[1]['peak_memory'])
            recommendations.append(f"Highest memory usage in '{max_phase[0]}' phase ({max_phase[1]['peak_memory']:.1f} MB). Focus optimization efforts here.")
        
        if not recommendations:
            recommendations.append("Memory usage appears optimal. No specific recommendations.")
        
        return recommendations
    
    def plot_memory_usage(self, filename: str = "memory_usage.png"):
        """Plot memory usage over time."""
        if not self.snapshots:
            print("No memory data to plot")
            return
        
        try:
            timestamps = [s.timestamp for s in self.snapshots]
            rss_memory = [s.rss_mb for s in self.snapshots]
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, rss_memory, 'b-', linewidth=1, alpha=0.7)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Memory Usage (MB)')
            plt.title('Memory Usage During Genetic Algorithm Execution')
            plt.grid(True, alpha=0.3)
            
            # Mark generation boundaries if available
            gen_changes = []
            current_gen = None
            for i, snapshot in enumerate(self.snapshots):
                if snapshot.generation != current_gen and snapshot.generation is not None:
                    gen_changes.append((snapshot.timestamp, snapshot.rss_mb, snapshot.generation))
                    current_gen = snapshot.generation
            
            if gen_changes:
                gen_times, gen_memory, gen_numbers = zip(*gen_changes)
                plt.scatter(gen_times, gen_memory, c='red', s=20, alpha=0.6, label='Generation Changes')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Memory usage plot saved to {filename}")
            
        except ImportError:
            print("matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plot: {e}")
    
    def save_raw_data(self, filename: str = "memory_data.csv"):
        """Save raw memory data to CSV file."""
        if not self.snapshots:
            print("No memory data to save")
            return
        
        try:
            with open(filename, 'w') as f:
                f.write("timestamp,rss_mb,vms_mb,percent,generation,phase\n")
                for snapshot in self.snapshots:
                    f.write(f"{snapshot.timestamp:.3f},{snapshot.rss_mb:.2f},{snapshot.vms_mb:.2f},"
                           f"{snapshot.percent:.2f},{snapshot.generation or ''},"
                           f"{snapshot.phase or ''}\n")
            
            print(f"Memory data saved to {filename}")
            
        except Exception as e:
            print(f"Error saving memory data: {e}")


# Context manager for easy profiling
class ProfiledOptimization:
    """Context manager for profiling genetic algorithm optimization."""
    
    def __init__(self, profiler: MemoryProfiler):
        self.profiler = profiler
    
    def __enter__(self):
        self.profiler.start_monitoring()
        return self.profiler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop_monitoring()


def main():
    """Demonstrate memory profiler usage."""
    profiler = MemoryProfiler(sampling_interval=0.05)
    
    print("Testing memory profiler...")
    
    with ProfiledOptimization(profiler) as prof:
        # Simulate optimization phases
        prof.set_phase("initialization")
        time.sleep(0.5)
        
        for gen in range(5):
            prof.set_generation(gen)
            prof.set_phase("fitness_evaluation")
            time.sleep(0.2)
            
            prof.set_phase("selection")
            time.sleep(0.1)
            
            prof.set_phase("crossover")
            time.sleep(0.1)
            
            prof.set_phase("mutation")
            time.sleep(0.1)
    
    # Generate report
    report = profiler.generate_report()
    
    print("\nMemory Usage Report:")
    print(f"Peak Memory: {report['summary']['peak_memory_mb']:.1f} MB")
    print(f"Average Memory: {report['summary']['average_memory_mb']:.1f} MB")
    print(f"Growth Rate: {report['summary']['memory_growth_rate_mb_per_sec']:.3f} MB/sec")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    # Save data
    profiler.save_raw_data("test_memory_data.csv")
    profiler.plot_memory_usage("test_memory_plot.png")


if __name__ == "__main__":
    main()