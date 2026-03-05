#!/usr/bin/env python3
"""
GDS Benchmark Results Visualization

This script parses and visualizes GDS benchmark results from gdsio output.
It creates comprehensive plots comparing GDS vs Non-GDS performance across
different IO sizes and operation types.

Usage:
    python visualize_gds_results.py [--input INPUT_FILE] [--output OUTPUT_DIR]
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GDSBenchmarkResult:
    """Stores a single benchmark result."""
    test_name: str
    io_type: str  # READ or WRITE
    xfer_type: str  # CPUONLY (Non-GDS) or GPUD (GDS)
    threads: int
    io_size_kb: int
    throughput_gibs: float
    latency_us: float
    ops: int
    total_time_secs: float
    
    @property
    def is_gds(self) -> bool:
        return self.xfer_type == "GPUD"
    
    @property
    def io_size_str(self) -> str:
        """Convert IO size to human readable string."""
        if self.io_size_kb >= 1024:
            return f"{self.io_size_kb // 1024}M"
        return f"{self.io_size_kb}K"


def parse_gds_results(filepath: str) -> List[GDSBenchmarkResult]:
    """Parse GDS benchmark results from file."""
    results = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to match result lines
    # Example: IoType: WRITE XferType: CPUONLY Threads: 8 DataSetSize: 1274953728/4096000(KiB) IOSize: 256(KiB) Throughput: 10.188280 GiB/sec, Avg_Latency: 191.702182 usecs ops: 4980288 total_time 119.342086 secs
    pattern = r'IoType:\s+(\w+)\s+XferType:\s+(\w+)\s+Threads:\s+(\d+)\s+DataSetSize:\s+[\d/]+\(KiB\)\s+IOSize:\s+(\d+)\(KiB\)\s+Throughput:\s+([\d.]+)\s+GiB/sec,\s+Avg_Latency:\s+([\d.]+)\s+usecs\s+ops:\s+(\d+)\s+total_time\s+([\d.]+)\s+secs'
    
    for match in re.finditer(pattern, content):
        io_type = match.group(1)
        xfer_type = match.group(2)
        threads = int(match.group(3))
        io_size_kb = int(match.group(4))
        throughput = float(match.group(5))
        latency = float(match.group(6))
        ops = int(match.group(7))
        total_time = float(match.group(8))
        
        # Determine test name
        gds_str = "GDS" if xfer_type == "GPUD" else "Non-GDS"
        test_name = f"{gds_str} {io_type}"
        
        results.append(GDSBenchmarkResult(
            test_name=test_name,
            io_type=io_type,
            xfer_type=xfer_type,
            threads=threads,
            io_size_kb=io_size_kb,
            throughput_gibs=throughput,
            latency_us=latency,
            ops=ops,
            total_time_secs=total_time,
        ))
    
    return results


def organize_results(results: List[GDSBenchmarkResult]) -> Dict:
    """Organize results by IO size and type."""
    organized = {
        'io_sizes': sorted(set(r.io_size_kb for r in results)),
        'gds_read': [],
        'gds_write': [],
        'non_gds_read': [],
        'non_gds_write': [],
    }
    
    for io_size in organized['io_sizes']:
        for r in results:
            if r.io_size_kb != io_size:
                continue
            
            if r.is_gds:
                if r.io_type == "READ":
                    organized['gds_read'].append(r)
                else:
                    organized['gds_write'].append(r)
            else:
                if r.io_type == "READ":
                    organized['non_gds_read'].append(r)
                else:
                    organized['non_gds_write'].append(r)
    
    return organized


def plot_throughput_comparison(data: Dict, output_dir: str):
    """Plot throughput comparison between GDS and Non-GDS."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    io_sizes = data['io_sizes']
    io_size_labels = [f"{s//1024}M" if s >= 1024 else f"{s}K" for s in io_sizes]
    x = np.arange(len(io_sizes))
    width = 0.35
    
    # Read throughput
    ax = axes[0]
    gds_read = [r.throughput_gibs for r in sorted(data['gds_read'], key=lambda x: x.io_size_kb)]
    non_gds_read = [r.throughput_gibs for r in sorted(data['non_gds_read'], key=lambda x: x.io_size_kb)]
    
    bars1 = ax.bar(x - width/2, non_gds_read, width, label='Non-GDS (CPU)', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, gds_read, width, label='GDS (GPU Direct)', color='seagreen', alpha=0.8)
    
    ax.set_xlabel('IO Size', fontsize=12)
    ax.set_ylabel('Throughput (GiB/s)', fontsize=12)
    ax.set_title('READ Throughput: GDS vs Non-GDS', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(io_size_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Write throughput
    ax = axes[1]
    gds_write = [r.throughput_gibs for r in sorted(data['gds_write'], key=lambda x: x.io_size_kb)]
    non_gds_write = [r.throughput_gibs for r in sorted(data['non_gds_write'], key=lambda x: x.io_size_kb)]
    
    bars1 = ax.bar(x - width/2, non_gds_write, width, label='Non-GDS (CPU)', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, gds_write, width, label='GDS (GPU Direct)', color='seagreen', alpha=0.8)
    
    ax.set_xlabel('IO Size', fontsize=12)
    ax.set_ylabel('Throughput (GiB/s)', fontsize=12)
    ax.set_title('WRITE Throughput: GDS vs Non-GDS', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(io_size_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_latency_comparison(data: Dict, output_dir: str):
    """Plot latency comparison between GDS and Non-GDS."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    io_sizes = data['io_sizes']
    io_size_labels = [f"{s//1024}M" if s >= 1024 else f"{s}K" for s in io_sizes]
    x = np.arange(len(io_sizes))
    width = 0.35
    
    # Read latency
    ax = axes[0]
    gds_read = [r.latency_us for r in sorted(data['gds_read'], key=lambda x: x.io_size_kb)]
    non_gds_read = [r.latency_us for r in sorted(data['non_gds_read'], key=lambda x: x.io_size_kb)]
    
    bars1 = ax.bar(x - width/2, non_gds_read, width, label='Non-GDS (CPU)', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, gds_read, width, label='GDS (GPU Direct)', color='seagreen', alpha=0.8)
    
    ax.set_xlabel('IO Size', fontsize=12)
    ax.set_ylabel('Latency (μs)', fontsize=12)
    ax.set_title('READ Latency: GDS vs Non-GDS', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(io_size_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Write latency
    ax = axes[1]
    gds_write = [r.latency_us for r in sorted(data['gds_write'], key=lambda x: x.io_size_kb)]
    non_gds_write = [r.latency_us for r in sorted(data['non_gds_write'], key=lambda x: x.io_size_kb)]
    
    bars1 = ax.bar(x - width/2, non_gds_write, width, label='Non-GDS (CPU)', color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, gds_write, width, label='GDS (GPU Direct)', color='seagreen', alpha=0.8)
    
    ax.set_xlabel('IO Size', fontsize=12)
    ax.set_ylabel('Latency (μs)', fontsize=12)
    ax.set_title('WRITE Latency: GDS vs Non-GDS', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(io_size_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_throughput_vs_io_size(data: Dict, output_dir: str):
    """Plot throughput vs IO size as line chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    io_sizes_kb = data['io_sizes']
    io_sizes_mb = [s / 1024 for s in io_sizes_kb]  # Convert to MB for x-axis
    
    # Sort results by IO size
    gds_read = sorted(data['gds_read'], key=lambda x: x.io_size_kb)
    gds_write = sorted(data['gds_write'], key=lambda x: x.io_size_kb)
    non_gds_read = sorted(data['non_gds_read'], key=lambda x: x.io_size_kb)
    non_gds_write = sorted(data['non_gds_write'], key=lambda x: x.io_size_kb)
    
    # Plot lines
    ax.plot(io_sizes_mb, [r.throughput_gibs for r in non_gds_read], 'o-', 
            label='Non-GDS READ', linewidth=2, markersize=8, color='coral')
    ax.plot(io_sizes_mb, [r.throughput_gibs for r in non_gds_write], 's--', 
            label='Non-GDS WRITE', linewidth=2, markersize=8, color='salmon')
    ax.plot(io_sizes_mb, [r.throughput_gibs for r in gds_read], '^-', 
            label='GDS READ', linewidth=2, markersize=8, color='seagreen')
    ax.plot(io_sizes_mb, [r.throughput_gibs for r in gds_write], 'D--', 
            label='GDS WRITE', linewidth=2, markersize=8, color='lightgreen')
    
    ax.set_xlabel('IO Size (MiB)', fontsize=12)
    ax.set_ylabel('Throughput (GiB/s)', fontsize=12)
    ax.set_title('GDS Benchmark: Throughput vs IO Size', fontsize=14)
    ax.legend(fontsize=10, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Set x-ticks to actual IO sizes
    ax.set_xticks(io_sizes_mb)
    ax.set_xticklabels([f"{s:.0f}M" if s >= 1 else f"{s*1024:.0f}K" for s in io_sizes_mb])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_vs_io_size.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_latency_vs_io_size(data: Dict, output_dir: str):
    """Plot latency vs IO size as line chart."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    io_sizes_kb = data['io_sizes']
    io_sizes_mb = [s / 1024 for s in io_sizes_kb]
    
    # Sort results by IO size
    gds_read = sorted(data['gds_read'], key=lambda x: x.io_size_kb)
    gds_write = sorted(data['gds_write'], key=lambda x: x.io_size_kb)
    non_gds_read = sorted(data['non_gds_read'], key=lambda x: x.io_size_kb)
    non_gds_write = sorted(data['non_gds_write'], key=lambda x: x.io_size_kb)
    
    # Plot lines
    ax.plot(io_sizes_mb, [r.latency_us for r in non_gds_read], 'o-', 
            label='Non-GDS READ', linewidth=2, markersize=8, color='coral')
    ax.plot(io_sizes_mb, [r.latency_us for r in non_gds_write], 's--', 
            label='Non-GDS WRITE', linewidth=2, markersize=8, color='salmon')
    ax.plot(io_sizes_mb, [r.latency_us for r in gds_read], '^-', 
            label='GDS READ', linewidth=2, markersize=8, color='seagreen')
    ax.plot(io_sizes_mb, [r.latency_us for r in gds_write], 'D--', 
            label='GDS WRITE', linewidth=2, markersize=8, color='lightgreen')
    
    ax.set_xlabel('IO Size (MiB)', fontsize=12)
    ax.set_ylabel('Latency (μs)', fontsize=12)
    ax.set_title('GDS Benchmark: Latency vs IO Size', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    
    ax.set_xticks(io_sizes_mb)
    ax.set_xticklabels([f"{s:.0f}M" if s >= 1 else f"{s*1024:.0f}K" for s in io_sizes_mb])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_vs_io_size.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_gds_speedup(data: Dict, output_dir: str):
    """Plot GDS speedup over Non-GDS."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    io_sizes = data['io_sizes']
    io_size_labels = [f"{s//1024}M" if s >= 1024 else f"{s}K" for s in io_sizes]
    
    # Calculate speedups
    gds_read = sorted(data['gds_read'], key=lambda x: x.io_size_kb)
    non_gds_read = sorted(data['non_gds_read'], key=lambda x: x.io_size_kb)
    gds_write = sorted(data['gds_write'], key=lambda x: x.io_size_kb)
    non_gds_write = sorted(data['non_gds_write'], key=lambda x: x.io_size_kb)
    
    read_speedup = [g.throughput_gibs / ng.throughput_gibs 
                    for g, ng in zip(gds_read, non_gds_read)]
    write_speedup = [g.throughput_gibs / ng.throughput_gibs 
                     for g, ng in zip(gds_write, non_gds_write)]
    
    x = np.arange(len(io_sizes))
    width = 0.35
    
    # Color bars based on speedup (>1 is good, <1 is bad)
    colors_read = ['seagreen' if s >= 1 else 'coral' for s in read_speedup]
    colors_write = ['seagreen' if s >= 1 else 'coral' for s in write_speedup]
    
    bars1 = ax.bar(x - width/2, read_speedup, width, label='READ', color=colors_read, alpha=0.8)
    bars2 = ax.bar(x + width/2, write_speedup, width, label='WRITE', color=colors_write, alpha=0.8)
    
    # Add baseline at 1.0
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Baseline (1.0x)')
    
    ax.set_xlabel('IO Size', fontsize=12)
    ax.set_ylabel('Speedup (GDS / Non-GDS)', fontsize=12)
    ax.set_title('GDS Speedup over Non-GDS\n(>1.0 means GDS is faster)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(io_size_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}x',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gds_speedup.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_in_one(data: Dict, output_dir: str):
    """Create a comprehensive all-in-one plot."""
    fig = plt.figure(figsize=(16, 12))
    
    io_sizes_kb = data['io_sizes']
    io_sizes_mb = [s / 1024 for s in io_sizes_kb]
    io_size_labels = [f"{s//1024}M" if s >= 1024 else f"{s}K" for s in io_sizes_kb]
    
    # Sort results
    gds_read = sorted(data['gds_read'], key=lambda x: x.io_size_kb)
    gds_write = sorted(data['gds_write'], key=lambda x: x.io_size_kb)
    non_gds_read = sorted(data['non_gds_read'], key=lambda x: x.io_size_kb)
    non_gds_write = sorted(data['non_gds_write'], key=lambda x: x.io_size_kb)
    
    # Subplot 1: Throughput comparison
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(io_sizes_kb))
    width = 0.2
    
    ax1.bar(x - 1.5*width, [r.throughput_gibs for r in non_gds_read], width, 
            label='Non-GDS READ', color='coral', alpha=0.8)
    ax1.bar(x - 0.5*width, [r.throughput_gibs for r in non_gds_write], width, 
            label='Non-GDS WRITE', color='salmon', alpha=0.8)
    ax1.bar(x + 0.5*width, [r.throughput_gibs for r in gds_read], width, 
            label='GDS READ', color='seagreen', alpha=0.8)
    ax1.bar(x + 1.5*width, [r.throughput_gibs for r in gds_write], width, 
            label='GDS WRITE', color='lightgreen', alpha=0.8)
    
    ax1.set_xlabel('IO Size', fontsize=11)
    ax1.set_ylabel('Throughput (GiB/s)', fontsize=11)
    ax1.set_title('Throughput Comparison', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(io_size_labels, rotation=45)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Latency comparison
    ax2 = fig.add_subplot(2, 2, 2)
    
    ax2.bar(x - 1.5*width, [r.latency_us for r in non_gds_read], width, 
            label='Non-GDS READ', color='coral', alpha=0.8)
    ax2.bar(x - 0.5*width, [r.latency_us for r in non_gds_write], width, 
            label='Non-GDS WRITE', color='salmon', alpha=0.8)
    ax2.bar(x + 0.5*width, [r.latency_us for r in gds_read], width, 
            label='GDS READ', color='seagreen', alpha=0.8)
    ax2.bar(x + 1.5*width, [r.latency_us for r in gds_write], width, 
            label='GDS WRITE', color='lightgreen', alpha=0.8)
    
    ax2.set_xlabel('IO Size', fontsize=11)
    ax2.set_ylabel('Latency (μs)', fontsize=11)
    ax2.set_title('Latency Comparison (log scale)', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(io_size_labels, rotation=45)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    # Subplot 3: Throughput vs IO Size (line)
    ax3 = fig.add_subplot(2, 2, 3)
    
    ax3.plot(io_sizes_mb, [r.throughput_gibs for r in non_gds_read], 'o-', 
             label='Non-GDS READ', linewidth=2, markersize=6, color='coral')
    ax3.plot(io_sizes_mb, [r.throughput_gibs for r in non_gds_write], 's--', 
             label='Non-GDS WRITE', linewidth=2, markersize=6, color='salmon')
    ax3.plot(io_sizes_mb, [r.throughput_gibs for r in gds_read], '^-', 
             label='GDS READ', linewidth=2, markersize=6, color='seagreen')
    ax3.plot(io_sizes_mb, [r.throughput_gibs for r in gds_write], 'D--', 
             label='GDS WRITE', linewidth=2, markersize=6, color='lightgreen')
    
    ax3.set_xlabel('IO Size (MiB)', fontsize=11)
    ax3.set_ylabel('Throughput (GiB/s)', fontsize=11)
    ax3.set_title('Throughput vs IO Size', fontsize=13)
    ax3.legend(fontsize=8, loc='center right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(io_sizes_mb)
    ax3.set_xticklabels([f"{s:.0f}M" if s >= 1 else f"{s*1024:.0f}K" for s in io_sizes_mb])
    
    # Subplot 4: GDS Speedup
    ax4 = fig.add_subplot(2, 2, 4)
    
    read_speedup = [g.throughput_gibs / ng.throughput_gibs 
                    for g, ng in zip(gds_read, non_gds_read)]
    write_speedup = [g.throughput_gibs / ng.throughput_gibs 
                     for g, ng in zip(gds_write, non_gds_write)]
    
    colors_read = ['seagreen' if s >= 1 else 'coral' for s in read_speedup]
    colors_write = ['lightgreen' if s >= 1 else 'salmon' for s in write_speedup]
    
    ax4.bar(x - width/2, read_speedup, width, label='READ Speedup', color=colors_read, alpha=0.8)
    ax4.bar(x + width/2, write_speedup, width, label='WRITE Speedup', color=colors_write, alpha=0.8)
    ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5)
    
    ax4.set_xlabel('IO Size', fontsize=11)
    ax4.set_ylabel('Speedup (GDS / Non-GDS)', fontsize=11)
    ax4.set_title('GDS Speedup (>1.0 = GDS faster)', fontsize=13)
    ax4.set_xticks(x)
    ax4.set_xticklabels(io_size_labels, rotation=45)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('GDS Benchmark Results Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gds_benchmark_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


def print_summary_table(data: Dict):
    """Print a summary table of results."""
    print("\n" + "="*100)
    print("GDS BENCHMARK RESULTS SUMMARY")
    print("="*100)
    
    # Header
    print(f"\n{'IO Size':<10} {'Type':<15} {'Non-GDS (GiB/s)':<18} {'GDS (GiB/s)':<15} {'Speedup':<10} {'Non-GDS Lat(μs)':<18} {'GDS Lat(μs)':<15}")
    print("-"*100)
    
    io_sizes = data['io_sizes']
    
    for io_size in io_sizes:
        io_size_str = f"{io_size//1024}M" if io_size >= 1024 else f"{io_size}K"
        
        # Find matching results
        gds_read = next((r for r in data['gds_read'] if r.io_size_kb == io_size), None)
        non_gds_read = next((r for r in data['non_gds_read'] if r.io_size_kb == io_size), None)
        gds_write = next((r for r in data['gds_write'] if r.io_size_kb == io_size), None)
        non_gds_write = next((r for r in data['non_gds_write'] if r.io_size_kb == io_size), None)
        
        if gds_read and non_gds_read:
            speedup = gds_read.throughput_gibs / non_gds_read.throughput_gibs
            print(f"{io_size_str:<10} {'READ':<15} {non_gds_read.throughput_gibs:<18.2f} {gds_read.throughput_gibs:<15.2f} {speedup:<10.2f}x {non_gds_read.latency_us:<18.1f} {gds_read.latency_us:<15.1f}")
        
        if gds_write and non_gds_write:
            speedup = gds_write.throughput_gibs / non_gds_write.throughput_gibs
            print(f"{io_size_str:<10} {'WRITE':<15} {non_gds_write.throughput_gibs:<18.2f} {gds_write.throughput_gibs:<15.2f} {speedup:<10.2f}x {non_gds_write.latency_us:<18.1f} {gds_write.latency_us:<15.1f}")
        
        print("-"*100)
    
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description="Visualize GDS Benchmark Results")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="gds_results_sdb1.txt",
        help="Input file containing GDS benchmark results",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="figures",
        help="Output directory for plots",
    )
    
    args = parser.parse_args()
    
    # Get the directory of the input file
    input_dir = os.path.dirname(os.path.abspath(args.input))
    output_dir = os.path.join(input_dir, args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Parsing results from: {args.input}")
    results = parse_gds_results(args.input)
    print(f"Found {len(results)} benchmark results")
    
    if not results:
        print("No results found. Please check the input file format.")
        return
    
    # Organize results
    data = organize_results(results)
    
    # Print summary table
    print_summary_table(data)
    
    # Generate plots
    print(f"\nGenerating plots in: {output_dir}")
    
    plot_throughput_comparison(data, output_dir)
    print("  - throughput_comparison.png")
    
    plot_latency_comparison(data, output_dir)
    print("  - latency_comparison.png")
    
    plot_throughput_vs_io_size(data, output_dir)
    print("  - throughput_vs_io_size.png")
    
    plot_latency_vs_io_size(data, output_dir)
    print("  - latency_vs_io_size.png")
    
    plot_gds_speedup(data, output_dir)
    print("  - gds_speedup.png")
    
    plot_all_in_one(data, output_dir)
    print("  - gds_benchmark_summary.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()