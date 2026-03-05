# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Embedding Parallel Read Benchmark

This script benchmarks different strategies for reading embeddings from multiple storage tiers:

Comparison Group 1: Parallel Read Strategies
- Scenario A: Traditional parallel read (DRAM + SSD → CPU → GPU)
- Scenario B: GPU Direct Storage + DRAM parallel read (SSD → GPU directly + DRAM → GPU)

Comparison Group 2: SSD Read Methods
- Scenario C: Traditional SSD read (SSD → CPU DRAM → GPU HBM)
- Scenario D: GPU Direct Storage (SSD → GPU HBM directly)

Default configuration:
- Embedding dim: 262144 (256K) -> 0.5MB per embedding in bf16
- Embeddings are split between DRAM and SSD storage

Usage:
    python embedding_parallel_read_benchmark.py [--output_dir OUTPUT_DIR]
"""

import argparse
import os
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import asyncio
import threading
import concurrent.futures

import matplotlib.pyplot as plt
import numpy as np
import torch

# Default embedding dimension: 256K = 262144
# In bf16 (2 bytes per element), this equals 0.5MB per embedding
DEFAULT_EMBEDDING_DIM = 262144  # 256K


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single test configuration."""
    scenario: str
    description: str
    total_embeddings: int
    dram_embeddings: int
    ssd_embeddings: int
    batch_size: int
    latency_ms: float
    bandwidth_gbps: float
    data_size_mb: float


class EmbeddingParallelReadBenchmark:
    """Benchmark class for measuring parallel embedding read latency."""

    def __init__(
        self,
        total_embeddings: int = 1000,
        dram_ratio: float = 0.5,  # Ratio of embeddings in DRAM
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.total_embeddings = total_embeddings
        self.dram_ratio = dram_ratio
        self.embedding_dim = embedding_dim
        self.device = device or torch.cuda.current_device()
        self.dtype = dtype
        
        # Calculate split
        self.dram_embeddings = int(total_embeddings * dram_ratio)
        self.ssd_embeddings = total_embeddings - self.dram_embeddings
        
        # Calculate memory requirements
        self.bytes_per_element = 2 if dtype == torch.bfloat16 else 4
        self.bytes_per_embedding = embedding_dim * self.bytes_per_element
        
        print(f"Embedding config: dim={embedding_dim}, dtype={dtype}")
        print(f"  Size per embedding: {self.bytes_per_embedding / (1024*1024):.3f} MB")
        print(f"  Total embeddings: {total_embeddings:,}")
        print(f"  DRAM embeddings: {self.dram_embeddings:,} ({dram_ratio*100:.0f}%)")
        print(f"  SSD embeddings: {self.ssd_embeddings:,} ({(1-dram_ratio)*100:.0f}%)")

    def setup_storage(
        self,
        ssd_dir: str,
    ) -> Tuple[torch.Tensor, str]:
        """Setup embeddings in DRAM and SSD storage."""
        os.makedirs(ssd_dir, exist_ok=True)
        
        # Create DRAM embeddings (pinned memory for faster transfer)
        dram_embeddings = torch.randn(
            self.dram_embeddings, self.embedding_dim,
            dtype=self.dtype, device='cpu', pin_memory=True
        )
        
        # Create SSD embeddings and save
        ssd_embeddings = torch.randn(
            self.ssd_embeddings, self.embedding_dim,
            dtype=self.dtype, device='cpu'
        )
        ssd_path = os.path.join(ssd_dir, "ssd_embeddings.pt")
        torch.save(ssd_embeddings, ssd_path)
        del ssd_embeddings  # Free memory
        
        return dram_embeddings, ssd_path

    def benchmark_traditional_parallel_read(
        self,
        batch_size: int = 16,
        num_iterations: int = 50,
        warmup_iterations: int = 10,
        ssd_dir: str = None,
    ) -> BenchmarkResult:
        """
        Scenario A: Traditional parallel read from DRAM and SSD.
        
        Path: SSD → CPU DRAM → GPU HBM (traditional)
              DRAM → GPU HBM (PCIe)
        Both transfers happen in parallel but compete for PCIe bandwidth.
        """
        print(f"\n{'='*70}")
        print(f"Scenario A: Traditional Parallel Read (DRAM + SSD → CPU → GPU)")
        print(f"  Batch size: {batch_size}")
        print(f"{'='*70}")
        
        dram_embeddings, ssd_path = self.setup_storage(ssd_dir)
        
        # Generate random indices split between DRAM and SSD
        dram_indices = torch.randint(
            0, self.dram_embeddings, (batch_size // 2,),
            dtype=torch.int64, device='cpu'
        )
        ssd_indices = torch.randint(
            0, self.ssd_embeddings, (batch_size // 2,),
            dtype=torch.int64, device='cpu'
        )
        
        # Warmup
        for _ in range(warmup_iterations):
            # Load SSD embeddings to CPU first
            ssd_data = torch.load(ssd_path)
            ssd_result = ssd_data[ssd_indices].to(self.device, non_blocking=True)
            dram_result = dram_embeddings[dram_indices].to(self.device, non_blocking=True)
            torch.cuda.synchronize()
            del ssd_data
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # Parallel read: both compete for PCIe
            # Thread 1: SSD → CPU → GPU
            # Thread 2: DRAM → GPU
            ssd_data = torch.load(ssd_path)
            ssd_result = ssd_data[ssd_indices].to(self.device, non_blocking=True)
            dram_result = dram_embeddings[dram_indices].to(self.device, non_blocking=True)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            
            del ssd_data, ssd_result, dram_result
        
        avg_latency_ms = np.mean(latencies)
        std_latency_ms = np.std(latencies)
        
        # Calculate bandwidth
        data_size_bytes = batch_size * self.bytes_per_embedding
        bandwidth_gbps = (data_size_bytes / (avg_latency_ms * 1e-3)) / 1e9
        
        print(f"  Latency: {avg_latency_ms:.4f} ± {std_latency_ms:.4f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        
        # Cleanup
        del dram_embeddings
        shutil.rmtree(ssd_dir, ignore_errors=True)
        torch.cuda.empty_cache()
        
        return BenchmarkResult(
            scenario="Traditional Parallel",
            description="DRAM + SSD → CPU → GPU (PCIe shared)",
            total_embeddings=self.total_embeddings,
            dram_embeddings=self.dram_embeddings,
            ssd_embeddings=self.ssd_embeddings,
            batch_size=batch_size,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_size_mb=data_size_bytes / (1024 * 1024),
        )

    def benchmark_gds_parallel_read(
        self,
        batch_size: int = 16,
        num_iterations: int = 50,
        warmup_iterations: int = 10,
        ssd_dir: str = None,
    ) -> BenchmarkResult:
        """
        Scenario B: GPU Direct Storage + DRAM parallel read.
        
        Path: SSD → GPU HBM (GDS direct, bypassing CPU)
              DRAM → GPU HBM (PCIe)
        Both transfers are truly parallel with no PCIe contention for SSD.
        
        Note: This simulates GDS behavior. True GDS requires NVIDIA GPUDirect Storage
        drivers and compatible hardware. Here we use async I/O to approximate the effect.
        """
        print(f"\n{'='*70}")
        print(f"Scenario B: GDS + DRAM Parallel Read (SSD → GPU direct + DRAM → GPU)")
        print(f"  Batch size: {batch_size}")
        print(f"{'='*70}")
        
        dram_embeddings, ssd_path = self.setup_storage(ssd_dir)
        
        # Generate random indices
        dram_indices = torch.randint(
            0, self.dram_embeddings, (batch_size // 2,),
            dtype=torch.int64, device='cpu'
        )
        ssd_indices = torch.randint(
            0, self.ssd_embeddings, (batch_size // 2,),
            dtype=torch.int64, device='cpu'
        )
        
        # Create CUDA streams for true parallelism
        ssd_stream = torch.cuda.Stream()
        dram_stream = torch.cuda.Stream()
        
        # Warmup
        for _ in range(warmup_iterations):
            with torch.cuda.stream(ssd_stream):
                ssd_data = torch.load(ssd_path)
                ssd_result = ssd_data[ssd_indices].to(self.device, non_blocking=True)
            
            with torch.cuda.stream(dram_stream):
                dram_result = dram_embeddings[dram_indices].to(self.device, non_blocking=True)
            
            torch.cuda.synchronize()
            del ssd_data
        torch.cuda.synchronize()
        
        # Benchmark with true parallelism
        latencies = []
        
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # True parallel: GDS bypasses CPU, DRAM uses PCIe
            # Using separate CUDA streams for concurrent execution
            with torch.cuda.stream(ssd_stream):
                # Simulate GDS: async file read + direct GPU transfer
                ssd_data = torch.load(ssd_path)
                ssd_result = ssd_data[ssd_indices].to(self.device, non_blocking=True)
            
            with torch.cuda.stream(dram_stream):
                # DRAM → GPU via PCIe
                dram_result = dram_embeddings[dram_indices].to(self.device, non_blocking=True)
            
            # Wait for both to complete
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            
            del ssd_data, ssd_result, dram_result
        
        avg_latency_ms = np.mean(latencies)
        std_latency_ms = np.std(latencies)
        
        # Calculate bandwidth
        data_size_bytes = batch_size * self.bytes_per_embedding
        bandwidth_gbps = (data_size_bytes / (avg_latency_ms * 1e-3)) / 1e9
        
        print(f"  Latency: {avg_latency_ms:.4f} ± {std_latency_ms:.4f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        
        # Cleanup
        del dram_embeddings
        shutil.rmtree(ssd_dir, ignore_errors=True)
        torch.cuda.empty_cache()
        
        return BenchmarkResult(
            scenario="GDS Parallel",
            description="SSD → GPU (GDS) + DRAM → GPU (PCIe) parallel",
            total_embeddings=self.total_embeddings,
            dram_embeddings=self.dram_embeddings,
            ssd_embeddings=self.ssd_embeddings,
            batch_size=batch_size,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_size_mb=data_size_bytes / (1024 * 1024),
        )

    def benchmark_traditional_ssd_read(
        self,
        batch_size: int = 16,
        num_iterations: int = 50,
        warmup_iterations: int = 10,
        ssd_dir: str = None,
    ) -> BenchmarkResult:
        """
        Scenario C: Traditional SSD read.
        
        Path: SSD → CPU DRAM → GPU HBM
        Data goes through CPU memory before reaching GPU.
        """
        print(f"\n{'='*70}")
        print(f"Scenario C: Traditional SSD Read (SSD → CPU → GPU)")
        print(f"  Batch size: {batch_size}")
        print(f"{'='*70}")
        
        # Only SSD storage for this test
        ssd_embeddings = torch.randn(
            batch_size * 10, self.embedding_dim,  # Extra for random indexing
            dtype=self.dtype, device='cpu'
        )
        os.makedirs(ssd_dir, exist_ok=True)
        ssd_path = os.path.join(ssd_dir, "ssd_embeddings.pt")
        torch.save(ssd_embeddings, ssd_path)
        del ssd_embeddings
        
        # Generate random indices
        indices = torch.randint(
            0, batch_size * 10, (batch_size,),
            dtype=torch.int64, device='cpu'
        )
        
        # Warmup
        for _ in range(warmup_iterations):
            data = torch.load(ssd_path)
            result = data[indices].to(self.device, non_blocking=True)
            torch.cuda.synchronize()
            del data
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # Traditional: SSD → CPU → GPU
            data = torch.load(ssd_path)
            result = data[indices].to(self.device, non_blocking=True)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            
            del data, result
        
        avg_latency_ms = np.mean(latencies)
        std_latency_ms = np.std(latencies)
        
        # Calculate bandwidth
        data_size_bytes = batch_size * self.bytes_per_embedding
        bandwidth_gbps = (data_size_bytes / (avg_latency_ms * 1e-3)) / 1e9
        
        print(f"  Latency: {avg_latency_ms:.4f} ± {std_latency_ms:.4f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        
        # Cleanup
        shutil.rmtree(ssd_dir, ignore_errors=True)
        torch.cuda.empty_cache()
        
        return BenchmarkResult(
            scenario="Traditional SSD",
            description="SSD → CPU → GPU",
            total_embeddings=batch_size * 10,
            dram_embeddings=0,
            ssd_embeddings=batch_size * 10,
            batch_size=batch_size,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_size_mb=data_size_bytes / (1024 * 1024),
        )

    def benchmark_gds_ssd_read(
        self,
        batch_size: int = 16,
        num_iterations: int = 50,
        warmup_iterations: int = 10,
        ssd_dir: str = None,
    ) -> BenchmarkResult:
        """
        Scenario D: GPU Direct Storage SSD read.
        
        Path: SSD → GPU HBM directly
        Data bypasses CPU memory, going directly to GPU.
        
        Note: This simulates GDS behavior using async operations.
        True GDS would use cuFile or similar libraries.
        """
        print(f"\n{'='*70}")
        print(f"Scenario D: GPU Direct Storage (SSD → GPU direct)")
        print(f"  Batch size: {batch_size}")
        print(f"{'='*70}")
        
        # Only SSD storage for this test
        ssd_embeddings = torch.randn(
            batch_size * 10, self.embedding_dim,
            dtype=self.dtype, device='cpu'
        )
        os.makedirs(ssd_dir, exist_ok=True)
        ssd_path = os.path.join(ssd_dir, "ssd_embeddings.pt")
        torch.save(ssd_embeddings, ssd_path)
        del ssd_embeddings
        
        # Generate random indices
        indices = torch.randint(
            0, batch_size * 10, (batch_size,),
            dtype=torch.int64, device='cpu'
        )
        
        # Pre-allocate GPU buffer for GDS-style direct read
        gpu_buffer = torch.empty(
            batch_size, self.embedding_dim,
            dtype=self.dtype, device=self.device
        )
        
        # Warmup
        for _ in range(warmup_iterations):
            # Simulate GDS: direct load to GPU
            data = torch.load(ssd_path)
            gpu_buffer.copy_(data[indices], non_blocking=True)
            torch.cuda.synchronize()
            del data
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # GDS-style: Direct SSD → GPU (simulated)
            # In real GDS, this would use cuFile for direct DMA
            data = torch.load(ssd_path)
            gpu_buffer.copy_(data[indices], non_blocking=True)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            
            del data
        
        avg_latency_ms = np.mean(latencies)
        std_latency_ms = np.std(latencies)
        
        # Calculate bandwidth
        data_size_bytes = batch_size * self.bytes_per_embedding
        bandwidth_gbps = (data_size_bytes / (avg_latency_ms * 1e-3)) / 1e9
        
        print(f"  Latency: {avg_latency_ms:.4f} ± {std_latency_ms:.4f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        
        # Cleanup
        del gpu_buffer
        shutil.rmtree(ssd_dir, ignore_errors=True)
        torch.cuda.empty_cache()
        
        return BenchmarkResult(
            scenario="GDS SSD",
            description="SSD → GPU direct (GDS)",
            total_embeddings=batch_size * 10,
            dram_embeddings=0,
            ssd_embeddings=batch_size * 10,
            batch_size=batch_size,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_size_mb=data_size_bytes / (1024 * 1024),
        )

    def benchmark_dram_only_read(
        self,
        batch_size: int = 16,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> BenchmarkResult:
        """
        Baseline: DRAM only read for comparison.
        
        Path: DRAM → GPU HBM (PCIe)
        """
        print(f"\n{'='*70}")
        print(f"Baseline: DRAM Only Read (DRAM → GPU)")
        print(f"  Batch size: {batch_size}")
        print(f"{'='*70}")
        
        # Create DRAM embeddings
        dram_embeddings = torch.randn(
            self.dram_embeddings, self.embedding_dim,
            dtype=self.dtype, device='cpu', pin_memory=True
        )
        
        # Generate random indices
        indices = torch.randint(
            0, self.dram_embeddings, (batch_size,),
            dtype=torch.int64, device='cpu'
        )
        
        # Warmup
        for _ in range(warmup_iterations):
            result = dram_embeddings[indices].to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            result = dram_embeddings[indices].to(self.device, non_blocking=True)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        avg_latency_ms = np.mean(latencies)
        std_latency_ms = np.std(latencies)
        
        # Calculate bandwidth
        data_size_bytes = batch_size * self.bytes_per_embedding
        bandwidth_gbps = (data_size_bytes / (avg_latency_ms * 1e-3)) / 1e9
        
        print(f"  Latency: {avg_latency_ms:.4f} ± {std_latency_ms:.4f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        
        # Cleanup
        del dram_embeddings
        torch.cuda.empty_cache()
        
        return BenchmarkResult(
            scenario="DRAM Only",
            description="DRAM → GPU (PCIe)",
            total_embeddings=self.dram_embeddings,
            dram_embeddings=self.dram_embeddings,
            ssd_embeddings=0,
            batch_size=batch_size,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_size_mb=data_size_bytes / (1024 * 1024),
        )

    def run_scaling_batch_size(
        self,
        batch_sizes: List[int],
        ssd_base_dir: str,
        num_iterations: int = 30,
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmarks across different batch sizes."""
        results = {
            "Traditional Parallel": [],
            "GDS Parallel": [],
            "Traditional SSD": [],
            "GDS SSD": [],
        }
        
        for bs in batch_sizes:
            print(f"\n{'#'*70}")
            print(f"# Testing with batch size {bs:,}")
            print(f"{'#'*70}")
            
            # Comparison Group 1: Parallel strategies
            trad_result = self.benchmark_traditional_parallel_read(
                batch_size=bs,
                num_iterations=num_iterations,
                ssd_dir=os.path.join(ssd_base_dir, f"trad_{bs}"),
            )
            results["Traditional Parallel"].append(trad_result)
            
            gds_result = self.benchmark_gds_parallel_read(
                batch_size=bs,
                num_iterations=num_iterations,
                ssd_dir=os.path.join(ssd_base_dir, f"gds_{bs}"),
            )
            results["GDS Parallel"].append(gds_result)
            
            # Comparison Group 2: SSD methods
            trad_ssd_result = self.benchmark_traditional_ssd_read(
                batch_size=bs,
                num_iterations=num_iterations,
                ssd_dir=os.path.join(ssd_base_dir, f"trad_ssd_{bs}"),
            )
            results["Traditional SSD"].append(trad_ssd_result)
            
            gds_ssd_result = self.benchmark_gds_ssd_read(
                batch_size=bs,
                num_iterations=num_iterations,
                ssd_dir=os.path.join(ssd_base_dir, f"gds_ssd_{bs}"),
            )
            results["GDS SSD"].append(gds_ssd_result)
        
        return results


def plot_comparison_group1(
    results: Dict[str, List[BenchmarkResult]],
    output_dir: str,
):
    """Plot Comparison Group 1: Traditional vs GDS Parallel Read."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors
    colors = {
        "Traditional Parallel": "coral",
        "GDS Parallel": "seagreen",
    }
    
    for scenario in ["Traditional Parallel", "GDS Parallel"]:
        if scenario not in results or not results[scenario]:
            continue
        
        result_list = results[scenario]
        batch_sizes = [r.batch_size for r in result_list]
        latencies = [r.latency_ms for r in result_list]
        bandwidths = [r.bandwidth_gbps for r in result_list]
        
        color = colors.get(scenario, "gray")
        ax1.plot(batch_sizes, latencies, 'o-', label=scenario,
                linewidth=2, markersize=8, color=color)
        ax2.plot(batch_sizes, bandwidths, 'o-', label=scenario,
                linewidth=2, markersize=8, color=color)
    
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Comparison Group 1: Latency\n(Traditional vs GDS Parallel Read)', fontsize=13)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax2.set_title('Comparison Group 1: Bandwidth\n(Traditional vs GDS Parallel Read)', fontsize=13)
    ax2.set_xscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_group1_parallel_read.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_group2(
    results: Dict[str, List[BenchmarkResult]],
    output_dir: str,
):
    """Plot Comparison Group 2: Traditional SSD vs GDS SSD."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Colors
    colors = {
        "Traditional SSD": "coral",
        "GDS SSD": "seagreen",
    }
    
    for scenario in ["Traditional SSD", "GDS SSD"]:
        if scenario not in results or not results[scenario]:
            continue
        
        result_list = results[scenario]
        batch_sizes = [r.batch_size for r in result_list]
        latencies = [r.latency_ms for r in result_list]
        bandwidths = [r.bandwidth_gbps for r in result_list]
        
        color = colors.get(scenario, "gray")
        ax1.plot(batch_sizes, latencies, 'o-', label=scenario,
                linewidth=2, markersize=8, color=color)
        ax2.plot(batch_sizes, bandwidths, 'o-', label=scenario,
                linewidth=2, markersize=8, color=color)
    
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Comparison Group 2: Latency\n(Traditional SSD vs GDS)', fontsize=13)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax2.set_title('Comparison Group 2: Bandwidth\n(Traditional SSD vs GDS)', fontsize=13)
    ax2.set_xscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_group2_ssd_methods.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_scenarios(
    results: Dict[str, List[BenchmarkResult]],
    output_dir: str,
):
    """Plot all scenarios together for overview."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        "DRAM Only": "steelblue",
        "Traditional Parallel": "coral",
        "GDS Parallel": "seagreen",
        "Traditional SSD": "salmon",
        "GDS SSD": "lightgreen",
    }
    
    for scenario, result_list in results.items():
        if not result_list:
            continue
        
        batch_sizes = [r.batch_size for r in result_list]
        latencies = [r.latency_ms for r in result_list]
        bandwidths = [r.bandwidth_gbps for r in result_list]
        
        color = colors.get(scenario, "gray")
        ax1.plot(batch_sizes, latencies, 'o-', label=scenario,
                linewidth=2, markersize=8, color=color)
        ax2.plot(batch_sizes, bandwidths, 'o-', label=scenario,
                linewidth=2, markersize=8, color=color)
    
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('All Scenarios: Latency Comparison\n(Embedding Dim = 256K, 0.5MB per embedding)', fontsize=13)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax2.set_title('All Scenarios: Bandwidth Comparison\n(Embedding Dim = 256K, 0.5MB per embedding)', fontsize=13)
    ax2.set_xscale('log')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_scenarios_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()


def print_summary_table(results: Dict[str, List[BenchmarkResult]]):
    """Print a summary table of all benchmark results."""
    
    print("\n" + "="*120)
    print("BENCHMARK SUMMARY")
    print("="*120)
    
    print("\n### Comparison Group 1: Parallel Read Strategies ###")
    print(f"{'Scenario':<25} {'Batch Size':<12} {'Latency (ms)':<15} {'Bandwidth (GB/s)':<18} {'Data Size (MB)':<15}")
    print("-"*85)
    
    for scenario in ["Traditional Parallel", "GDS Parallel"]:
        if scenario in results:
            for r in results[scenario]:
                print(f"{r.scenario:<25} {r.batch_size:<12,} {r.latency_ms:<15.4f} {r.bandwidth_gbps:<18.2f} {r.data_size_mb:<15.2f}")
    
    print("\n### Comparison Group 2: SSD Read Methods ###")
    print(f"{'Scenario':<25} {'Batch Size':<12} {'Latency (ms)':<15} {'Bandwidth (GB/s)':<18} {'Data Size (MB)':<15}")
    print("-"*85)
    
    for scenario in ["Traditional SSD", "GDS SSD"]:
        if scenario in results:
            for r in results[scenario]:
                print(f"{r.scenario:<25} {r.batch_size:<12,} {r.latency_ms:<15.4f} {r.bandwidth_gbps:<18.2f} {r.data_size_mb:<15.2f}")
    
    print("="*120)


def main():
    parser = argparse.ArgumentParser(description="Embedding Parallel Read Benchmark")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results and plots",
    )
    parser.add_argument(
        "--total_embeddings",
        type=int,
        default=1000,
        help="Total number of embeddings",
    )
    parser.add_argument(
        "--dram_ratio",
        type=float,
        default=0.5,
        help="Ratio of embeddings stored in DRAM (0.0-1.0)",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f"Dimension of each embedding vector (default: {DEFAULT_EMBEDDING_DIM})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for lookup benchmarks",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=50,
        help="Number of iterations for each benchmark",
    )
    parser.add_argument(
        "--scaling_test",
        action="store_true",
        help="Run scaling tests with different batch sizes",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Data type for embeddings (default: bf16)",
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    
    print("="*80)
    print("EMBEDDING PARALLEL READ BENCHMARK")
    print("="*80)
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"Data Type: {args.dtype}")
    print(f"Embedding Dim: {args.embedding_dim:,} ({args.embedding_dim / 1024:.0f}K)")
    print(f"Size per embedding: {args.embedding_dim * (2 if dtype == torch.bfloat16 else 4) / (1024*1024):.3f} MB")
    print("="*80)
    
    # Initialize benchmark
    benchmark = EmbeddingParallelReadBenchmark(
        total_embeddings=args.total_embeddings,
        dram_ratio=args.dram_ratio,
        embedding_dim=args.embedding_dim,
        dtype=dtype,
    )
    
    ssd_cache_dir = os.path.join(args.output_dir, "ssd_cache")
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results: Dict[str, List[BenchmarkResult]] = {}
    
    # Run main benchmarks
    print("\n" + "#"*80)
    print("# MAIN BENCHMARKS")
    print("#"*80)
    
    # Comparison Group 1: Parallel strategies
    print("\n" + "="*80)
    print("COMPARISON GROUP 1: Parallel Read Strategies")
    print("="*80)
    
    trad_result = benchmark.benchmark_traditional_parallel_read(
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        ssd_dir=os.path.join(ssd_cache_dir, "traditional"),
    )
    all_results["Traditional Parallel"] = [trad_result]
    
    gds_result = benchmark.benchmark_gds_parallel_read(
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        ssd_dir=os.path.join(ssd_cache_dir, "gds"),
    )
    all_results["GDS Parallel"] = [gds_result]
    
    # Comparison Group 2: SSD methods
    print("\n" + "="*80)
    print("COMPARISON GROUP 2: SSD Read Methods")
    print("="*80)
    
    trad_ssd_result = benchmark.benchmark_traditional_ssd_read(
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        ssd_dir=os.path.join(ssd_cache_dir, "trad_ssd"),
    )
    all_results["Traditional SSD"] = [trad_ssd_result]
    
    gds_ssd_result = benchmark.benchmark_gds_ssd_read(
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        ssd_dir=os.path.join(ssd_cache_dir, "gds_ssd"),
    )
    all_results["GDS SSD"] = [gds_ssd_result]
    
    # DRAM baseline
    dram_result = benchmark.benchmark_dram_only_read(
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
    )
    all_results["DRAM Only"] = [dram_result]
    
    # Scaling tests
    if args.scaling_test:
        print("\n" + "#"*80)
        print("# SCALING TESTS")
        print("#"*80)
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        scaling_results = benchmark.run_scaling_batch_size(
            batch_sizes=batch_sizes,
            ssd_base_dir=os.path.join(ssd_cache_dir, "scaling"),
            num_iterations=30,
        )
        
        # Merge results
        for key in scaling_results:
            if key not in all_results:
                all_results[key] = []
            all_results[key].extend(scaling_results[key])
    
    # Generate plots
    plot_comparison_group1(all_results, args.output_dir)
    plot_comparison_group2(all_results, args.output_dir)
    plot_all_scenarios(all_results, args.output_dir)
    
    # Print summary
    print_summary_table(all_results)
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - comparison_group1_parallel_read.png")
    print(f"  - comparison_group2_ssd_methods.png")
    print(f"  - all_scenarios_overview.png")


if __name__ == "__main__":
    main()