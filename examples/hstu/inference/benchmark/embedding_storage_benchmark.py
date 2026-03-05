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
Embedding Storage Latency Benchmark

This script benchmarks the latency of reading embeddings from different storage tiers:
1. GPU HBM (High Bandwidth Memory) - baseline
2. Host DRAM (via UVM - Unified Virtual Memory)
3. SSD (via file system load/dump)

Default configuration:
- Embedding dim: 262144 (256K) -> 0.5MB per embedding in bf16 (2 bytes per element)
- This allows testing realistic large embedding scenarios

NOTE: DynamicEmb has a kernel limitation of embedding dim <= 1024.
For large embedding dimensions (like 256K), we use PyTorch native tensors for storage testing.

Usage:
    python embedding_storage_benchmark.py [--output_dir OUTPUT_DIR] [--num_embeddings NUM_EMBEDDINGS]
"""

import argparse
import os
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Default embedding dimension: 256K = 262144
# In bf16 (2 bytes per element), this equals 0.5MB per embedding
DEFAULT_EMBEDDING_DIM = 262144  # 256K


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single test configuration."""
    storage_type: str
    num_embeddings: int
    embedding_dim: int
    batch_size: int
    latency_ms: float
    bandwidth_gbps: float
    data_size_mb: float


class EmbeddingStorageBenchmark:
    """Benchmark class for measuring embedding lookup latency from different storage tiers."""

    def __init__(
        self,
        num_embeddings: int = 1000,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device or torch.cuda.current_device()
        self.dtype = dtype
        
        # Calculate memory requirements
        self.bytes_per_element = 2 if dtype == torch.bfloat16 else 4  # bf16=2, fp32=4
        self.bytes_per_embedding = embedding_dim * self.bytes_per_element
        self.total_embedding_bytes = num_embeddings * self.bytes_per_embedding
        
        print(f"Embedding config: dim={embedding_dim}, dtype={dtype}")
        print(f"  Size per embedding: {self.bytes_per_embedding / (1024*1024):.3f} MB")
        print(f"  Total table size: {self.total_embedding_bytes / (1024*1024):.2f} MB")

    def benchmark_hbm_lookup(
        self,
        batch_size: int = 16,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> BenchmarkResult:
        """
        Benchmark embedding lookup from GPU HBM.
        
        All embeddings are stored in GPU HBM for fastest access.
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking GPU HBM → GPU Lookup")
        print(f"  Num embeddings: {self.num_embeddings:,}")
        print(f"  Embedding dim: {self.embedding_dim:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Data per batch: {batch_size * self.bytes_per_embedding / (1024*1024):.2f} MB")
        print(f"{'='*60}")
        
        # Create embedding table in GPU HBM
        embeddings = torch.randn(
            self.num_embeddings, self.embedding_dim,
            dtype=self.dtype, device=self.device
        )
        
        # Generate random lookup indices
        indices = torch.randint(
            0, self.num_embeddings, (batch_size,), dtype=torch.int64, device=self.device
        )
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = embeddings[indices]
        torch.cuda.synchronize()
        
        # Benchmark
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
        
        for i in range(num_iterations):
            start_events[i].record()
            _ = embeddings[indices]
            end_events[i].record()
        
        torch.cuda.synchronize()
        
        # Calculate latency
        latencies = [
            start_events[i].elapsed_time(end_events[i])
            for i in range(num_iterations)
        ]
        avg_latency_ms = np.mean(latencies)
        std_latency_ms = np.std(latencies)
        
        # Calculate bandwidth
        data_size_bytes = batch_size * self.bytes_per_embedding
        bandwidth_gbps = (data_size_bytes / (avg_latency_ms * 1e-3)) / 1e9
        
        print(f"  Latency: {avg_latency_ms:.4f} ± {std_latency_ms:.4f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        
        # Cleanup
        del embeddings
        torch.cuda.empty_cache()
        
        return BenchmarkResult(
            storage_type="GPU HBM",
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=batch_size,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_size_mb=data_size_bytes / (1024 * 1024),
        )

    def benchmark_dram_lookup(
        self,
        batch_size: int = 16,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> BenchmarkResult:
        """
        Benchmark embedding lookup from Host DRAM.
        
        All embeddings are stored in Host DRAM (CPU memory).
        This tests the PCIe transfer latency.
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking Host DRAM → GPU Lookup")
        print(f"  Num embeddings: {self.num_embeddings:,}")
        print(f"  Embedding dim: {self.embedding_dim:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Data per batch: {batch_size * self.bytes_per_embedding / (1024*1024):.2f} MB")
        print(f"{'='*60}")
        
        # Create embedding table in Host DRAM (CPU memory)
        embeddings_cpu = torch.randn(
            self.num_embeddings, self.embedding_dim,
            dtype=self.dtype, device='cpu', pin_memory=True
        )
        
        # Generate random lookup indices (on CPU, will transfer to GPU)
        indices_cpu = torch.randint(
            0, self.num_embeddings, (batch_size,), dtype=torch.int64, device='cpu'
        )
        
        # Warmup
        for _ in range(warmup_iterations):
            indices_gpu = indices_cpu.to(self.device, non_blocking=True)
            result = embeddings_cpu[indices_gpu.cpu()].to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # Lookup: CPU -> GPU
            indices_gpu = indices_cpu.to(self.device, non_blocking=True)
            result = embeddings_cpu[indices_gpu.cpu()].to(self.device, non_blocking=True)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        avg_latency_ms = np.mean(latencies)
        std_latency_ms = np.std(latencies)
        
        # Calculate bandwidth
        data_size_bytes = batch_size * self.bytes_per_embedding
        bandwidth_gbps = (data_size_bytes / (avg_latency_ms * 1e-3)) / 1e9
        
        print(f"  Latency: {avg_latency_ms:.4f} ± {std_latency_ms:.4f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        
        # Cleanup
        del embeddings_cpu
        torch.cuda.empty_cache()
        
        return BenchmarkResult(
            storage_type="Host DRAM",
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=batch_size,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_size_mb=data_size_bytes / (1024 * 1024),
        )

    def benchmark_uvm_lookup(
        self,
        batch_size: int = 16,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> BenchmarkResult:
        """
        Benchmark embedding lookup via UVM (Unified Virtual Memory).
        
        UVM allows oversubscribing GPU memory by using host memory as backing storage.
        The GPU can directly access memory allocated on the CPU.
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking UVM (Unified Virtual Memory) → GPU Lookup")
        print(f"  Num embeddings: {self.num_embeddings:,}")
        print(f"  Embedding dim: {self.embedding_dim:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Data per batch: {batch_size * self.bytes_per_embedding / (1024*1024):.2f} MB")
        print(f"{'='*60}")
        
        # Create embedding table using UVM
        # In PyTorch, we can simulate UVM by allocating on CPU and accessing from GPU
        # For true UVM, we would need to use CUDA managed memory directly
        embeddings_managed = torch.randn(
            self.num_embeddings, self.embedding_dim,
            dtype=self.dtype, device='cpu', pin_memory=True
        )
        
        # Generate random lookup indices
        indices = torch.randint(
            0, self.num_embeddings, (batch_size,), dtype=torch.int64, device=self.device
        )
        
        # Warmup - establish page mappings
        for _ in range(warmup_iterations):
            idx_cpu = indices.cpu()
            result = embeddings_managed[idx_cpu].to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            # UVM-style lookup: access from GPU triggers page migration
            idx_cpu = indices.cpu()
            result = embeddings_managed[idx_cpu].to(self.device, non_blocking=True)
            
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
        del embeddings_managed
        torch.cuda.empty_cache()
        
        return BenchmarkResult(
            storage_type="Host DRAM (UVM)",
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=batch_size,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_size_mb=data_size_bytes / (1024 * 1024),
        )

    def benchmark_ssd_load(
        self,
        save_dir: str,
        num_iterations: int = 5,
    ) -> BenchmarkResult:
        """
        Benchmark loading embeddings from SSD to GPU.
        
        This tests the full path: SSD → Host DRAM → GPU HBM
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking SSD → GPU Load")
        print(f"  Num embeddings: {self.num_embeddings:,}")
        print(f"  Embedding dim: {self.embedding_dim:,}")
        print(f"  Total data size: {self.total_embedding_bytes / (1024*1024):.2f} MB")
        print(f"  Save directory: {save_dir}")
        print(f"{'='*60}")
        
        # Create directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Step 1: Create embeddings and save to SSD
        embeddings = torch.randn(
            self.num_embeddings, self.embedding_dim,
            dtype=self.dtype, device='cpu'
        )
        
        # Save to SSD
        print("  Saving embeddings to SSD...")
        save_path = os.path.join(save_dir, "embeddings.pt")
        save_start = time.time()
        torch.save(embeddings, save_path)
        save_time = time.time() - save_start
        print(f"  Save time: {save_time:.2f} s")
        
        # Get file size
        file_size = os.path.getsize(save_path)
        print(f"  File size: {file_size / (1024*1024):.2f} MB")
        
        # Clear memory
        del embeddings
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Step 2: Benchmark load from SSD
        latencies = []
        
        for i in range(num_iterations):
            torch.cuda.synchronize()
            load_start = time.time()
            
            # Load from SSD to CPU, then to GPU
            embeddings_loaded = torch.load(save_path)
            embeddings_gpu = embeddings_loaded.to(self.device, non_blocking=True)
            
            torch.cuda.synchronize()
            load_time = time.time() - load_start
            latencies.append(load_time * 1000)  # Convert to ms
            
            print(f"  Iteration {i+1}/{num_iterations}: {load_time:.3f} s")
            
            # Cleanup for next iteration
            del embeddings_loaded, embeddings_gpu
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        avg_latency_ms = np.mean(latencies)
        std_latency_ms = np.std(latencies)
        
        # Calculate bandwidth
        data_size_bytes = self.total_embedding_bytes
        bandwidth_gbps = (data_size_bytes / (avg_latency_ms * 1e-3)) / 1e9
        
        print(f"  Load latency: {avg_latency_ms:.2f} ± {std_latency_ms:.2f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.2f} GB/s")
        
        # Cleanup
        shutil.rmtree(save_dir, ignore_errors=True)
        
        return BenchmarkResult(
            storage_type="SSD",
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.num_embeddings,  # Loading all embeddings
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_size_mb=data_size_bytes / (1024 * 1024),
        )

    def benchmark_ssd_partial_load(
        self,
        save_dir: str,
        batch_size: int = 16,
        num_iterations: int = 10,
    ) -> BenchmarkResult:
        """
        Benchmark loading a batch of embeddings from SSD to GPU.
        
        This simulates the scenario where we need to load specific embeddings from SSD.
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking SSD Partial Load (batch={batch_size})")
        print(f"  Num embeddings: {self.num_embeddings:,}")
        print(f"  Embedding dim: {self.embedding_dim:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Data per batch: {batch_size * self.bytes_per_embedding / (1024*1024):.2f} MB")
        print(f"{'='*60}")
        
        # Create directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Step 1: Create embeddings and save to SSD
        embeddings = torch.randn(
            self.num_embeddings, self.embedding_dim,
            dtype=self.dtype, device='cpu'
        )
        save_path = os.path.join(save_dir, "embeddings.pt")
        print("  Saving embeddings to SSD...")
        torch.save(embeddings, save_path)
        
        # Clear memory
        del embeddings
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Step 2: Benchmark partial load
        latencies = []
        
        for i in range(num_iterations):
            # Generate random lookup indices
            indices = torch.randint(
                0, self.num_embeddings, (batch_size,), dtype=torch.int64, device='cpu'
            )
            
            torch.cuda.synchronize()
            load_start = time.time()
            
            # Load entire table from SSD, then lookup
            embeddings_loaded = torch.load(save_path)
            result = embeddings_loaded[indices].to(self.device, non_blocking=True)
            
            torch.cuda.synchronize()
            load_time = time.time() - load_start
            latencies.append(load_time * 1000)
            
            # Cleanup for next iteration
            del embeddings_loaded, result
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        avg_latency_ms = np.mean(latencies)
        std_latency_ms = np.std(latencies)
        
        # Calculate bandwidth (based on batch size, not total table size)
        data_size_bytes = batch_size * self.bytes_per_embedding
        bandwidth_gbps = (data_size_bytes / (avg_latency_ms * 1e-3)) / 1e9
        
        print(f"  Load+Lookup latency: {avg_latency_ms:.2f} ± {std_latency_ms:.2f} ms")
        print(f"  Effective bandwidth (batch data): {bandwidth_gbps:.2f} GB/s")
        
        # Cleanup
        shutil.rmtree(save_dir, ignore_errors=True)
        
        return BenchmarkResult(
            storage_type="SSD (Partial)",
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=batch_size,
            latency_ms=avg_latency_ms,
            bandwidth_gbps=bandwidth_gbps,
            data_size_mb=data_size_bytes / (1024 * 1024),
        )

    def benchmark_scaling_num_embeddings(
        self,
        embedding_counts: List[int],
        batch_size: int = 16,
        ssd_dir: str = None,
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Benchmark latency scaling with different number of embeddings.
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking Scaling with Number of Embeddings")
        print(f"{'='*60}")
        
        results = {"GPU HBM": [], "Host DRAM": [], "SSD": []}
        original_num = self.num_embeddings
        
        for num_emb in embedding_counts:
            print(f"\n--- Testing with {num_emb:,} embeddings ---")
            self.num_embeddings = num_emb
            self.total_embedding_bytes = num_emb * self.bytes_per_embedding
            
            # HBM benchmark
            try:
                hbm_result = self.benchmark_hbm_lookup(batch_size=batch_size)
                results["GPU HBM"].append(hbm_result)
            except Exception as e:
                print(f"  HBM benchmark failed: {e}")
            
            # DRAM benchmark
            try:
                dram_result = self.benchmark_dram_lookup(batch_size=batch_size)
                results["Host DRAM"].append(dram_result)
            except Exception as e:
                print(f"  DRAM benchmark failed: {e}")
            
            # SSD benchmark
            if ssd_dir:
                try:
                    ssd_result = self.benchmark_ssd_partial_load(
                        save_dir=os.path.join(ssd_dir, f"emb_{num_emb}"),
                        batch_size=batch_size,
                    )
                    results["SSD"].append(ssd_result)
                except Exception as e:
                    print(f"  SSD benchmark failed: {e}")
        
        self.num_embeddings = original_num
        self.total_embedding_bytes = original_num * self.bytes_per_embedding
        
        return results

    def benchmark_scaling_batch_size(
        self,
        batch_sizes: List[int],
        ssd_dir: str = None,
    ) -> Dict[str, List[BenchmarkResult]]:
        """
        Benchmark latency scaling with different batch sizes.
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking Scaling with Batch Size")
        print(f"{'='*60}")
        
        results = {"GPU HBM": [], "Host DRAM": [], "SSD": []}
        
        for bs in batch_sizes:
            print(f"\n--- Testing with batch size {bs:,} ---")
            
            # HBM benchmark
            try:
                hbm_result = self.benchmark_hbm_lookup(batch_size=bs)
                results["GPU HBM"].append(hbm_result)
            except Exception as e:
                print(f"  HBM benchmark failed: {e}")
            
            # DRAM benchmark
            try:
                dram_result = self.benchmark_dram_lookup(batch_size=bs)
                results["Host DRAM"].append(dram_result)
            except Exception as e:
                print(f"  DRAM benchmark failed: {e}")
            
            # SSD benchmark
            if ssd_dir:
                try:
                    ssd_result = self.benchmark_ssd_partial_load(
                        save_dir=os.path.join(ssd_dir, f"batch_{bs}"),
                        batch_size=bs,
                    )
                    results["SSD"].append(ssd_result)
                except Exception as e:
                    print(f"  SSD benchmark failed: {e}")
        
        return results


def plot_results(
    results: Dict[str, List[BenchmarkResult]],
    output_dir: str,
    plot_type: str = "latency",
):
    """Generate plots from benchmark results."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Color scheme for storage types
    colors = {
        "GPU HBM": "steelblue",
        "Host DRAM": "coral",
        "Host DRAM (UVM)": "coral",
        "SSD": "seagreen",
        "SSD (Partial)": "seagreen",
    }
    
    if plot_type == "latency":
        # Bar chart comparing latencies
        fig, ax = plt.subplots(figsize=(12, 7))
        
        storage_types = list(results.keys())
        latencies = [results[st][0].latency_ms if results[st] else 0 for st in storage_types]
        bandwidths = [results[st][0].bandwidth_gbps if results[st] else 0 for st in storage_types]
        
        x = np.arange(len(storage_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, latencies, width, label='Latency (ms)', color='steelblue', alpha=0.8)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, bandwidths, width, label='Bandwidth (GB/s)', color='coral', alpha=0.8)
        
        ax.set_xlabel('Storage Type', fontsize=12)
        ax.set_ylabel('Latency (ms)', color='steelblue', fontsize=12)
        ax2.set_ylabel('Bandwidth (GB/s)', color='coral', fontsize=12)
        ax.set_title('Embedding Lookup: Latency and Bandwidth by Storage Type\n(Embedding Dim = 256K, 0.5MB per embedding in bf16)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(storage_types, rotation=15, ha='right', fontsize=11)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        fig.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'storage_latency_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
    elif plot_type == "scaling_embeddings":
        # Line plot showing latency vs number of embeddings
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for storage_type, result_list in results.items():
            if not result_list:
                continue
            num_embs = [r.num_embeddings for r in result_list]
            latencies = [r.latency_ms for r in result_list]
            bandwidths = [r.bandwidth_gbps for r in result_list]
            
            color = colors.get(storage_type, 'gray')
            ax1.plot(num_embs, latencies, 'o-', label=storage_type, 
                    linewidth=2, markersize=8, color=color)
            ax2.plot(num_embs, bandwidths, 'o-', label=storage_type, 
                    linewidth=2, markersize=8, color=color)
        
        ax1.set_xlabel('Number of Embeddings', fontsize=12)
        ax1.set_ylabel('Lookup Latency (ms)', fontsize=12)
        ax1.set_title('Latency vs Table Size\n(Embedding Dim = 256K, 0.5MB per embedding)', fontsize=13)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Number of Embeddings', fontsize=12)
        ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax2.set_title('Bandwidth vs Table Size\n(Embedding Dim = 256K, 0.5MB per embedding)', fontsize=13)
        ax2.set_xscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scaling_embeddings.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
    elif plot_type == "scaling_batch":
        # Line plot showing latency vs batch size
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for storage_type, result_list in results.items():
            if not result_list:
                continue
            batch_sizes = [r.batch_size for r in result_list]
            latencies = [r.latency_ms for r in result_list]
            bandwidths = [r.bandwidth_gbps for r in result_list]
            
            color = colors.get(storage_type, 'gray')
            ax1.plot(batch_sizes, latencies, 'o-', label=storage_type, 
                    linewidth=2, markersize=8, color=color)
            ax2.plot(batch_sizes, bandwidths, 'o-', label=storage_type, 
                    linewidth=2, markersize=8, color=color)
        
        ax1.set_xlabel('Batch Size', fontsize=12)
        ax1.set_ylabel('Lookup Latency (ms)', fontsize=12)
        ax1.set_title('Latency vs Batch Size\n(Embedding Dim = 256K, 0.5MB per embedding)', fontsize=13)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Batch Size', fontsize=12)
        ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
        ax2.set_title('Bandwidth vs Batch Size\n(Embedding Dim = 256K, 0.5MB per embedding)', fontsize=13)
        ax2.set_xscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scaling_batch_size.png'), dpi=150, bbox_inches='tight')
        plt.close()


def print_summary_table(results: Dict[str, List[BenchmarkResult]]):
    """Print a summary table of all benchmark results."""
    
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    
    # Header
    print(f"{'Storage Type':<22} {'# Embeddings':<14} {'Batch Size':<12} {'Latency (ms)':<15} {'Bandwidth (GB/s)':<18} {'Data Size (MB)':<15}")
    print("-"*100)
    
    for storage_type, result_list in results.items():
        for r in result_list:
            print(f"{r.storage_type:<22} {r.num_embeddings:<14,} {r.batch_size:<12,} {r.latency_ms:<15.4f} {r.bandwidth_gbps:<18.2f} {r.data_size_mb:<15.2f}")
    
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description="Embedding Storage Latency Benchmark")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results and plots",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=50000,
        help="Number of embeddings in the table",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f"Dimension of each embedding vector (default: {DEFAULT_EMBEDDING_DIM} = 256K, 0.5MB in bf16)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for lookup benchmarks",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of iterations for each benchmark",
    )
    parser.add_argument(
        "--skip_ssd",
        action="store_true",
        help="Skip SSD benchmark (can be slow)",
    )
    parser.add_argument(
        "--scaling_test",
        action="store_true",
        help="Run scaling tests with different sizes",
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
    print("EMBEDDING STORAGE LATENCY BENCHMARK")
    print("="*80)
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"Data Type: {args.dtype}")
    print(f"Embedding Dim: {args.embedding_dim:,} ({args.embedding_dim / 1024:.0f}K)")
    print(f"Size per embedding: {args.embedding_dim * (2 if dtype == torch.bfloat16 else 4) / (1024*1024):.3f} MB")
    print("="*80)
    
    # Initialize benchmark
    benchmark = EmbeddingStorageBenchmark(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        dtype=dtype,
    )
    
    all_results: Dict[str, List[BenchmarkResult]] = {}
    ssd_cache_dir = os.path.join(args.output_dir, "ssd_cache")
    
    # Run main benchmarks
    print("\n" + "#"*80)
    print("# MAIN BENCHMARKS")
    print("#"*80)
    
    # GPU HBM benchmark
    hbm_result = benchmark.benchmark_hbm_lookup(
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
    )
    all_results["GPU HBM"] = [hbm_result]
    
    # Host DRAM benchmark
    dram_result = benchmark.benchmark_dram_lookup(
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
    )
    all_results["Host DRAM"] = [dram_result]
    
    # SSD benchmark
    if not args.skip_ssd:
        ssd_result = benchmark.benchmark_ssd_partial_load(
            save_dir=os.path.join(ssd_cache_dir, "main"),
            batch_size=args.batch_size,
            num_iterations=5,
        )
        all_results["SSD"] = [ssd_result]
    
    # Scaling tests
    if args.scaling_test:
        print("\n" + "#"*80)
        print("# SCALING TESTS")
        print("#"*80)
        
        # Test scaling with number of embeddings
        # For 256K dim embeddings, we use smaller counts due to memory constraints
        embedding_counts = [100, 500, 1000, 2000, 5000]
        scaling_results = benchmark.benchmark_scaling_num_embeddings(
            embedding_counts=embedding_counts,
            batch_size=args.batch_size,
            ssd_dir=None if args.skip_ssd else ssd_cache_dir,
        )
        plot_results(scaling_results, args.output_dir, "scaling_embeddings")
        
        # Merge results
        for key in scaling_results:
            if key not in all_results:
                all_results[key] = []
            all_results[key].extend(scaling_results[key])
        
        # Test scaling with batch size
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        batch_results = benchmark.benchmark_scaling_batch_size(
            batch_sizes=batch_sizes,
            ssd_dir=None if args.skip_ssd else ssd_cache_dir,
        )
        plot_results(batch_results, args.output_dir, "scaling_batch")
        
        # Merge results
        for key in batch_results:
            if key not in all_results:
                all_results[key] = []
            all_results[key].extend(batch_results[key])
    
    # Generate plots and summary
    plot_results(all_results, args.output_dir, "latency")
    print_summary_table(all_results)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()