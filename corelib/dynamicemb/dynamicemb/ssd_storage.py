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
SSD Storage for Dynamic Embedding Offload.

This module provides a SSD-based storage backend for offloading embeddings
from DRAM to SSD, enabling larger embedding tables with limited memory.
"""

import os
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class SSDStorage:
    """
    SSD-based storage for embedding offload.
    
    This class provides a simple key-value storage on SSD using memory-mapped files.
    It supports efficient batch insert, lookup, and evict operations.
    
    Attributes:
        storage_path (str): Directory path for SSD storage files.
        dim (int): Dimension of embedding vectors.
        dtype (torch.dtype): Data type of embeddings.
        key_dtype (torch.dtype): Data type of keys (default: torch.int64).
    """
    
    def __init__(
        self,
        storage_path: str,
        dim: int,
        dtype: torch.dtype = torch.float32,
        key_dtype: torch.dtype = torch.int64,
        initial_capacity: int = 100000,
    ):
        """
        Initialize SSD storage.
        
        Args:
            storage_path: Directory path for SSD storage files.
            dim: Dimension of embedding vectors.
            dtype: Data type of embeddings.
            key_dtype: Data type of keys.
            initial_capacity: Initial capacity for the index map.
        """
        self._storage_path = storage_path
        self._dim = dim
        self._dtype = dtype
        self._key_dtype = key_dtype
        
        # Create storage directory if not exists
        os.makedirs(storage_path, exist_ok=True)
        
        # File paths
        self._data_file_path = os.path.join(storage_path, "embeddings.bin")
        self._index_file_path = os.path.join(storage_path, "index.bin")
        
        # In-memory index: key -> (offset, length)
        # offset is the byte offset in the data file
        self._index: Dict[int, int] = {}  # key -> offset (in terms of embedding rows)
        self._reverse_index: Dict[int, int] = {}  # offset -> key
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Current write offset (in terms of embedding rows)
        self._current_offset = 0
        
        # Free slots from evicted embeddings (for reuse)
        self._free_slots: list = []
        
        # Load existing index if available
        self._load_index()
        
        # Open data file for appending
        self._data_file = None
        self._open_data_file()
    
    def _open_data_file(self):
        """Open the data file for read/write operations."""
        mode = "r+b" if os.path.exists(self._data_file_path) else "wb"
        self._data_file = open(self._data_file_path, mode + "+")
    
    def _load_index(self):
        """Load index from disk if available."""
        if os.path.exists(self._index_file_path):
            try:
                index_data = np.load(self._index_file_path, allow_pickle=True)
                self._index = dict(index_data.item().get("index", {}))
                self._reverse_index = dict(index_data.item().get("reverse_index", {}))
                self._current_offset = index_data.item().get("current_offset", 0)
                self._free_slots = list(index_data.item().get("free_slots", []))
            except Exception as e:
                print(f"Warning: Failed to load SSD index: {e}")
                self._index = {}
                self._reverse_index = {}
                self._current_offset = 0
                self._free_slots = []
    
    def _save_index(self):
        """Save index to disk."""
        index_data = {
            "index": self._index,
            "reverse_index": self._reverse_index,
            "current_offset": self._current_offset,
            "free_slots": self._free_slots,
        }
        np.save(self._index_file_path, index_data)
    
    def _get_embedding_size(self) -> int:
        """Get the size of a single embedding in bytes."""
        return self._dim * torch.tensor([], dtype=self._dtype).element_size()
    
    def insert(
        self,
        keys: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> int:
        """
        Insert embeddings into SSD storage.
        
        Args:
            keys: Tensor of keys (shape: [N]).
            embeddings: Tensor of embeddings (shape: [N, dim]).
            
        Returns:
            Number of embeddings inserted.
        """
        if keys.numel() == 0:
            return 0
        
        assert keys.shape[0] == embeddings.shape[0], \
            f"Keys and embeddings count mismatch: {keys.shape[0]} vs {embeddings.shape[0]}"
        assert embeddings.shape[1] == self._dim, \
            f"Embedding dimension mismatch: {embeddings.shape[1]} vs {self._dim}"
        
        with self._lock:
            # Convert to numpy for efficient file operations
            keys_np = keys.cpu().numpy()
            embeddings_np = embeddings.cpu().to(self._dtype).numpy()
            
            # Seek to end of file
            self._data_file.seek(0, 2)  # Seek to end
            
            inserted_count = 0
            for i, key in enumerate(keys_np):
                key_int = int(key)
                
                # Check if key already exists
                if key_int in self._index:
                    # Overwrite existing embedding
                    offset = self._index[key_int]
                    self._write_embedding(offset, embeddings_np[i])
                else:
                    # Use free slot if available, otherwise append
                    if self._free_slots:
                        offset = self._free_slots.pop(0)
                    else:
                        offset = self._current_offset
                        self._current_offset += 1
                    
                    # Write embedding
                    self._write_embedding(offset, embeddings_np[i])
                    
                    # Update index
                    self._index[key_int] = offset
                    self._reverse_index[offset] = key_int
                
                inserted_count += 1
            
            # Flush and save index
            self._data_file.flush()
            self._save_index()
            
            return inserted_count
    
    def _write_embedding(self, offset: int, embedding: np.ndarray):
        """Write a single embedding at the given offset."""
        byte_offset = offset * self._get_embedding_size()
        self._data_file.seek(byte_offset)
        self._data_file.write(embedding.tobytes())
    
    def _read_embedding(self, offset: int) -> np.ndarray:
        """Read a single embedding from the given offset."""
        byte_offset = offset * self._get_embedding_size()
        self._data_file.seek(byte_offset)
        data = self._data_file.read(self._get_embedding_size())
        return np.frombuffer(data, dtype=np.dtype(str(self._dtype))).reshape(self._dim)
    
    def lookup(
        self,
        keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Look up embeddings by keys.
        
        Args:
            keys: Tensor of keys to look up (shape: [N]).
            
        Returns:
            Tuple of (embeddings, found_mask):
                - embeddings: Tensor of found embeddings (shape: [N, dim]).
                - found_mask: Boolean tensor indicating which keys were found.
        """
        if keys.numel() == 0:
            return torch.empty((0, self._dim), dtype=self._dtype), torch.empty((0,), dtype=torch.bool)
        
        with self._lock:
            keys_np = keys.cpu().numpy()
            embeddings = np.zeros((len(keys_np), self._dim), dtype=np.dtype(str(self._dtype)))
            found_mask = np.zeros(len(keys_np), dtype=bool)
            
            for i, key in enumerate(keys_np):
                key_int = int(key)
                if key_int in self._index:
                    offset = self._index[key_int]
                    embeddings[i] = self._read_embedding(offset)
                    found_mask[i] = True
            
            return (
                torch.from_numpy(embeddings).to(self._dtype),
                torch.from_numpy(found_mask),
            )
    
    def evict(
        self,
        keys: torch.Tensor,
    ) -> int:
        """
        Evict embeddings by keys.
        
        Args:
            keys: Tensor of keys to evict (shape: [N]).
            
        Returns:
            Number of embeddings evicted.
        """
        if keys.numel() == 0:
            return 0
        
        with self._lock:
            keys_np = keys.cpu().numpy()
            evicted_count = 0
            
            for key in keys_np:
                key_int = int(key)
                if key_int in self._index:
                    offset = self._index.pop(key_int)
                    self._reverse_index.pop(offset, None)
                    self._free_slots.append(offset)
                    evicted_count += 1
            
            if evicted_count > 0:
                self._save_index()
            
            return evicted_count
    
    def contains(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Check which keys exist in storage.
        
        Args:
            keys: Tensor of keys to check (shape: [N]).
            
        Returns:
            Boolean tensor indicating which keys exist.
        """
        with self._lock:
            keys_np = keys.cpu().numpy()
            found = np.array([int(key) in self._index for key in keys_np], dtype=bool)
            return torch.from_numpy(found)
    
    def size(self) -> int:
        """Return the number of embeddings in storage."""
        with self._lock:
            return len(self._index)
    
    def get_all_keys(self) -> torch.Tensor:
        """Return all keys in storage."""
        with self._lock:
            return torch.tensor(list(self._index.keys()), dtype=self._key_dtype)
    
    def clear(self):
        """Clear all embeddings from storage."""
        with self._lock:
            self._index.clear()
            self._reverse_index.clear()
            self._free_slots.clear()
            self._current_offset = 0
            
            # Close and delete data file
            if self._data_file:
                self._data_file.close()
            
            if os.path.exists(self._data_file_path):
                os.remove(self._data_file_path)
            
            if os.path.exists(self._index_file_path):
                os.remove(self._index_file_path)
            
            # Reopen data file
            self._open_data_file()
    
    def close(self):
        """Close the storage and release resources."""
        with self._lock:
            if self._data_file:
                self._data_file.close()
                self._data_file = None
            self._save_index()
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()
    
    def __len__(self) -> int:
        return self.size()
    
    def __contains__(self, key: int) -> bool:
        with self._lock:
            return key in self._index