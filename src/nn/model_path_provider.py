import os
import shutil
import tempfile
import atexit
from typing import List, Optional

class IsolatedModelPathProvider:
    """
    Manages temporary copies of model files to prevent file access
    conflicts between concurrent processes.

    Provides paths to these temporary copies for external loading.
    Ensures temporary directories are cleaned up on script exit.
    """
    _temp_dirs: List[str] = []  # Class attribute to track all temp dirs
    _cleanup_registered: bool = False # Ensure atexit is registered only once

    def __init__(self):
        # Register cleanup only once
        if not IsolatedModelPathProvider._cleanup_registered:
            atexit.register(self._cleanup_all)
            IsolatedModelPathProvider._cleanup_registered = True

    def get_path(self, original_path: str) -> str:
        """
        Copies the model file to a unique temporary location and returns its path.

        Args:
            original_path: The path to the original model file (.keras, etc.).

        Returns:
            The path to the isolated, temporary copy of the model file.

        Raises:
            FileNotFoundError: If the original_path does not exist.
            Exception: If copying fails.
        """
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Original model file not found: {original_path}")

        instance_temp_dir: Optional[str] = None
        try:
            # Create a unique temporary directory for this model copy
            # Using original filename in prefix for slightly better identification
            safe_basename = "".join(c if c.isalnum() else "_" for c in os.path.basename(original_path))
            instance_temp_dir = tempfile.mkdtemp(prefix=f"isolated_{safe_basename}_{os.getpid()}_")
            self._temp_dirs.append(instance_temp_dir) # Track for cleanup

            # Construct the path for the copied file
            copied_model_path = os.path.join(instance_temp_dir, os.path.basename(original_path))

            # Copy the original model file
            #logging.info(f"Copying '{os.path.basename(original_path)}' to temporary location: {instance_temp_dir}")
            shutil.copy2(original_path, copied_model_path) # copy2 preserves metadata
            #logging.info(f"Successfully created temporary copy: {copied_model_path}")

            # Return the path to the copy
            return copied_model_path

        except Exception as e:
            raise # Re-raise the exception

    @classmethod
    def _cleanup_all(cls):
        """Static method (called by atexit) to clean up all temporary directories."""
        if not cls._temp_dirs:
            return

        cleaned_count = 0
        failed_count = 0
        dirs_to_remove = list(cls._temp_dirs) # Iterate over a copy
        cls._temp_dirs.clear() # Clear original list

        for temp_dir in dirs_to_remove:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    cleaned_count += 1
                except Exception as e:
                    failed_count += 1

