"""
Audiobook Creator
Copyright (C) 2025 Prakhar Sharma

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import subprocess
import shutil
import os

def get_system_python_paths():
    system_paths = []
    
    # Get Python version
    python_version = subprocess.run(
        ["/usr/bin/python3", "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True
    ).stdout.strip()
    
    # Common base directories for Python packages
    base_dirs = [
        "/usr/lib/python3",
        "/usr/local/lib/python3",
        f"/usr/lib/python{python_version}",
        f"/usr/local/lib/python{python_version}"
    ]
    
    # Common package directories
    package_dirs = [
        "dist-packages",
        "site-packages"
    ]
    
    # Find all existing paths
    for base in base_dirs:
        for package_dir in package_dirs:
            path = os.path.join(base, package_dir)
            if os.path.exists(path):
                system_paths.append(path)
                
        # Also check for direct dist-packages in python3 directory
        if os.path.exists(base) and os.path.isdir(base):
            system_paths.append(base)
            
    return list(set(system_paths))  # Remove duplicates

def check_if_calibre_is_installed():
    calibre_installed = shutil.which("calibre") or shutil.which("ebook-convert")
    
    if calibre_installed:
        return True
    else:
        return False
    
def check_if_ffmpeg_is_installed():
    ffmpeg_installed = shutil.which("ffmpeg")
    
    if ffmpeg_installed:
        return True
    else:
        return False

def run_shell_command_without_virtualenv(command):
    # Get the original PYTHONPATH
    original_pythonpath = os.environ.get('PYTHONPATH', '')
    
    try:
        # Temporarily modify the environment
        modified_env = os.environ.copy()
        
        # Get system Python paths automatically
        system_paths = get_system_python_paths()
        
        if not system_paths:
            raise Exception("No system Python paths found")
            
        modified_env['PYTHONPATH'] = ':'.join(system_paths + [original_pythonpath])
        
        # Run the command with modified environment
        cmd = f"/usr/bin/python3 {command}"
        result = subprocess.run(
            cmd,
            shell=True,
            env=modified_env,
            capture_output=True,
            text=True
        )

        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None