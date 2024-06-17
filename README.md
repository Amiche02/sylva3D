# Installation

### Clone the project repository
```bash
git clone -b visfusion https://github.com/Amiche02/sylva3D.git
cd VisFusion
```

### Create project environment
```bash
conda create --name visfusion python=3.9 -y
conda activate visfusion
```

### Install dependencies
```bash
cd env
pip install Cython
pip install -r torch_requirements.txt
```

### Add CUDA to the PATH
```bash	
nano ~/.bashrc
```

Add the following lines to the bottom of the file:
```bash	
# Add CUDA to PATH
if [ -d "/usr/local/cuda/bin" ]; then
  export PATH=/usr/local/cuda/bin:$PATH
fi
if [ -d "/usr/local/cuda/lib64" ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi
```

Reload the bash configuration and verify:
```bash
source ~/.bashrc
conda activate visfusion
echo $LD_LIBRARY_PATH
```

### Install remaining dependencies
```bash
pip install -r requirements.txt
```

### Verify installation
You can verify that everything is correctly installed by running the following Python script:
```bash
python3 check_gpu.py
```
