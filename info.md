Ubuntu :

sudo apt-get install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

sudo apt-get update
sudo apt-get install build-essential ninja-build


pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch


colab :

curl -L "https://github.com/OutofAi/tiny-cuda-nn-wheels/releases/download/1.7.0/tinycudann-1.7+arch75+torch221+cuda121-cp310-cp310-linux_x86_64.whl" -o tinycudann-1.7+arch75+torch221+cuda121-cp310-cp310-linux_x86_64.whl
pip install tinycudann-1.7+arch75+torch221+cuda121-cp310-cp310-linux_x86_64.whl --force-reinstall
import tinycudann as tcnn


pip uninstall -y jax jaxlib diffusers torch tensorflow 

pip install -r requirements.txt
pip install fire streamlit
