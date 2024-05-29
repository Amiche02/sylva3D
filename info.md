Ubuntu :
```bash
sudo apt-get install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

sudo apt-get update
sudo apt-get install build-essential ninja-build


pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

colab :
```bash
curl -L "https://github.com/OutofAi/tiny-cuda-nn-wheels/releases/download/1.7.0/tinycudann-1.7+arch75+torch221+cuda121-cp310-cp310-linux_x86_64.whl" -o tinycudann-1.7+arch75+torch221+cuda121-cp310-cp310-linux_x86_64.whl
pip install tinycudann-1.7+arch75+torch221+cuda121-cp310-cp310-linux_x86_64.whl --force-reinstall
import tinycudann as tcnn
```

uninstall old libraries : 
```bash
chmod +x uninstall_requirements.sh
./uninstall_requirements.sh
```

install the right version :
```bash
pip install -r requirements.txt
pip install fire streamlit
```

The usage of the script would look like this:
```sh
python inference.py --input_image /root/sylva3D/example_images/cat_head.png --output_dir /root/sylva3D/outputs --guidance_scale 2.0 --steps 100 --seed 42 --write_image
```

3D reconstruction :
```sh
python inference_recon.py --input_image /root/sylva3D/example_images/cat_head.png --output_dir /root/sylva3D/outputs --guidance_scale 2.0 --steps 100 --seed 42 --write_image
```