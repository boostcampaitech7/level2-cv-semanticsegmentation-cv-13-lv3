pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0

apt install libgl1-mesa-glx
apt-get install libglib2.0-0

cd mmsegmentation
pip install -v -e .

pip install ftfy
pip install regex

mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
python demo/image_demo.py demo/demo.png pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
