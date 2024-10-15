# Create a new conda environment
conda create -n dreamwaltz python=3.11

# Install with conda
## CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 cuda==11.8 -c pytorch -c nvidia/label/cuda-11.8.0
conda install fvcore -c conda-forge  # required by pytorch3d
conda install pytorch3d=0.7.5=py311_cu118_pyt210 -c pytorch3d
## CUDA 12.1
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 cuda==12.1 -c pytorch -c nvidia
# conda install -c conda-forge fvcore
# conda install pytorch3d=0.7.5=py311_cu121_pyt210 -c pytorch3d

conda install ninja git-lfs

# Install with pip
pip install scikit-image matplotlib imageio plotly open3d trimesh pyrender av decord
pip install mediapipe accelerate xatlas libigl
pip install pyrallis loguru omegaconf plyfile jaxtyping

pip install -U "huggingface_hub[cli]"

pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install git+https://github.com/vchoutas/smplx.git
pip install git+https://github.com/nghorbani/human_body_prior.git
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/diffusers.git
pip install controlnet-aux==0.0.7

pip install git+https://github.com/ashawkey/diff-gaussian-rasterization.git
pip install git+https://github.com/graphdeco-inria/gaussian-splatting.git#subdirectory=submodules/simple-knn
