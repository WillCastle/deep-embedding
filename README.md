# deep-embedding

1. Create env:
   conda env create -f environment.yml

2. Activate environment:
   conda activate de

3. Install PyTorch with GPU enabled:
   conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

4. Install pip requirements
   pip install -r requirements.txt

Remove env:
conda deactivate
conda remove --name de --all

Update environment:
conda env update -f environment.yml
