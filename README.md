# LS-BERT Notebook
The LS-BERT project can use network languages produced by HiCKUP to train a BERT model. Everything relevant is included in the jupyter notebook ```LS-BERT.ipynb```.

## Installation
For installing torch with cuda to run on all nodes of the cluster:

```
pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Do this before installin the requirements file with:
```
pip install -r requirements.txt
```

## Running Jupyter Notebook on cluster
LS-BERT fits into VRAM of all nodes with medium-sized model configurations. For connecting jupyter notebook to a node, first run the ```runner.sh``` script with ```sbatch runner.sh```.

Next, initiate the ssh tunnel with the appropriate port (see in ```runner.sh``` file), for example port 9998 on node 1 (A100 GPUs).
```
ssh -L 9998:localhost:9998 node01
```

Connect to the jupyter notebook with the website shown in the **err** log file.