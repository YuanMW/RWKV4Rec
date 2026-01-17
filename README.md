## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/YuanMW/RWKV4Rec.git
cd RWKV4Rec
pip install -r requirements.txt

# Train RWKV4Rec model
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --use_lora=False --model='RWKV4Rec' --device=cuda
python main.py --dataset=Video --train_dir=default --maxlen=50 --use_lora=True  --model='RWKV4Rec' --device=cuda
```


## 📁 Project Structure
```bash
RWKV4Rec/
├── README.md
├── main.py
├── a.ipynb
├── data/
│ ├── Beauty.txt
│ ├── ml-1m.txt
│ ├── Steam.txt
│ ├── Video.txt
├── models/
│ ├── baseline/
│ │ ├── BERT4Rec.py
│ │ ├── BSARec.py
│ │ ├── CL4SRec.py
│ │ ├── DuoRec.py
│ │ ├── FEARec.py
│ │ ├── GRU4Rec.py
│ │ ├── MAERec.py
│ │ ├── SASRec.py
│ ├── RWKV4Rec.py
├── requirements.txt
├── utils.py
```