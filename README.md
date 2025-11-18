# FROM STATIC TO DYNAMIC: ADAPTIVE MONTECARLO SEARCH FOR MATHEMATICAL PROCESS SUPERVISION

This project is the official implementation of the paper **"FROM STATIC TO DYNAMIC: ADAPTIVE MONTE CARLO SEARCH FOR MATHEMATICAL PROCESS SUPERVISION"** .

Large scale language models (LLMs) still face challenges when dealing with complex multi-step mathematical reasoning problems. Process Reward Models (PRMs) have been proven to be an effective way to enhance the reasoning ability of models by supervising each step of the reasoning process. However, obtaining high-quality process supervision data is the main bottleneck for training PRM. Existing methods typically rely on fixed budget sampling strategies, which are inefficient and lack flexibility in large search spaces.

To address these issues, we propose the Adaptive Monte Carlo Search (AMCS) framework. AMCS fundamentally improves the process of generating process supervision data:
1.  **Uncertainty driven adaptive sampling:** AMCS can dynamically allocate more computing resources (samples) to inference steps with high uncertainty, while reducing sampling of simple and high certainty steps, thereby significantly improving the efficiency of data annotation.
2.  **Dynamic Exploration and Utilization Strategy:** AMCS uses Monte Carlo Tree Search (MCTS) to explore inference paths, which smoothly transitions from extensive exploration in the early stages to deep utilization in the later stages, thereby more effectively discovering high-quality inference paths and locating erroneous steps.

Based on AMCS, we constructed a high-quality process supervision dataset containing approximately 200000 samples and trained AMCS-PRM. Experimental results have shown that our method achieves the current best performance on multiple mathematical inference benchmarks such as MATH, AIME, Olympiad Bench, etc. This demonstrates the significant value of high-quality process supervision in enhancing model capabilities.

## 🛠️ Project Structure

```
AMCS/
├── adaptive_omegaprm/                #data generation    
│   ├── grader.py                     
│   ├── llm_utils.py                  
│   ├── omegaprm.py                   #Core data generation logic
│   ├── process_json.py
│   ├── run_omegaprm.py          
│   └── run_omegaprm_multi_gpu.sh               
├── envs/                             #Data environment
│   ├── MATH/            
│   ├── tests/    
│   ├── init.py        
│   └── base_env.py       
├── reason/                           #Core reasoning logic
│   ├── evaluation/ 
│   ├── guided_search/ 
│   ├── inference/ 
│   ├── llm_service/ 
│   └── reranking/
├── scripts/eval/
│   ├── beam_search.sh      
│   ├── cot_greedy.sh  
│   ├── cot_rerank.sh  
│   └── vanila_mcts.sh
├── README.md
└── requirements.txt                   # Python dependencies   
```



## 🚀Getting Start

### 1.Create a virtual environment

```bash
conda create -n amcs python=3.10
conda activate amcs
```
### 2.Install dependencies

```bash
pip install -r requirements.txt
```
### 3. Generation of MathSearch-200K Datasets

Run `adaptive_omegaprm/process.py`  to split the original data.

```
python adaptive_omegaprm/process.py
```

Then,use `adaptive_omegaprm/run_omegaprm.py`  to generate process supervision data.

```bash
# Run data generation on multiple GPUs
bash adaptive_omegaprm/run_omegaprm_multi_gpu.sh
```

### 4. Process Reward Model Training

Use `gen_rm/fine_tuning.py` to train a process reward model.

```
python gen_rm/fine_tuning.py \
--model_name_or_path "path to your model" \ 
--train_file "path/to/your/generated_data.jsonl" \
--output_dir "./models/amcs-prm" \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \# ... other training parameters
```

### 5. Inference-time Verification

#### Start Services

Before running inference, please modify the following variables in the scripts under `reason/llm_service/` .

- `$POLICY_MODEL_NAME`: Set this to the name of the policy model you wish to use.
- `$VALUE_MODEL_NAME`: Set this to the name of the value model you wish to use.

To start the model service,run:

```
sh reason/llm_service/create_service_qwen2.5_math_vllm.sh
```

#### Run Inference

Make sure the parameters (`--LM`, `--RM`) in the script aligns with the variables (`$POLICY_MODEL_NAME`, `$VALUE_MODEL_NAME`).

```
export PYTHONPATH=$(pwd)
sh scripts/eval/cot_rerank.sh
```

## 📑Citation

If you find our work helpful for your research, please consider citing our paper:

```
@misc{ma2025staticdynamicadaptivemonte,
      title={From Static to Dynamic: Adaptive Monte Carlo Search for Mathematical Process Supervision}, 
      author={Jie Ma and Shihao Qi and Rui Xing and Ziang Yin and Bifan Wei and Jun Liu and Tongliang Liu},
      year={2025},
      eprint={2509.24351},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.24351}, 
}
```

