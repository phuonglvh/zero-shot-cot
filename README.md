This is the official implementation of `Large Language Models are Zero-Shot Reasoners` .

## Installation
Make sure you have Python>=3.8 installed on your machine.
```
pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt
```

## Set your OpenAI API key
```
# https://beta.openai.com/account/api-keys
export OPENAI_API_KEY=(YOUR OPENAI API KEY)
```

## Set arguments.
```
model=gpt3-xl # {"gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"}. "gpt3" is the smallest model.
dataset=multiarith # We can use other datasets. See help for the details.
```

## Let's get started !

### Zero-shot-CoT (our proposal)
```
python main.py --method=zero_shot_cot --model=${model} --dataset=${dataset}
```

### Zero-shot
```
python main.py --method=zero_shot --model=${model} --dataset=${dataset}
```

### Few-shot-CoT
```
# Only MultiArith and GSM8K are supported ...
python main.py --method=few_shot_cot --model=${model} --dataset=${dataset}
```

### Few-shot
```
# Only MultiArith and GSM8K are supported ...
python main.py --method=few_shot --model=${model} --dataset=${dataset}
```