# fintech
## virtualenv
### run_venv.sh

```
move to fintech directory:
cd ~/my/fintech/
```
> #### train data
> ```
> ./scripts/run_venv.sh python train.py
> ./scripts/run_venv.sh python train_script.py
> ```

> #### evalution model
> ```
> ./scripts/run_venv.sh python evaluation_script.py
> ```

> #### tensorboard
> ```
> # Default port: 6006
> ./scripts/run_venv.sh tensorboard --logdir=./tensorboard/['PPO2', 'A2C', 'ACKTR-PPO', 'ACKTR-A2C']/MlpPolicy/[Data_Name]/ --host=0.0.0.0
> example :
> ./scripts/run_venv.sh tensorboard --logdir=./tensorboard/A2C/MlpPolicy/BIV/ --host=0.0.0.0
> ```

> #### clean
> ```
> # delete train results
> ./scripts/clean.sh
> ```