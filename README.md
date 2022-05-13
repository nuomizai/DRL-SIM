# DRL-SIM
This is the code accompanying the paper: "Social-Aware Incentive Mechanism for Vehicular Crowdsensing by Deep Reinforcement Learning" by Yinuo Zhao and Chi Harold Liu, published at TITS. 

## Description
This simplified code implements a DRL-based social-aware incentive mechanism to solve the optimal sensing strategy for all vehicles in vehicular crowdsensing. 

## Dependencies

You just need to install **torch**, numpy, random, csv, time, json, argparse by pip or conda

## Usage

To generate E-R social graph, first you need to config the variable `mu` and `user_number` in `generate_data.py`. And then run the following command by

```
python generate_data.py
```

Then, copy the value of E-R social graph into `self.V['relationship']` in `env_setting.py`, and config other environment parameters there. 

After that, run the training and testing process by

```
python train.py --root-path [PATH to where to save results file and model] --user-num [USER NUMBER]
```

Finally, find the training and testing results under `--root-path`

## Contact
If you have any question, please email `ynzhao@bit.edu.cn`.

## Paper

If you are interested in our work, please cite our paper as 
```
@article{zhao2020social,
  title={Social-aware incentive mechanism for vehicular crowdsensing by deep reinforcement learning},
  author={Zhao, Yinuo and Liu, Chi Harold},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={22},
  number={4},
  pages={2314--2325},
  year={2020},
  publisher={IEEE}
}
```
