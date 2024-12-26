
def set_seed(seed):
    import torch, numpy, random, os
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

if __name__=="__main__":
    # example
    model_config, train_config, eval_config, data_config = read_config("configs.config")

    print("train_config", train_config.__dict__)
    print("eval_config", eval_config.__dict__)
    print("data_config", data_config.__dict__)
