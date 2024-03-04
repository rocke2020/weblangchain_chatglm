import argparse


class ArgparseUtil(object):
    """
    参数解析工具类
    """
    def __init__(self):
        """comm args"""
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--seed", default=2, type=int)

    def task(self):
        """task args"""
        self.parser.add_argument("--gpu_id", default=0, type=int, help="the GPU NO.")        
        self.parser.add_argument("--task", type=str, default="", help="")
        self.parser.add_argument("--input_root_dir", type=str, default="", help="")
        self.parser.add_argument("--out_root_dir", type=str, default="", help="")
        args = self.parser.parse_args()
        return args
