import os
from args import get_runtime_args

parser = get_runtime_args()
args = parser.parse_args()

print(f"PWD: {os.getcwd()}")

print("Arguments:")

for arg in vars(args):
    print(f"{arg}:      {getattr(args, arg)}")

with open(f"{args.saving_path}/test.txt", "w+") as f:
    f.write("Hello, World!")