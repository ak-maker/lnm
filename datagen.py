import typer
from typing_extensions import Annotated
from config import cfg
from prompt import improve, answer, question, reduce
import sys
from rich.progress import Progress, SpinnerColumn, TextColumn
import time
from rich import print


def gen(ctx_fn, to_fn=""):
    with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
        task = progress.add_task("[cyan]Generating instruction-following data...")
        writer = open(to_fn, 'a', encoding='utf-8') if to_fn else sys.stdout # 如果 to_fn 有值（即指定了输出文件名），
        # 则执行前面的 open(to_fn, 'w') 部分。如果 to_fn 为空（没有指定输出文件名），
        # 则使用 sys.stdout 作为 writer 的值。sys.stdout 代表标准输出流，通常是终端（控制台）。
        for q in reduce(question(ctx_fn)):
            ans = answer(q, ctx_fn)
            writer.write(f"INSTRUCTION:{q}\n")
            writer.write(f"ANSWER:{ans}\n")
            writer.write("\n")
            writer.flush()
            progress.update(task, advance=1)  # This will just spin without a total
    print("Processing complete.")

start_time = time.time()
gen("context.txt", "data_gen.txt")
end_time = time.time()
print(end_time-start_time)