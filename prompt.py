from llm import llm
from rich.status import Status
from functools import wraps
from rich.console import Console


console = Console()


def notify_entry(func):
    @wraps(func) # 它使用@wraps(func)来确保装饰器不会改变原函数的名称和文档字符串等元数据
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        with Status(f"[bold green]Entering[/bold green] function: [bold yellow]{func_name}[/bold yellow]", console=console):
            result = func(*args, **kwargs)
        return result
    return wrapper


@notify_entry
def improve(prompt):
    prompt = ("A prompt written by human is as follows.\n\n"
        "<PROMPT>\n"
        f"{prompt}\n\n"
        "</PROMPT>\n"
        "Please provide me an improved one. Be precise. Do not add additional information. Your revised prompt should start with: <START>"
    )
    ans = llm(prompt)
    return ans.split("<START>")[1]


@notify_entry
def answer(instruction, ctx_fn):
    # context=open(ctx_fn, 'r').read()
    context = open(ctx_fn, 'r', encoding='utf-8').read()
    prompt = (
        "Answer this instruction:\n\n"
        "<INSTRUCTION>\n"
        f"{instruction}\n"
        "</INSTRUCTION>\n\n"
        "given solely on the following context\n\n"
        "<CONTEXT>\n"
        f"{context}\n"
        "</CONTEXT>\n"
    )
    return llm(prompt)


@notify_entry
def question(ctx_fn):
    # context = open(ctx_fn, 'r').read()
    context = open(ctx_fn, 'r', encoding='utf-8').read()

    prompt = (
        # "I am crafting instruction-following data from a python package called \"sionna\"."
        # "Help me crafting some instructions (user questions) about sionna given the following context. "
        # "Because I need to train a model to write code related to Sionna, besides regular questions to context. I need you to pose some questions to form a dataset that can teach the model how to write Sionna code!"
        # "Each instruction should be start with \"INSTRUCTION:\".\n\n"
        # "<CONTEXT>\n"
        # f"{context}\n"
        # "</CONTEXT>\n\n"
        "I'm building a dataset for model training focused on the \"sionna\" Python package, based on the given following context. Cover the context as much as you can."
        "Your role is to generate clear, concise instructions (as user questions) that will guide the model in mastering Sionna coding."
        "Start each instruction with \"INSTRUCTION:\" and tailor it to fit the provided context, ensuring to cover as much of the context's information as possible for comprehensive learning.\n\n"
        "<CONTEXT>\n"
        f"{context}\n"
        "</CONTEXT>\n\n"
        "These instructions are crucial for teaching the model to effectively understand and apply Sionna's code and APIs, tailored to real-world programming scenarios."
    )
    ans = llm(prompt)
    # for line in ans.split("\n"):
    #     if line.strip().startswith("INSTRUCTION:"):
    #         yield line.split("INSTRUCTION:")[1]
    return ans


@notify_entry
def reduce(qs):
    prompt = (
        "Several questions are generated as follows:\n"
        f"{qs}\n"
        "These questions may be duplicated or off-the-topic of sionna. Please reduce the candidates.\n"
        "Return me the instructions with each line start with \"INSTRUCTION:\".\n\n"
    )
    ans = llm(prompt)
    for line in ans.split("\n"):
        if line.strip().startswith("INSTRUCTION:"):
            yield line.split("INSTRUCTION:")[1]