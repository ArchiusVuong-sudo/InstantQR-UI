from core.model import PlaywrightSession
from agents import function_tool, RunContextWrapper

@function_tool
async def run_script(
    wrapper: RunContextWrapper[PlaywrightSession],
    script_path: str
):
    """Use to run a script and return its result

    Args:
        script_path: The path to the script
    """
    print(f"- Running script: {script_path}")
    page = wrapper.context.page
    with open(script_path, "r", encoding="utf-8") as f:
        script = f.read()
    result = await page.evaluate(script)

    return result
