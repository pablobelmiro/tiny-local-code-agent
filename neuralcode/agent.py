import argparse
import os
from pathlib import Path

from . import budget
from . import commands
from . import compact
from . import history
from . import session
from . import workdir
from .context import reminder
from .llm import system_prompt, call_llm
from . import sandbox
from .todos import active_form
from .tools import execute
from .ui import ui


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="continue the last session")
    parser.add_argument("--debug", action="store_true", help="show the raw model response")
    cli = parser.parse_args()

    repo_root = Path.cwd()
    session_dir = workdir.choose_or_create(repo_root, ui)
    os.chdir(session_dir)

    ui.banner(sandbox.name())

    messages = [{"role": "system", "content": system_prompt()}]
    if cli.resume:
        saved = session.all_sessions()
        if saved:
            messages = session.open_session(saved[0]["id"])
            history.strip(messages)
            ui.resumed(messages)
            ui.replay(messages)

    while True:
        user_input = ui.ask()
        if not user_input:
            break

        if user_input.startswith("/"):
            messages = commands.handle(user_input, messages)
            session.save(messages)
            continue

        messages.append({"role": "user", "content": user_input})
        turn_user_message = messages[-1]

        while True:
            if budget.should_warn():
                confirmed = ui.budget_warning(budget.total(), budget.limit())
                if not confirmed:
                    break
                messages = session.reset_context(system_prompt(), carry=[turn_user_message])
                budget.reset()
                ui.context_reset()

            injection = reminder()
            ui.injection(injection["content"])

            if history.fit(messages):
                ui.note("dropped old tool output to make this request fit")

            with ui.working(active_form()):
                message, usage = call_llm(messages + [injection])

            messages.append(message.model_dump(exclude_none=True))
            session.save(messages)
            budget.track(usage)
            ui.usage(usage, budget_remaining=budget.remaining(), budget_total=budget.limit())

            if cli.debug:
                ui.debug(message.model_dump(exclude_none=True))

            if message.content:
                ui.agent(message.content)

            if not message.tool_calls:
                break

            for tool_call in message.tool_calls:
                args, result = execute(tool_call)
                ui.tool(tool_call.function.name, args, result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                session.save(messages)

        history.sweep()   # the turn is over: bin its temp files
        history.strip(messages)  # ...and shrink the tool output it produced

        if compact.needed(usage):
            messages = commands.compact(messages)

    ui.summary()


if __name__ == "__main__":
    main()
