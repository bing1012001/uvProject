How to use uv tool
1. download and install: 
    brew install uv
2. mkdir uvProject && cd uvProject
3. uv init (OR uv init uvProject --bare to skip the git repo creation)
4. Create the git repo and doing the repo git link
5. uv add requests (same as pip install requests, and also update the pyproject.toml and uv.lock)
6. uv sync 


How to link with VSCode
1. create sub file .vscode/settings.json (Ctrl/Cmd+Shift+P choose the intepretor also can create this file)
2. update the .vscode/launch.json to update the run / debug 
