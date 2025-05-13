import re
with open("route_handlers.py.bak", "r") as f:
    content = f.read()
fix1 = """                    # Convert numpy data types to Python native types before inserting