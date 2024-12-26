import os
import sys
import importlib
import importlib.util

print(os.getcwd())


# current_cwd = os.getcwd()

# if current_cwd.endswith("src"):
#     # remove "/src" to point to the project directory
#     current_cwd = current_cwd[:-4] 
# sys.path.insert(1, current_cwd)

# print(sys.path)

# config = importlib.import_module("configs.configs.configs_cola_baseline")
# breakpoint()


# import importlib.util
# import sys

# # Absolute path to your module
# module_path = "/path/to/your/module.py"

# # Create a module spec
# spec = importlib.util.spec_from_file_location("module_name", module_path)

# # Load the module
# module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(module)

# # Now you can use the module
# module.my_function()  # Call a function from the module