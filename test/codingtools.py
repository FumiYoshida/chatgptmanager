import importlib.util
import sys
from pathlib import Path

def create_module_from_path(module_path):
    module_path = Path(module_path).absolute().resolve()
    if module_path.name == "__init__.py":
        module_name = module_path.parent.stem + ".py" # モジュール名に"."がついていないと相対インポートでエラー
    else:
        module_name = module_path.stem + ".py"

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None, "load failed"

    new_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = new_module
    spec.loader.exec_module(new_module)

    return new_module

def create_module_from_dir(base_dir):
    base_dir = Path(base_dir)
    assert base_dir.is_dir()
    
    init_path = base_dir / "__init__.py"
    if init_path.exists():
        new_module = create_module_from_path(init_path)
        
    for item in base_dir.iterdir():
        if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
            module = create_module_from_path(item)
            setattr(new_module, item.stem, module)
        elif item.is_dir() and (item / "__init__.py").exists():
            module = create_module_from_path(item / "__init__.py")
            setattr(new_module, item.stem, module)
            
    return new_module