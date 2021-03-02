import os
import importlib
import site
import inspect


def loaded_module_from_dotpath_and_filepath(dotpath, filepath):
    """Get module object from file path and module dotpath"""
    module_spec = importlib.util.spec_from_file_location(dotpath, filepath)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def module_path(module):
    try:
        _module_path = module.__path__[0]
    except:
        _module_path = inspect.getfile(module)
        if _module_path.endswith('__init__.py'):
            _module_path = module_path[:len('__init__.py')]
    return _module_path


def submodules(module, on_error='print', prefix=None):
    from py2store.filesys import FileCollection

    prefix = prefix or site.getsitepackages()[0]

    _module_path = module_path(module)
    assert os.path.isdir(_module_path), f"Should be a directory: {module_path}"

    for filepath in FileCollection(_module_path, '{f}.py', max_levels=0):
        dotpath = '.'.join(filepath[len(prefix):(-len('.py'))].split(os.path.sep))
        if dotpath.startswith('.'):
            dotpath = dotpath[1:]
        if not dotpath.endswith('__init__'):
            try:
                yield loaded_module_from_dotpath_and_filepath(dotpath, filepath)
            except Exception as e:
                if on_error == 'print':
                    print(f"{e} ({dotpath}: {filepath})")
                elif on_error == 'raise':
                    raise


def objects_of_module(module, max_levels=0):
    for a in dir(module):
        if not a.startswith('_'):
            yield getattr(module, a)
    if os.path.isdir(module_path(module)):
        if max_levels > 0:
            for submodule in submodules(module):
                yield from objects_of_module(submodule, max_levels - 1)


def obj_to_dotpath(obj):
    return f"{obj.__module__}.{obj.__name__}"


def finding_objects_of_module_with_given_methods(module, method_names=None, max_levels=1):
    module_dotpath = module.__name__

    objects = {obj_to_dotpath(obj): obj for obj in
               filter(lambda o: isinstance(o, type) and o.__module__.startswith(module_dotpath),
                      objects_of_module(module, max_levels))}

    if method_names is None:
        return objects
    else:
        if isinstance(method_names, str):
            method_names = [method_names]
        method_names = set(method_names)

        return {dotpath: obj for dotpath, obj in objects.items() if method_names.issubset(dir(obj))}
