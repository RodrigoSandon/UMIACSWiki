from importlib import reload
import ragtest2 as module


cache = None
while True:
    if not cache:
        cache = module.load()
    module.run(cache)
    if input("rerun? (anything/n): ") == "n":
        break
    reload(module)
