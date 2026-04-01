from hoshicore.engine.wiring import run_from_yaml
import asyncio

global_inputs = {
    "fnames": []
}
global_configs = {
    "fin": 0.2,
    "fout": 0.2,
    "int_weight": True,
}

asyncio.run(
    run_from_yaml("./hoshicore/dag/simple_startrail.yaml", global_inputs,
                  global_configs))
