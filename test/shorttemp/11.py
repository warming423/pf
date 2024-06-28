建立一个json文件，配置profiler

def get_profiler(config_path):
    try:
        with open(config_path, 'r') as filehandle:
            config_dict = json.load(filehandle)
    except Exception as e:
        print(f'Load config file for profiler error: {e}')
        print('Use default parameters instead.')
        return Profiler()
    
    translated_config_dict = {}
    if "targets" in config_dict:
        try:
            translated_config_dict['targets'] = []
            for target in config_dict['targets']:
                if target.lower() == "cpu":
                    translated_config_dict['targets'].append(ProfilerTarget.CPU)
                elif target.lower() == 'gpu':
                    translated_config_dict['targets'].append(ProfilerTarget.GPU)
        except:
            print('Set targets parameter error, use default parameter instead.')
            translated_config_dict['targets'] = None
    if "scheduler" in config_dict:
        try:
            if isinstance(config_dict['scheduler'], dict):
                for key, value in config_dict['scheduler'].items():
                    module_path = value['module']
                    use_direct = value['use_direct']
                    module = importlib.import_module(module_path)
                    method = getattr(module, key)
                    if not use_direct:
                        translated_config_dict['scheduler'] = method(
                            *value['args'], **value['kwargs']
                        )
                    else:
                        translated_config_dict['scheduler'] = method
            else:
                translated_config_dict['scheduler'] = [
                    config_dict['scheduler'][0],
                    config_dict['scheduler'][1],
                ]

        except:
            print(
                'Set scheduler parameter error, use default parameter instead.'
            )
            translated_config_dict['scheduler'] = None
    if "on_trace_ready" in config_dict:
        try:
            if isinstance(config_dict['on_trace_ready'], dict):
                for key, value in config_dict['on_trace_ready'].items():
                    module_path = value['module']
                    use_direct = value['use_direct']
                    module = importlib.import_module(module_path)
                    method = getattr(module, key)
                    if not use_direct:
                        translated_config_dict['on_trace_ready'] = method(
                            *value['args'], **value['kwargs']
                        )
                    else:
                        translated_config_dict['on_trace_ready'] = method
        except:
            print(
                'Set on_trace_ready parameter error, use default parameter instead.'
            )
            translated_config_dict['on_trace_ready'] = None
    if "timer_only" in config_dict:
        if isinstance(config_dict['timer_only'], bool):
            translated_config_dict['timer_only'] = config_dict['timer_only']
        else:
            print(
                'Set timer_only parameter error, use default parameter instead.'
            )

    return Profiler(**translated_config_dict)
