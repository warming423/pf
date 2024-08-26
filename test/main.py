import re
import json
import os


def process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for event in data.get("traceEvents", []):
        # 检查并处理 "tid" 字段
        if 'tid' in event and isinstance(event['tid'], str):
            tid_str = re.sub(r'\(.*?\)', '', event['tid']).strip()
            if tid_str.isdigit():  # 检查结果是否为数字
                event['tid'] = int(tid_str)  # 转换为整数类型

        # 检查并处理 "name" 字段
        if 'name' in event and isinstance(event['name'], str):
            # 使用正则表达式，仅删除 "name" 字段中带有时间格式的中括号
            event['name'] = re.sub(r'\s*\[\d+(\.\d+)?\s*(us|ms)\]', '', event['name']).strip()

    # 检查并删除尾部的空字典对象 "{}"
    if data["traceEvents"] and isinstance(data["traceEvents"][-1], dict) and not data["traceEvents"][-1]:
        data["traceEvents"].pop()

    new_file_path = re.sub(r'pfdata', 'pt.trace', file_path)
    with open(new_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

    print(f'Processed file saved as {new_file_path}')

process_json(r'/home/wangm/pf/data/aisec-dell-server_502315.1724139770635.pfdata.json')
