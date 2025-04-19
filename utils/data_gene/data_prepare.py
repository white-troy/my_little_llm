import json
import os
from tqdm import tqdm

def get_text_from_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ori_text = json.loads(line)
                    data.append(ori_text['text'])
                except json.JSONDecodeError as e:
                    print(f"跳过无效行 (错误: {e})")
    return data

def prepare_text_data(datas):
    final_data = []
    for data in datas:
        if len(data) > 1:
            for item in data:
                if len(item) > 200:
                    json_obj = {"text": item.replace('\n\n','\n')}
                    final_data.append(json_obj)
        else:
            json_obj = {"text": data.replace('\n\n','\n')}
            final_data.append(json_obj)

    json_content = "\n".join(json.dumps(item, ensure_ascii=False) for item in final_data)

    with open('output_data.jsonl','a',encoding='utf-8') as f:
        f.write(json_content)

def merge_file(file_path):

    for dir in tqdm(os.listdir(file_path)):
        dir_path = os.path.join(file_path,dir)
        if os.path.isfile(dir_path):
            # print(dir_path)
            ori_json_datas = []
            ori_json_data = get_text_from_jsonl(dir_path)
            ori_json_datas.append(ori_json_data)
            prepare_text_data(ori_json_datas)
        else:
            merge_file(dir_path)


if __name__ == "__main__":
    dir_path = r'D:\python\pythonpj\datasets\llm\wiki_zh_2019\wiki_zh'
    merge_file(dir_path)