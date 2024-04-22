import asyncio
import time

import openai
import os

import pandas as pd

from src.util import process_batch
import json

import click


@click.command()
@click.option('--result_file', type=click.Path(dir_okay=False, exists=True))
@click.option('--batch_size', type=int, default=8)
def scoring(result_file, batch_size: int):
    outputs = []
    with open(result_file, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            outputs.append(json_obj['content'])

    assert os.getenv('OPENAI_API_KEY', None) is not None, "OPENAI_API_KEY env variable must be set to use .env file"

    client = openai.AsyncClient(api_key=os.environ['OPENAI_API_KEY'])

    openai_inputs = list(map(lambda output: score_prmt(output), outputs))

    start_time = time.time()
    print(f"Start scoring with openai GPT-4 model")
    tasks = [get_response(client, pmt) for pmt in openai_inputs]
    loop = asyncio.get_event_loop()
    s = loop.run_until_complete(process_batch(tasks, batch_size))
    end_time = time.time()

    print(f"GPT-4 scoring time: {end_time - start_time:.3f}초 걸림")

    df['score_raw'] = s

    # 문자가 있으면 다 날림
    df['score'] = df['score_raw'].str.replace('[^0-9]', '', regex=True)

    df['score'] = pd.to_numeric(df['score'])

    display_score(df)

    return df


async def get_response(openai_client, pmt):
    response = await openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{'role': 'user', 'content': pmt}])

    output = response.choices[0].message.content

    return output


# 프롬프트화
def score_prmt(pmt):
    query = f"당신은 질문과 그에 관한 답변을 보고 답변에 대한 점수를 메기는 채점용 인공지능 도구입니다.\n다음 질문과 답변을 보고 점수만 답하되, 0점에서 10점사이로 정수형 숫자로만 답하세요.\n예시:10\n\
        {pmt}\n\n점수:"

    return query


# score 디스플레이
def display_score(df):
    mean = df['score'].mean()
    min_value = df['score'].min()
    max_value = df['score'].max()
    std_dev = df['score'].std()

    if mean > 10:
        print('Warning :중간에 이상한 숫자가 있으므로, 점수가 불완전합니다. 수동으로 확인해주세요.')

    else:
        print("평균:", mean)
        print("최소값:", min_value)
        print("최대값:", max_value)
        print("표준편차:", std_dev)


if __name__ == '__main__':
    scoring()
